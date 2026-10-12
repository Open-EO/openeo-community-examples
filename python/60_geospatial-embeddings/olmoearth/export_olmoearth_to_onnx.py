"""Offline export of joint 12-month OlmoEarth Nano S1/S2 annual embeddings.

Based on Ai2's OlmoEarth encoder. Artifacts and derivatives retain the use
restrictions in README.md#licensing and the OlmoEarth Artifact License:
https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/blob/main/LICENSE.txt
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from huggingface_hub import hf_hub_download
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path
from olmoearth_pretrain.nn.flexi_vit import Encoder

from openeo_olmoearth import INPUT_BANDS, MODEL_NAME, OUTPUT_BANDS, S2_BANDS

ROOT = Path(__file__).resolve().parent
SOURCE_REPO = "allenai/OlmoEarth-v1_2-Nano"
SOURCE_REVISION = "e1f693ae2a7d5b57871a978e9d09e22d05206747"
SOURCE_SHA256 = "2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378"


def load_pretrained_model() -> torch.nn.Module:
    """Download only the pinned config/weights and verify the trusted source weights."""
    config = hf_hub_download(SOURCE_REPO, "config.json", revision=SOURCE_REVISION)
    weights = hf_hub_download(SOURCE_REPO, "weights.pth", revision=SOURCE_REVISION)
    if hashlib.sha256(Path(weights).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("OlmoEarth source weights failed SHA-256 verification")
    return load_model_from_path(Path(config).parent).eval()


class AnnualPatchEmbeddings(torch.nn.Module):
    """Month-major raster -> jointly attended, valid-only annual BCHW embeddings."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        assert not encoder.use_flash_attn and not encoder.has_register_tokens
        assert encoder.embedding_projector is None and encoder.embedding_size == 128
        assert S2_BANDS == list(Modality.SENTINEL2_L2A.band_order)
        assert all(encoder.tokenization_config.get_num_bandsets(m.name) == 1 for m in
                   [Modality.SENTINEL2_L2A, Modality.SENTINEL1])
        self.encoder = encoder
        configs = Normalizer(Strategy.COMPUTED).norm_config
        for modality in [Modality.SENTINEL2_L2A, Modality.SENTINEL1]:
            stats = [configs[modality.name][b] for b in modality.band_order]
            mean = torch.tensor([s["mean"] for s in stats], dtype=torch.float64)
            std = torch.tensor([s["std"] for s in stats], dtype=torch.float64)
            minimum, maximum = mean - 2 * std, mean + 2 * std
            self.register_buffer(f"{modality.name}_minimum", minimum.reshape(1, 1, -1, 1, 1))
            self.register_buffer(f"{modality.name}_range", (maximum - minimum).reshape(1, 1, -1, 1, 1))

    def prepare_sample(self, raster: torch.Tensor) -> MaskedOlmoEarthSample:
        b, _, h, w = raster.shape
        months = raster.reshape(b, 12, 17, h, w)
        images, masks = [], []
        for name, raw in [("sentinel2_l2a", months[:, :, :12]), ("sentinel1", months[:, :, 12:14])]:
            finite = torch.isfinite(raw)
            # One missing pixel/band invalidates its whole 4x4 modality token.
            valid = finite.all(dim=2).reshape(b, 12, h // 4, 4, w // 4, 4).all(dim=5).all(dim=3)
            valid = valid.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
            mask = torch.where(valid, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)
            masks.append(mask.permute(0, 2, 3, 1).unsqueeze(-1).to(torch.int32))
            # Patch embedding interpolates across neighbours: discard all invalid patch contents.
            safe = torch.where(finite & valid.unsqueeze(2), raw, 0)
            normalized = ((safe - getattr(self, f"{name}_minimum")) / getattr(self, f"{name}_range")).float()
            images.append(normalized.permute(0, 3, 4, 1, 2))
        return MaskedOlmoEarthSample(
            timestamps=months[:, :, 14:, 0, 0].to(torch.int64),
            sentinel2_l2a=images[0], sentinel2_l2a_mask=masks[0],
            sentinel1=images[1], sentinel1_mask=masks[1],
        )

    def encode(self, sample: MaskedOlmoEarthSample) -> dict[str, torch.Tensor]:
        e = self.encoder
        patches = e.patch_embeddings(sample, patch_size=4)
        tokens_only, masks, dims = e.split_tokens_masks_and_dims(patches)
        timestamps = sample.timestamps
        assert isinstance(timestamps, torch.Tensor)
        tokens = e.composite_encodings(tokens_only, timestamps, 4, 10)
        positions = e.build_rope_positions(tokens_only, masks, 4, 10, timestamps=timestamps)
        tokens.update(masks)
        tokens, mask = e.collapse_and_combine_hwtc(tokens)
        valid = mask == MaskValue.ONLINE_ENCODER.value
        # Fixed slots avoid data-dependent removal; invalid keys never participate.
        # Empty tiles take a safe attention path, then pooling returns nodata.
        attention_mask = valid | ~valid.any(dim=1, keepdim=True)
        for block in e.blocks:
            tokens = block(tokens, attn_mask=attention_mask, rope_positions=positions)
        result = e.split_and_expand_per_modality(e.norm(tokens), dims)
        result.update(masks)
        return result

    @staticmethod
    def pool(tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled, available = [], []
        for name in ["sentinel2_l2a", "sentinel1"]:
            valid = tokens[f"{name}_mask"] == MaskValue.ONLINE_ENCODER.value
            count = valid.sum(dim=(3, 4)).unsqueeze(-1)
            values = torch.where(valid.unsqueeze(-1), tokens[name], 0)
            pooled.append(values.sum(dim=(3, 4)) / count.clamp_min(1))
            available.append(count > 0)
        count = torch.stack(available).sum(dim=0)
        fused = torch.stack(pooled).sum(dim=0) / count.clamp_min(1)
        result = torch.where(count > 0, fused, float("nan"))
        return result.permute(0, 3, 1, 2)

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encode(self.prepare_sample(raster)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / f"{MODEL_NAME}.onnx")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_pretrained_model()
    assert isinstance(model.encoder, Encoder)
    wrapper = AnnualPatchEmbeddings(model.encoder).eval()
    raster = torch.ones(1, 204, 32, 32)
    months = raster.reshape(1, 12, 17, 32, 32)
    months[:, :, 12:14] = -12
    months[:, :, 14] = 16
    months[:, :, 15] = torch.arange(12).reshape(1, 12, 1, 1)
    months[:, :, 16] = 2024
    torch.onnx.export(
        wrapper, (raster,), args.output, dynamo=True, opset_version=18,
        external_data=False, input_names=["raster"], output_names=["annual_embeddings"],
    )
    onnx.checker.check_model(str(args.output))
    stac = json.loads((ROOT / f"{MODEL_NAME}.stac.json").read_text())
    for tensor, bands in [(stac["properties"]["mlm:input"][0], INPUT_BANDS),
                          (stac["properties"]["mlm:output"][0], OUTPUT_BANDS)]:
        assert tensor["bands"] == bands
    stac["assets"]["model"]["href"] = args.output.name
    args.output.with_suffix(".stac.json").write_text(json.dumps(stac, indent=2) + "\n")
    print(f"Saved {args.output} (SHA-256 {hashlib.sha256(args.output.read_bytes()).hexdigest()})")
    print(f"Check it with: python check_olmoearth_onnx.py --model {args.output}")


if __name__ == "__main__":
    main()
