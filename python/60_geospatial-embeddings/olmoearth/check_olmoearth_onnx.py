"""Check the standalone annual export against the stock encoder and ONNX Runtime.

Run after export_olmoearth_to_onnx.py. Artifacts retain the OlmoEarth Artifact
License restrictions described in README.md#licensing.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.nn.flexi_vit import Encoder

from export_olmoearth_to_onnx import AnnualPatchEmbeddings, load_pretrained_model
from openeo_olmoearth import INPUT_BANDS, MODEL_NAME, OUTPUT_BANDS, calendar_months, patch_validity

ROOT = Path(__file__).resolve().parent


def synthetic_raster() -> torch.Tensor:
    rng = np.random.default_rng(0)
    months = np.empty((1, 12, 17, 32, 32), dtype=np.float32)
    months[:, :, :12] = rng.uniform(500, 7000, (1, 12, 12, 32, 32))
    months[:, :, 12:14] = rng.normal(-15, 5, (1, 12, 2, 32, 32))
    months[:, :, 14:] = np.array(calendar_months(2024)[1]).reshape(1, 12, 3, 1, 1)
    return torch.from_numpy(months.reshape(1, 204, 32, 32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / f"{MODEL_NAME}.onnx")
    args = parser.parse_args()
    torch.set_num_threads(4)
    onnx.checker.check_model(str(args.model))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    contract = json.loads(args.model.with_suffix(".stac.json").read_text())["properties"]
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    assert inp.name == "raster" and inp.shape == [1, 204, 32, 32] and inp.type == "tensor(float)"
    assert out.name == "annual_embeddings" and out.shape == [1, 128, 8, 8] and out.type == "tensor(float)"
    for actual, declared, nested, bands in [(inp, contract["mlm:input"][0], "input", INPUT_BANDS),
                                           (out, contract["mlm:output"][0], "result", OUTPUT_BANDS)]:
        assert actual.shape == declared["shape"] == declared[nested]["shape"]
        assert declared["bands"] == bands
        assert declared["data_type"] == declared[nested]["data_type"] == "float32"
        assert declared["dim_order"] == declared[nested]["dim_order"] == ["batch", "bands", "height", "width"]
    intervals, timestamps = calendar_months(2024)
    assert len(intervals) == len(timestamps) == 12
    assert intervals[1] == ["2024-02-01", "2024-03-01"] and timestamps[1] == [15, 1, 2024]
    assert intervals[0][0] == "2024-01-01" and intervals[-1][1] == "2025-01-01"
    assert [t[1] for t in timestamps] == list(range(12))
    model = load_pretrained_model()
    assert isinstance(model.encoder, Encoder)
    wrapper = AnnualPatchEmbeddings(model.encoder).eval()
    full = synthetic_raster()
    sample = wrapper.prepare_sample(full)
    normalizer = Normalizer(Strategy.COMPUTED)
    months = full.reshape(1, 12, 17, 32, 32)
    for modality, raw, image in [
        (Modality.SENTINEL2_L2A, months[:, :, :12], sample.sentinel2_l2a),
        (Modality.SENTINEL1, months[:, :, 12:14], sample.sentinel1),
    ]:
        raw = raw.permute(0, 3, 4, 1, 2).numpy()
        expected = normalizer.normalize(modality, raw).astype(np.float32)
        assert isinstance(image, torch.Tensor)
        np.testing.assert_array_equal(image.numpy(), expected)

    partial = full.clone()
    monthly = partial.reshape(1, 12, 17, 32, 32)
    monthly[:, 1, :12] = float("nan")  # clouded month
    monthly[:, 3, 12:14] = float("nan")  # absent radar month
    monthly[:, :, :14, :4, :4] = float("nan")  # no annual evidence at one patch
    monthly[:, :, :12, 4:8, :4] = float("nan")  # S1-only annual patch
    monthly[:, 5, 0, 4, 4] = float("inf")  # single invalid pixel masks whole S2 token
    valid = patch_validity(partial.numpy()[0])
    assert valid.shape == (12, 2, 8, 8) and not valid[5, 0, 1, 1]
    missing_s2 = full.clone()
    missing_s2.reshape(1, 12, 17, 32, 32)[:, :, :12] = float("nan")
    missing_s1 = full.clone()
    missing_s1.reshape(1, 12, 17, 32, 32)[:, :, 12:14] = float("nan")
    empty = full.clone()
    empty.reshape(1, 12, 17, 32, 32)[:, :, :14] = float("nan")
    with torch.inference_mode():
        baseline = None
        for label, raster in [("complete", full), ("partial", partial), ("S1 only", missing_s2),
                              ("S2 only", missing_s1), ("empty", empty)]:
            started = time.monotonic()
            actual = wrapper(raster).numpy()
            expected = actual
            if label != "empty":
                reference = model.encoder(wrapper.prepare_sample(raster), patch_size=4, fast_pass=False)["tokens_and_masks"]
                # Independent valid-only annual pooling, using the unmodified encoder.
                sums, present = [], []
                for name in ["sentinel2_l2a", "sentinel1"]:
                    tokens, mask = getattr(reference, name), getattr(reference, f"{name}_mask")
                    assert isinstance(tokens, torch.Tensor) and isinstance(mask, torch.Tensor)
                    valid = mask.numpy() == 0
                    counts = valid.sum(axis=(3, 4))[..., None]
                    sums.append(np.where(valid[..., None], tokens.numpy(), 0).sum(axis=(3, 4)) / np.maximum(counts, 1))
                    present.append(counts > 0)
                count = np.sum(present, axis=0)
                expected = (np.sum(sums, axis=0) / np.maximum(count, 1)).transpose(0, 3, 1, 2)
                expected = np.where((count > 0).transpose(0, 3, 1, 2), expected, np.nan)
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
            remote = session.run(None, {"raster": raster.numpy()})[0]
            assert isinstance(remote, np.ndarray)
            np.testing.assert_allclose(remote, expected, rtol=1e-3, atol=1e-4)
            np.testing.assert_array_equal(np.isnan(remote), np.isnan(actual))
            if label == "complete":
                assert np.isfinite(actual).all()
                baseline = actual
            elif label == "partial":
                np.testing.assert_array_equal(np.isfinite(actual).all(axis=1)[0], patch_validity(raster.numpy()[0]).any(axis=(0, 1)))
                # Invalid token contents cannot affect any valid output token.
                changed = raster.clone().reshape(1, 12, 17, 32, 32)
                changed[:, 5, :12, 4:8, 4:8] = 999999
                changed[:, 5, 0, 4, 4] = float("inf")
                np.testing.assert_allclose(wrapper(changed.reshape_as(raster)).numpy(), actual, rtol=1e-3, atol=1e-4)
            elif label == "empty":
                assert np.isnan(actual).all()
            else:
                assert np.isfinite(actual).all()
            error = np.nanmax(np.abs(actual - expected)) if label != "empty" else 0
            print(f"{label}: {actual.shape}, encoder max error {error:.6g}, {time.monotonic() - started:.2f}s")
        # A changed later month affects the annual joint representation.
        changed = full.clone()
        changed.reshape(1, 12, 17, 32, 32)[:, 11, :14] += 2
        assert baseline is not None and not np.allclose(wrapper(changed).numpy(), baseline)
    print("Verified calendar, normalization, masking, annual pooling and ONNX parity")


if __name__ == "__main__":
    main()
