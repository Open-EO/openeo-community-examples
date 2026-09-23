#%%

"""Offline helper: export the TerraMind encoder to ONNX.

Run cell-by-cell. Set ``MODEL`` in the config cell and execute.

The exported graph:
    - takes two positional inputs, ``s2 (B, 12, 224, 224)`` and ``s1 (B, 2, 224, 224)``
      in their **raw** value ranges (S2 L2A DN, S1 GRD linear power),
    - applies TerraMind's pre-training standardization internally (baked as
      Constants in the ONNX graph so the UDF doesn't need to know the values),
    - returns the last-layer patch tokens ``(B, 196, 768)`` — the same output as
      ``model({"S2L2A": ..., "S1GRD": ...})`` with ``merge_method="mean"``.

Then upload the ``.onnx`` to a location the openEO backend can reach (an S3
bucket, an artefact server) and pass its URL via the UDF context.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _pretrain_stats():
    """Read S2L2A / S1GRD pretraining mean+std from terratorch's register module.

    In terratorch's TerraMind register the values live in the module-level
    ``v1_pretraining_mean`` / ``v1_pretraining_std`` dicts, keyed by the
    internal modality names ``untok_sen2l2a@224`` and ``untok_sen1grd@224``.
    """
    from terratorch.models.backbones.terramind.model import terramind_register as reg

    s2_mean = np.asarray(reg.v1_pretraining_mean["untok_sen2l2a@224"], dtype=np.float32)
    s2_std  = np.asarray(reg.v1_pretraining_std["untok_sen2l2a@224"],  dtype=np.float32)
    s1_mean = np.asarray(reg.v1_pretraining_mean["untok_sen1grd@224"], dtype=np.float32)
    s1_std  = np.asarray(reg.v1_pretraining_std["untok_sen1grd@224"],  dtype=np.float32)
    return s2_mean, s2_std, s1_mean, s1_std


class TerraMindEncoderONNX(nn.Module):
    """Standardize + backbone(mean-merge) + return last-layer tokens."""

    def __init__(self, backbone: nn.Module,
                 s2_mean, s2_std, s1_mean, s1_std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("s2_mean", torch.tensor(s2_mean).view(1, -1, 1, 1))
        self.register_buffer("s2_std", torch.tensor(s2_std).view(1, -1, 1, 1))
        self.register_buffer("s1_mean", torch.tensor(s1_mean).view(1, -1, 1, 1))
        self.register_buffer("s1_std", torch.tensor(s1_std).view(1, -1, 1, 1))

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s2 = (s2 - self.s2_mean) / (self.s2_std + 1e-9)
        s1 = (s1 - self.s1_mean) / (self.s1_std + 1e-9)
        out = self.backbone({"S2L2A": s2, "S1GRD": s1})
        # backbone can return a list of per-layer token tensors; take the last.
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out   # (B, 196, 768)


# %% Config — edit and re-run
MODEL = "terramind_v1_base"   # tiny / small / base / large
OUT_PATH = Path(f"{MODEL}_encoder.onnx")
OPSET = 17

# HF repo id per model size (published by IBM+ESA).
HF_REPO_ID = {
    "terramind_v1_tiny":  "ibm-esa-geospatial/TerraMind-1.0-Tiny",
    "terramind_v1_small": "ibm-esa-geospatial/TerraMind-1.0-Small",
    "terramind_v1_base":  "ibm-esa-geospatial/TerraMind-1.0-Base",
    "terramind_v1_large": "ibm-esa-geospatial/TerraMind-1.0-Large",
}[MODEL]
WEIGHTS_DIR = Path("terramind_weights") / MODEL

# %% Download the HF repo into ./terramind_weights/<MODEL>/ so we can inspect it
from huggingface_hub import snapshot_download

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
local_dir = snapshot_download(repo_id=HF_REPO_ID, local_dir=str(WEIGHTS_DIR))
print("Downloaded to:", local_dir)
print("Contents:")
for p in sorted(Path(local_dir).rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(local_dir), f"({p.stat().st_size/1e6:.2f} MB)")

# %% Build backbone (uses cached weights)
from terratorch.registry import BACKBONE_REGISTRY

print(f"Building {MODEL} from terratorch...")
backbone = BACKBONE_REGISTRY.build(
    MODEL,
    pretrained=True,
    modalities=["S2L2A", "S1GRD"],
    merge_method="mean",
).eval()

s2_mean, s2_std, s1_mean, s1_std = _pretrain_stats()
wrapped = TerraMindEncoderONNX(backbone, s2_mean, s2_std, s1_mean, s1_std).eval()

# %% Export to ONNX
s2_dummy = torch.zeros(1, 12, 224, 224)
s1_dummy = torch.zeros(1, 2, 224, 224)

print(f"Exporting to {OUT_PATH} (opset {OPSET})...")
torch.onnx.export(
    wrapped,
    (s2_dummy, s1_dummy),
    OUT_PATH.as_posix(),
    input_names=["s2", "s1"],
    output_names=["tokens"],
    dynamic_axes={
        "s2": {0: "batch"},
        "s1": {0: "batch"},
        "tokens": {0: "batch"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
)
print(f"Done. Wrote {OUT_PATH.stat().st_size / 1e6:.1f} MB.")

# %%
