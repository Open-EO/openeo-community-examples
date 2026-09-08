"""openEO UDF: TESSERA v2 "student" pixel embeddings from Sentinel-1/2.

This vendors (with minor adaptations noted inline) the model definition and
inference helpers from `tessera_infer_v2/student/{model.py,infer.py}` in
https://github.com/ucam-eo/tessera, so the UDF only needs torch + numpy - no
custom reimplementation of the forward pass. See that repo for the original,
authoritative source and license.

Torch is not part of the default openEO UDF sandbox, so it must be supplied
via the "udf-dependency-archives" job option, e.g.:

    job_options = {
        "udf-dependency-archives": [
            "https://s3.waw3-1.cloudferro.com/project_dependencies/torch_deps_python311.zip#feature_deps",
        ],
    }

The checkpoint itself (a `student_*.pt` file) is downloaded by the UDF at
runtime from a URL passed via context, e.g.:

    openeo.UDF.from_file("udf_tessera_embedding.py",
                          context={"weights_url": "https://.../student_nano.pt"})

Expects the merged input cube to contain (at least) these bands, produced by
merge_cubes() of three separately-loaded collections (see TesseraEmbedding.ipynb):
    S2:        B04, B02, B03, B08, B8A, B05, B06, B07, B11, B12
    S1 asc:    VV_ASC, VH_ASC
    S1 desc:   VV_DESC, VH_DESC
Cloud masking happens outside this UDF: the S2 bands are expected to already
have cloudy/shadowed/cirrus pixels set to no-data (NaN), e.g. via `.mask()`
with an SCL-derived mask, before this UDF runs. Because merge_cubes() unions
the "t" dimension, a given band is NaN on dates that don't belong to its
source - that's how we tell the three sources apart.
"""
import functools
import io
import math
import sys
import urllib.request
from typing import Optional, Tuple

sys.path.insert(0, "feature_deps")  # extracted torch_deps_python311.zip (job_options)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from openeo.metadata import CubeMetadata
from openeo.udf import XarrayDataCube

# =============================================================================
# Vendored from tessera_infer_v2/student/model.py
# =============================================================================
S2_BAND_MEAN = np.array(
    [1633.0042, 1341.1090, 1539.5536, 3054.8269, 3117.4658,
     2004.1648, 2694.7275, 2945.1504, 2266.6079, 1657.3094],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [1999.4603, 2014.7549, 1929.2201, 1754.2493, 1649.9807,
     1936.8988, 1748.6041, 1708.6991, 1207.5250, 1108.6046],
    dtype=np.float32,
)
S1A_BAND_MEAN = np.array([5909.3921, 3405.0322], dtype=np.float32)
S1A_BAND_STD = np.array([1507.1750, 1531.2615], dtype=np.float32)
S1D_BAND_MEAN = np.array([5816.1382, 3277.7576], dtype=np.float32)
S1D_BAND_STD = np.array([1554.6475, 1546.4733], dtype=np.float32)

S2_BAND_ORDER = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
S1_BAND_ORDER = ["VV", "VH"]


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding using the (raw integer) DOY value."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class AttentionPooling(nn.Module):
    """Plain single-head softmax attention pool over T."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T == 0:
            return torch.zeros(B, D, device=x.device, dtype=x.dtype)
        if T == 1:
            return x.squeeze(1)
        w = torch.softmax(self.query(x), dim=1)
        return (w * x).sum(dim=1)


class TransformerEncoder(nn.Module):
    """Per-pixel band embedding + DOY positional + Transformer + attention pooling."""

    def __init__(self, band_num: int, latent_dim: int, nhead: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 256) -> None:
        super().__init__()
        input_dim = band_num
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attn_pool = AttentionPooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, band_num + 1) - last channel is the (raw integer) DOY value
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        bands_embedded = self.embedding(bands)
        temporal_encoding = self.temporal_encoder(doy)
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        return self.attn_pool(x)


class PixelStudent(nn.Module):
    """2-backbone pixel encoder. emb_dim = 128 (Matryoshka-ordered).

    Inputs (per pixel):
        s2_x : (B, T_s2, 11)   10 bands + 1 DOY (raw integer 1..365)
        s1_x : (B, T_s1,  3)    2 bands + 1 DOY (raw integer; s1a + s1d MERGED)

    encode() returns (B, 128). The first K dims are usable on their own for
    K in {16, 32, 64, 128}.
    """

    def __init__(
        self,
        repr_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.matryoshka_dims = tuple(int(d) for d in matryoshka_dims)
        for d in self.matryoshka_dims:
            if d > repr_dim:
                raise ValueError(f"matryoshka cut {d} > repr_dim {repr_dim}")

        def make_enc(band_num: int) -> TransformerEncoder:
            return TransformerEncoder(
                band_num=band_num, latent_dim=latent_dim,
                nhead=nhead, num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                max_seq_len=max_seq_len,
            )

        self.s2_backbone = make_enc(10)
        self.s1_backbone = make_enc(2)

        backbone_out = latent_dim * 4
        fused_in = 2 * backbone_out
        self.dim_reducer = nn.Sequential(
            nn.Linear(fused_in, fused_in * 2),
            nn.LayerNorm(fused_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(fused_in * 2, repr_dim),
            # Final non-affine LayerNorm: normalizes the 128-d output to
            # per-pixel mean 0 / std 1. No learnable params.
            nn.LayerNorm(repr_dim, elementwise_affine=False),
        )

    def encode(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        s2 = self.s2_backbone(s2_x)
        s1 = self.s1_backbone(s1_x)
        fused = torch.cat([s2, s1], dim=-1)
        return self.dim_reducer(fused)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        return self.encode(s2_x, s1_x)


PixelStudentV11 = PixelStudent


def load_model(ckpt_source, device: torch.device = torch.device("cpu")):
    """Load a pretrained pixel student (encoder) from a .pt file path or file-like object."""
    payload = torch.load(ckpt_source, map_location=device, weights_only=False)
    cfg = payload.get("args", {}) or {}
    matryoshka_dims = tuple(
        int(d) for d in str(cfg.get("matryoshka_dims", "16,32,64,128")).split(",")
    )
    model = PixelStudent(
        repr_dim=int(cfg.get("repr_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        num_layers=int(cfg.get("num_layers", 4)),
        nhead=int(cfg.get("nhead", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=0.0,
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        matryoshka_dims=matryoshka_dims,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


# =============================================================================
# Vendored from tessera_infer_v2/student/infer.py
# =============================================================================
BIN_EDGES = list(range(8, 257, 8))  # [8, 16, 24, ..., 256]


def get_bin_size(n_obs: int) -> int:
    if n_obs <= 0:
        return 0
    for b in BIN_EDGES:
        if n_obs <= b:
            return b
    return BIN_EDGES[-1]


def _vec_get_bin_size(n_obs: np.ndarray) -> np.ndarray:
    out = np.full_like(n_obs, BIN_EDGES[-1])
    out[n_obs <= 0] = 0
    for b in reversed(BIN_EDGES):
        out = np.where((n_obs > 0) & (n_obs <= b), b, out)
    return out


def _pad_pattern(n: int, B: int) -> np.ndarray:
    """(B,) int64 indices into [0, n) reproducing the training pad_to_bin."""
    if n == 0:
        return np.zeros(B, dtype=np.int64)
    if n >= B:
        return np.linspace(0, n - 1, B, dtype=np.int64)
    remain = B - n
    if remain <= n:
        groups = np.array_split(np.arange(n), remain)
        fill = np.array([gp[len(gp) // 2] for gp in groups], dtype=np.int64)
    else:
        fill = (np.arange(remain) % n).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), fill])


def _build_source_indices(valid_per_pix: np.ndarray, B: int) -> np.ndarray:
    """valid_per_pix: (G, T) bool. Returns (G, B) int64."""
    G, T = valid_per_pix.shape
    src = np.zeros((G, B), dtype=np.int64)
    if G == 0 or B == 0:
        return src
    n_per = valid_per_pix.sum(axis=1).astype(np.int64)
    sorted_pos = np.argsort(~valid_per_pix, axis=1, kind="stable").astype(np.int64)
    unique_n, inverse = np.unique(n_per, return_inverse=True)
    for ki, n_val in enumerate(unique_n):
        n = int(n_val)
        if n == 0:
            continue
        pix = np.where(inverse == ki)[0]
        pattern = _pad_pattern(n, B)
        src[pix] = sorted_pos[pix][:, pattern]
    return src


@torch.no_grad()
def encode_pixels(
    model,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    s2_masks: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cpu"),
    standardize: bool = True,
) -> np.ndarray:
    """Encode B independent pixels' time series into 128-d embeddings.

    Args:
        s2_bands     : (B, T_s2, 10)
        s2_doys      : (B, T_s2) or (T_s2,) ints (1..365)
        s1_asc_bands : (B, T_s1a, 2) raw (or None)
        s1_asc_doys  : (B, T_s1a) or (T_s1a,)
        s1_desc_bands: (B, T_s1d, 2) raw (or None)
        s1_desc_doys : (B, T_s1d) or (T_s1d,)
        s2_masks     : (B, T_s2) 1=valid, 0=cloud (or None -> all valid)
    Returns: (B, 128) float32.
    """
    B = s2_bands.shape[0]
    out = np.empty((B, model.repr_dim), dtype=np.float32)
    if B == 0:
        return out

    T_s2 = s2_bands.shape[1]

    # Build per-pixel S1 (merged) arrays. asc and desc each have their own
    # mean/std - z-score per-source BEFORE concatenation so the merged S1
    # stream the model consumes is already standardized.
    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        s1a_b = s1_asc_bands.astype(np.float32)
        s1a_valid = np.any((s1a_b != 0) & ~np.isnan(s1a_b), axis=-1)
        s1a_b = np.nan_to_num(s1a_b, nan=0.0)
        if standardize:
            s1a_b = (s1a_b - S1A_BAND_MEAN) / (S1A_BAND_STD + 1e-9)
        if s1_asc_doys.ndim == 1:
            s1a_d = np.broadcast_to(s1_asc_doys[None, :], (B, s1_asc_doys.shape[0])).copy()
        else:
            s1a_d = s1_asc_doys
    else:
        s1a_b = np.zeros((B, 0, 2), dtype=np.float32)
        s1a_d = np.zeros((B, 0), dtype=np.float32)
        s1a_valid = np.zeros((B, 0), dtype=bool)

    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        s1d_b = s1_desc_bands.astype(np.float32)
        s1d_valid = np.any((s1d_b != 0) & ~np.isnan(s1d_b), axis=-1)
        s1d_b = np.nan_to_num(s1d_b, nan=0.0)
        if standardize:
            s1d_b = (s1d_b - S1D_BAND_MEAN) / (S1D_BAND_STD + 1e-9)
        if s1_desc_doys.ndim == 1:
            s1d_d = np.broadcast_to(s1_desc_doys[None, :], (B, s1_desc_doys.shape[0])).copy()
        else:
            s1d_d = s1_desc_doys
    else:
        s1d_b = np.zeros((B, 0, 2), dtype=np.float32)
        s1d_d = np.zeros((B, 0), dtype=np.float32)
        s1d_valid = np.zeros((B, 0), dtype=bool)

    if s1a_b.shape[1] + s1d_b.shape[1] > 0:
        s1_b_merged = np.concatenate([s1a_b, s1d_b], axis=1)
        s1_d_merged = np.concatenate([s1a_d, s1d_d], axis=1)
    else:
        s1_b_merged = np.zeros((B, 0, 2), dtype=np.float32)
        s1_d_merged = np.zeros((B, 0), dtype=np.float32)

    if s2_masks is not None:
        s2_v = s2_masks.astype(bool)
    else:
        s2_v = np.ones((B, T_s2), dtype=bool)
    s1_v = (np.concatenate([s1a_valid, s1d_valid], axis=1)
            if (s1a_valid.shape[1] + s1d_valid.shape[1]) > 0
            else np.zeros((B, 0), dtype=bool))

    n_s2 = s2_v.sum(axis=1)
    n_s1 = s1_v.sum(axis=1)
    s2_bin = _vec_get_bin_size(n_s2).astype(np.int32)
    s1_bin = _vec_get_bin_size(n_s1).astype(np.int32)
    keys = s2_bin * 1000 + s1_bin
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    for ki, key in enumerate(unique_keys):
        s2_b_size = int(key // 1000)
        s1_b_size = int(key % 1000)
        idxs = np.where(inverse == ki)[0]
        if s2_b_size == 0 and s1_b_size == 0:
            continue
        s2_B = max(s2_b_size, 1)
        s1_B = max(s1_b_size, 1)
        for s in range(0, idxs.size, batch_pixels):
            chunk = idxs[s: s + batch_pixels]
            G = len(chunk)
            s2_in = np.zeros((G, s2_B, 11), dtype=np.float32)
            s1_in = np.zeros((G, s1_B, 3), dtype=np.float32)

            if s2_b_size > 0:
                src = _build_source_indices(s2_v[chunk], s2_B)
                gathered = np.take_along_axis(
                    s2_bands[chunk], src[:, :, None].repeat(10, axis=2), axis=1
                )
                if standardize:
                    gathered = (gathered - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
                s2_in[:, :, :10] = gathered
                if s2_doys.ndim == 1:
                    s2_doys_pix = np.broadcast_to(s2_doys[None, :], (G, T_s2))
                else:
                    s2_doys_pix = s2_doys[chunk]
                s2_in[:, :, 10] = np.take_along_axis(s2_doys_pix, src, axis=1).astype(np.float32)

            if s1_b_size > 0:
                src = _build_source_indices(s1_v[chunk], s1_B)
                gathered = np.take_along_axis(
                    s1_b_merged[chunk], src[:, :, None].repeat(2, axis=2), axis=1
                )
                s1_in[:, :, :2] = gathered
                s1_in[:, :, 2] = np.take_along_axis(s1_d_merged[chunk], src, axis=1).astype(np.float32)

            s2_t = torch.from_numpy(s2_in).to(device, non_blocking=True)
            s1_t = torch.from_numpy(s1_in).to(device, non_blocking=True)
            emb = model.encode(s2_t, s1_t)
            out[chunk] = emb.float().cpu().numpy()
    return out


@torch.no_grad()
def encode_tile(
    model,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s2_masks: Optional[np.ndarray] = None,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cpu"),
    standardize: bool = True,
) -> np.ndarray:
    """Encode one tile into an (H, W, 128) embedding map.

    Args:
        s2_bands     : (T_s2, H, W, 10)  raw reflectance
        s2_doys      : (T_s2,)           day-of-year per S2 frame
        s2_masks     : (T_s2, H, W) or None (1=valid)
        s1_asc_bands : (T_s1a, H, W, 2) optional
        s1_asc_doys  : (T_s1a,)         optional
        s1_desc_bands: (T_s1d, H, W, 2) optional
        s1_desc_doys : (T_s1d,)         optional
    Returns: (H, W, 128) float32.
    """
    T_s2, H, W, _ = s2_bands.shape
    N = H * W
    s2_flat = s2_bands.transpose(1, 2, 0, 3).reshape(N, T_s2, 10)
    s2_doys_flat = np.broadcast_to(s2_doys[None, :], (N, T_s2)).copy()
    s2_masks_flat = (s2_masks.transpose(1, 2, 0).reshape(N, T_s2) if s2_masks is not None else None)

    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        Ta = s1_asc_bands.shape[0]
        s1a_flat = s1_asc_bands.transpose(1, 2, 0, 3).reshape(N, Ta, 2)
        s1a_doys_flat = np.broadcast_to(s1_asc_doys[None, :], (N, Ta)).copy()
    else:
        s1a_flat = None
        s1a_doys_flat = None

    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        Td = s1_desc_bands.shape[0]
        s1d_flat = s1_desc_bands.transpose(1, 2, 0, 3).reshape(N, Td, 2)
        s1d_doys_flat = np.broadcast_to(s1_desc_doys[None, :], (N, Td)).copy()
    else:
        s1d_flat = None
        s1d_doys_flat = None

    out = encode_pixels(
        model, s2_flat, s2_doys_flat,
        s1_asc_bands=s1a_flat, s1_asc_doys=s1a_doys_flat,
        s1_desc_bands=s1d_flat, s1_desc_doys=s1d_doys_flat,
        s2_masks=s2_masks_flat,
        batch_pixels=batch_pixels, device=device, standardize=standardize,
    )
    return out.reshape(H, W, model.repr_dim)


# =============================================================================
# openEO entry points.
# =============================================================================
@functools.lru_cache(maxsize=1)
def _load_model_from_url(url: str):
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    return load_model(io.BytesIO(data), device=torch.device("cpu"))


def _get_model(context: dict):
    url = (context or {}).get("weights_url")
    if not url:
        raise ValueError(
            "context['weights_url'] is required, e.g. openeo.UDF.from_file(..., "
            "context={'weights_url': '<url to a student_*.pt checkpoint>'})"
        )
    return _load_model_from_url(url)


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    model = _get_model(context)
    band_names = [f"tessera_{i}" for i in range(model.repr_dim)]
    return metadata.rename_labels(dimension="bands", target=band_names)


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    model = _get_model(context)
    da = cube.get_array()  # dims: t, bands, y, x

    t_values = pd.DatetimeIndex(da.coords["t"].values)
    doy = t_values.dayofyear.to_numpy().astype(np.float32)

    def select(band_names):
        sub = da.sel(bands=band_names)
        valid = sub.notnull().any(dim=["bands", "y", "x"]).values
        idx = np.where(valid)[0]
        arr = sub.isel(t=idx).transpose("t", "y", "x", "bands").values.astype(np.float32)
        return arr, doy[idx]

    # s2_bands_raw keeps its NaNs (from cloud masking done outside this UDF via
    # .mask()) so we can derive a per-pixel/per-date clear-sky mask before filling.
    s2_bands_raw, s2_doys = select(S2_BAND_ORDER)
    s2_masks = (~np.isnan(s2_bands_raw).any(axis=-1)).astype(np.float32)
    s2_bands = np.nan_to_num(s2_bands_raw)

    s1a_names = [f"{b}_ASC" for b in S1_BAND_ORDER]
    s1d_names = [f"{b}_DESC" for b in S1_BAND_ORDER]
    s1a_bands, s1a_doys = select(s1a_names)
    s1a_bands = np.nan_to_num(s1a_bands)
    s1d_bands, s1d_doys = select(s1d_names)
    s1d_bands = np.nan_to_num(s1d_bands)

    embedding = encode_tile(
        model, s2_bands, s2_doys, s2_masks=s2_masks,
        s1_asc_bands=s1a_bands, s1_asc_doys=s1a_doys,
        s1_desc_bands=s1d_bands, s1_desc_doys=s1d_doys,
        device=torch.device("cpu"),
    )  # (H, W, repr_dim)

    result = xr.DataArray(
        embedding.transpose(2, 0, 1),
        dims=["bands", "y", "x"],
        coords={"bands": [f"tessera_{i}" for i in range(model.repr_dim)], "y": da.coords["y"], "x": da.coords["x"]},
    )
    return XarrayDataCube(result)
