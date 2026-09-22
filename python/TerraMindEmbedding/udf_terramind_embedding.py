"""openEO UDF: TerraMind patch embeddings via ONNX Runtime.

No PyTorch, no TerraTorch, this UDF only needs ``onnxruntime`` + ``numpy`` in
the sandbox. The TerraMind encoder is exported to ONNX **offline** with
``export_terramind_to_onnx.py`` and shipped to the backend as a zip via the
``udf-dependency-archives`` job option (same mechanism as ``onnx_deps.zip``).
The archive must contain both the ``.onnx`` graph and its ``.onnx.data``
external-weights sidecar in the same folder.

Standardization is baked into the exported ONNX graph, so this UDF hands raw
Sentinel-2 L2A DN and Sentinel-1 GRD linear power directly to the model.

Expected input cube bands (order irrelevant, resolved by name):
    S2L2A: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    S1GRD: VV, VH   (linear power, sigma0-ellipsoid from openEO ``sar_backscatter``)

Context:
    onnx_dir      : folder alias the archive extracts to (default: "terramind_onnx")
    onnx_filename : .onnx filename inside that folder
                    (default: "terramind_v1_base_encoder.onnx")
"""
import functools
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from openeo.metadata import CubeMetadata
from openeo.udf import XarrayDataCube

# onnxruntime is supplied via the ``udf-dependency-archives`` job option that
# extracts to ./onnx_deps (same convention as ../OnnxMLInference).
sys.path.append("onnx_deps")
import onnxruntime as ort


S2_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                 "B08", "B8A", "B09", "B11", "B12"]
S1_BAND_ORDER = ["VV", "VH"]

PATCH = 224
PATCH_TOKEN = 16   # 224 / 14 tokens per side; also the output/input step ratio
TOKENS_PER_SIDE = PATCH // PATCH_TOKEN   # 14
EMB_DIM = 768


@functools.lru_cache(maxsize=2)
def _load_session(onnx_path: str) -> ort.InferenceSession:
    """Return a cached InferenceSession for the ONNX file at ``onnx_path``.

    The .onnx and its .onnx.data sidecar must both live in the same folder
    (the archive alias set via ``udf-dependency-archives``).
    """
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Make sure the terramind ONNX "
            f"zip is listed in the 'udf-dependency-archives' job option and "
            f"that context['onnx_dir']/context['onnx_filename'] match its layout."
        )
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 2
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def _select_bands(cube: xr.DataArray, names) -> np.ndarray:
    avail = list(cube.coords["bands"].values.tolist())
    missing = [b for b in names if b not in avail]
    if missing:
        raise ValueError(f"Missing bands in input cube: {missing}. Available: {avail}")
    return cube.sel(bands=list(names)).values.astype(np.float32)


def _encode_tile(sess: ort.InferenceSession,
                 s2_tile: np.ndarray, s1_tile: np.ndarray) -> np.ndarray:
    """Encode one PATCH x PATCH tile.

    Returns ``(768, 224, 224)`` as float32: the encoder's native (768, 14, 14)
    token grid nearest-neighbour upsampled back to the input pixel grid.
    Yes, this inflates the intermediate ~256x with duplicate values, but the
    backend's ``apply_neighborhood`` overlap trim is pixel-index based and
    ignores any resolution change declared via ``apply_metadata``; returning
    tokens at their native 160 m grid causes ``GridBounds do not intersect``
    errors as soon as ``overlap`` is non-empty. Upsampling here keeps the
    output on the input grid so overlap works, and downstream steps can
    ``resample_spatial`` back down if they want to shrink the cube.
    """
    s2 = s2_tile[np.newaxis, ...]   # (1, 12, 224, 224)
    s1 = s1_tile[np.newaxis, ...]   # (1, 2, 224, 224)
    tokens = sess.run(None, {"s2": s2, "s1": s1})[0]   # (1, 196, 768)
    tokens = tokens[0].astype(np.float32)
    grid = tokens.reshape(TOKENS_PER_SIDE, TOKENS_PER_SIDE, EMB_DIM).transpose(2, 0, 1)
    return np.repeat(np.repeat(grid, PATCH_TOKEN, axis=1), PATCH_TOKEN, axis=2)


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    context = context or {}
    onnx_dir = context.get("onnx_dir", "terramind_onnx")
    onnx_filename = context.get("onnx_filename", "terramind_v1_base_onnx\terramind_v1_base_encoder.onnx")
    onnx_path = (Path(onnx_dir) / onnx_filename).as_posix()

    arr = cube.get_array()   # (bands, y, x) or (t, bands, y, x)
    if "t" in arr.dims:
        arr = arr.median(dim="t", skipna=True)

    sess = _load_session(onnx_path)

    s2 = np.nan_to_num(_select_bands(arr, S2_BAND_ORDER), nan=0.0)
    s1 = np.nan_to_num(_select_bands(arr, S1_BAND_ORDER), nan=0.0)

    in_x = arr.coords["x"].values
    in_y = arr.coords["y"].values
    H, W = s2.shape[-2:]
    if (H, W) != (PATCH, PATCH):
        pad_h = max(0, PATCH - H)
        pad_w = max(0, PATCH - W)
        s2 = np.pad(s2, ((0, 0), (0, pad_h), (0, pad_w)))
        s1 = np.pad(s1, ((0, 0), (0, pad_h), (0, pad_w)))

    emb = _encode_tile(sess, s2, s1)   # (768, 224, 224) — token-replicated

    # Match the original (unpadded) input footprint so edge tiles don't emit
    # embeddings over what was zero padding.
    emb = emb[:, :H, :W]

    out = xr.DataArray(
        emb,
        dims=["bands", "y", "x"],
        coords={
            "bands": [f"emb_{i:03d}" for i in range(emb.shape[0])],
            "y": in_y,
            "x": in_x,
        },
    )
    return XarrayDataCube(out)


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    # Output stays on the input pixel grid (see ``_encode_tile``), so we only
    # rename the 14 input bands to the 768 embedding bands.
    return metadata.rename_labels(
        dimension="bands",
        target=[f"emb_{i:03d}" for i in range(EMB_DIM)],
    )
