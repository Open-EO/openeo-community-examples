#%%
"""Local UDF test: run the TerraMind ONNX UDF on input_terramind.nc.

Loads the local S2 + S1 merged NetCDF, extracts a 224x224 tile,
feeds it straight into ``udf_terramind_embedding.apply_datacube``,
and saves a preview of the first embedding band to
``tests/test_outputs/embedding_output.png``.

Requirements (installed in the local Python env, not via the openEO deps
archive)::

    pip install onnxruntime xarray netCDF4 matplotlib numpy openeo

This test expects a local ONNX model + external-weights sidecar in
``terramind_weights/terramind_v1_base/terramind_v1_base_onnx/``.

Usage::

    python tests/test_local_udf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _resolve_repo_root() -> Path:
    """Return the TerraMindEmbedding folder (the one holding the UDF + NetCDF)."""
    if "__file__" in globals():
        here = Path(__file__).resolve().parent
    else:
        here = Path.cwd().resolve()

    for candidate in (here, *here.parents):
        if (
            (candidate / "udf_terramind_embedding.py").is_file()
            and (candidate / "input_terramind.nc").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the TerraMindEmbedding folder (expected "
        "`udf_terramind_embedding.py` + `input_terramind.nc` side by side). "
        f"Searched from {here} upward."
    )


_REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(_REPO_ROOT))

from udf_terramind_embedding import apply_datacube
from openeo.udf import XarrayDataCube

# ---------------------------------------------------------------------------
# Config — tweak here
# ---------------------------------------------------------------------------
INPUT_NC = _REPO_ROOT / "input_terramind.nc"
OUT_DIR = _REPO_ROOT / "tests" / "test_outputs"
LOCAL_ONNX_DIR = (
    _REPO_ROOT
    / "terramind_weights"
    / "terramind_v1_base"
    / "terramind_v1_base_onnx"
)
LOCAL_ONNX_FILENAME = "terramind_v1_base_encoder.onnx"

TILE_SIZE = 224           # TerraMind encoder is architecturally 224x224
TILE_X_START = 0
TILE_Y_START = 0

# UDF context: point at the local ONNX (folder + filename), same keys the UDF
# reads on the backend (where the folder comes from a `udf-dependency-archives`
# alias instead of a local path).
CONTEXT = {
    "onnx_dir": str(LOCAL_ONNX_DIR),
    "onnx_filename": LOCAL_ONNX_FILENAME,
}


# ---------------------------------------------------------------------------
# Load input_terramind.nc → (bands, y, x) DataArray in raw units
# ---------------------------------------------------------------------------

def load_input_cube(path: Path) -> xr.DataArray:
    """Return a (bands, y, x) DataArray from the merged S2 + S1 NetCDF."""
    ds = xr.open_dataset(path)
    print(f"Variables : {list(ds.data_vars)}")
    print(f"Dims      : {dict(ds.sizes)}")
    print(f"Coords    : {list(ds.coords)}")

    _NON_BAND_VARS = {"crs"}

    if "bands" in ds.dims:
        var_name = next(v for v in ds.data_vars if v not in _NON_BAND_VARS)
        da = ds[var_name]
    else:
        band_names = [
            v for v in ds.data_vars
            if v not in _NON_BAND_VARS
            and np.issubdtype(ds[v].dtype, np.number)
            and {"y", "x"}.issubset(set(ds[v].dims))
        ]
        if not band_names:
            raise ValueError(f"No band-like variables found in {path}")
        da = xr.concat([ds[v] for v in band_names], dim="bands")
        da = da.assign_coords(bands=band_names)

    t_dim = next((d for d in da.dims if d in ("t", "time")), None)
    if t_dim is not None:
        da = da.squeeze(t_dim) if da.sizes[t_dim] == 1 else da.median(dim=t_dim)

    da = da.astype(np.float32)

    print(f"Cube shape: {da.shape}, dims={list(da.dims)}, dtype={da.dtype}")
    if "bands" in da.dims:
        for i, name in enumerate(da.coords["bands"].values):
            v = da.isel(bands=i).values
            valid = v[~np.isnan(v)]
            if valid.size:
                print(
                    f"  band {name}: min={valid.min():.3f}, "
                    f"max={valid.max():.3f}, mean={valid.mean():.3f}"
                )
    return da


def extract_tile(da: xr.DataArray, x0: int, y0: int, size: int) -> xr.DataArray:
    dims = list(da.dims)
    y_dim = next(d for d in dims if d in ("y", "lat", "latitude"))
    x_dim = next(d for d in dims if d in ("x", "lon", "longitude"))
    return da.isel({y_dim: slice(y0, y0 + size), x_dim: slice(x0, x0 + size)})


def main() -> None:
    print("=" * 60)
    print("LOCAL TERRAMIND UDF TEST")
    print("=" * 60)
    print(f"Loading {INPUT_NC}")

    cube = load_input_cube(INPUT_NC)
    tile = extract_tile(cube, TILE_X_START, TILE_Y_START, TILE_SIZE)
    print(f"\nTile: shape={tile.shape}, dims={list(tile.dims)}")

    print("\nRunning UDF...")
    result = apply_datacube(XarrayDataCube(tile), CONTEXT)
    result_da = result.get_array() if hasattr(result, "get_array") else result
    emb = result_da.values                          # (768, y_out, x_out)
    print(
        f"Embedding: shape={emb.shape}, dtype={emb.dtype}, "
        f"min={emb.min():.3f}, max={emb.max():.3f}, mean={emb.mean():.3f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # S2 RGB preview from B04/B03/B02 (or first 3 bands as fallback).
    band_names = list(tile.coords["bands"].values) if "bands" in tile.coords else []
    rgb_bands = ["B04", "B03", "B02"]
    if all(b in band_names for b in rgb_bands):
        s2_rgb = np.stack(
            [tile.sel(bands=b).values for b in rgb_bands], axis=-1
        ).astype(np.float32)
    else:
        s2_rgb = tile.values[:3].transpose(1, 2, 0).astype(np.float32)

    s2_rgb = np.nan_to_num(s2_rgb, nan=0.0)
    s2_display = np.zeros_like(s2_rgb)
    for i in range(3):
        band = s2_rgb[:, :, i]
        lo, hi = np.percentile(band, (2, 98))
        if hi > lo:
            s2_display[:, :, i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            s2_display[:, :, i] = band

    emb_band0 = emb[1]

    # K-means on the 768-d tokens -> label per 16x16 patch, upsampled to 224x224.
    from sklearn.cluster import KMeans

    n_clusters = 8
    n_bands, h_tok, w_tok = emb.shape
    tokens = emb.reshape(n_bands, -1).T             # (196, 768)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(tokens)
    label_map = labels.reshape(h_tok, w_tok)
    # Nearest-neighbour upsample to the S2 tile resolution (integer labels).
    scale_y = s2_display.shape[0] // h_tok
    scale_x = s2_display.shape[1] // w_tok
    label_up = np.kron(label_map, np.ones((scale_y, scale_x), dtype=label_map.dtype))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"TerraMind embedding — 224 -> {h_tok}x{w_tok} tokens",
        fontsize=13,
    )

    axes[0].imshow(s2_display)
    axes[0].set_title("Input S2 RGB (2–98% stretch)")
    axes[0].axis("off")

    im = axes[1].imshow(label_up, cmap="tab20", interpolation="nearest",
                        vmin=0, vmax=max(n_clusters - 1, 1))
    axes[1].set_title(f"K-means labels (k={n_clusters})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(s2_display)
    axes[2].imshow(label_up, cmap="tab20", alpha=0.55, interpolation="nearest",
                   vmin=0, vmax=max(n_clusters - 1, 1))
    axes[2].set_title(f"K-means on tokens (k={n_clusters}), upsampled")
    axes[2].axis("off")

    plt.tight_layout()
    out_png = OUT_DIR / "embedding_output.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    plt.show()


if __name__ == "__main__":
    main()

# %%
