"""Small calendar/validation helpers; the openEO processing lives in the notebook.

OlmoEarth artifacts and derivatives retain the restrictions described in
README.md#licensing and https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE.
This module does not import PyTorch, OlmoEarth, or the offline exporter.
"""

from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rasterio

MODEL_NAME = "olmoearth-v1_2-nano-annual"
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
MONTH_BANDS = S2_BANDS + ["VV", "VH", "day", "month0", "year"]
INPUT_BANDS = [f"m{month:02d}_{band}" for month in range(1, 13) for band in MONTH_BANDS]
OUTPUT_BANDS = [f"embedding_{i:03d}" for i in range(128)]


def calendar_months(year: int) -> tuple[list[list[str]], list[list[int]]]:
    """Half-open calendar intervals and [day, zero-based month, year] midpoints."""
    starts = [date(year, month, 1) for month in range(1, 13)] + [date(year + 1, 1, 1)]
    intervals, timestamps = [], []
    for start, end in zip(starts, starts[1:]):
        midpoint = start + (end - start) // 2
        intervals.append([start.isoformat(), end.isoformat()])
        timestamps.append([midpoint.day, midpoint.month - 1, midpoint.year])
    return intervals, timestamps


def patch_validity(raw: np.ndarray) -> np.ndarray:
    """Monthly modality validity [12,2,H/4,W/4]: all pixels/bands must be finite."""
    assert raw.ndim == 3 and raw.shape[0] == 204
    h, w = raw.shape[-2:]
    assert h % 4 == w % 4 == 0
    months = raw.reshape(12, 17, h, w)
    validity = []
    for bands in [months[:, :12], months[:, 12:14]]:
        finite = np.isfinite(bands).all(axis=1)
        validity.append(finite.reshape(12, h // 4, 4, w // 4, 4).all(axis=(2, 4)))
    return np.stack(validity, axis=1)


def check_results(folder: Path, model_path: Path, extent: dict, year: int) -> None:
    """Check actual grids/masks and every output patch against local ONNX Runtime."""
    rasters, bounds = {}, {}
    for path in folder.glob("*.tif*"):
        with rasterio.open(path) as src:
            assert src.crs is not None and src.crs.to_epsg() == extent["crs"]
            assert all(dtype == "float32" for dtype in src.dtypes)
            assert src.count in {204, 128} and src.count not in rasters
            assert src.transform.b == src.transform.d == 0 and src.transform.e < 0
            if src.count == 128:
                assert src.descriptions == tuple(OUTPUT_BANDS)
            else:
                # CDSE's stitched writer drops descriptions; assert input order before submission.
                assert src.descriptions in {tuple(INPUT_BANDS), (None,) * 204}
            assert src.res == ((10, 10) if src.count == 204 else (40, 40))
            rasters[src.count] = src.read(masked=True).filled(np.nan)
            bounds[src.count] = src.bounds
            print(f"{path.name}: {src.count} bands, {src.width}x{src.height}, {src.res[0]:g} m, {tuple(src.bounds)}")
    assert set(rasters) == {204, 128}, "Expected packed monthly inputs and annual embeddings"
    inputs, actual = rasters[204], rasters[128]
    outer, inner = bounds[204], bounds[128]
    assert outer.left <= inner.left and outer.bottom <= inner.bottom
    assert outer.right >= inner.right and outer.top >= inner.top
    assert 0 <= extent["west"] - inner.left < 40 and 0 <= inner.right - extent["east"] < 40
    assert 0 <= extent["south"] - inner.bottom < 40 and 0 <= inner.top - extent["north"] < 40
    offsets = np.array([(outer.top - inner.top) / 40, (inner.left - outer.left) / 40])
    np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-8, rtol=0)
    row_offset, col_offset = np.round(offsets).astype(int)
    height, width = actual.shape[-2:]
    window = (slice(row_offset, row_offset + height), slice(col_offset, col_offset + width))
    _, timestamps = calendar_months(year)
    for month, timestamp in enumerate(timestamps):
        for index, value in enumerate(timestamp, start=14):
            np.testing.assert_array_equal(inputs[month * 17 + index], value)
    validity = patch_validity(inputs)[:, :, window[0], window[1]]
    evidence = validity.any(axis=(0, 1))
    np.testing.assert_array_equal(np.isfinite(actual).all(axis=0), evidence)
    assert not np.isinf(actual).any() and evidence.any()
    assert np.isnan(actual[:, ~evidence]).all()
    print("Valid 40 m patches by month [S2, S1]:", validity.sum(axis=(2, 3)).tolist())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "raster"
    assert session.get_inputs()[0].shape == [1, 204, 32, 32]
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].name == "annual_embeddings"
    assert session.get_outputs()[0].shape == [1, 128, 8, 8]
    assert session.get_outputs()[0].type == "tensor(float)"
    size = 32
    input_height, input_width = inputs.shape[-2:]
    assert input_height % size == input_width % size == 0, "Export full source tiles for parity"
    predicted = np.empty((128, input_height // 4, input_width // 4), dtype=np.float32)
    for row in range(0, input_height, size):
        for col in range(0, input_width, size):
            expected = session.run(None, {"raster": inputs[None, :, row:row + size, col:col + size]})[0]
            assert isinstance(expected, np.ndarray)
            predicted[:, row // 4:(row + size) // 4, col // 4:(col + size) // 4] = expected[0]
    expected = predicted[:, window[0], window[1]]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    maximum = np.nanmax(np.abs(actual - expected))
    print(f"Verified all {height * width} saved patches with full tile context; maximum ONNX error {maximum:.6g}")
