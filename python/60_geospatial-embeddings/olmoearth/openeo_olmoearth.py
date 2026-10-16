"""Small calendar/validation helpers; the openEO processing lives in the notebook.

OlmoEarth artifacts and derivatives retain the restrictions described in
README.md#licensing and https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE.
This module does not import PyTorch, OlmoEarth, or the offline exporter.
"""

from datetime import date
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rasterio

# Official immutable checkpoints; verify weights before exporting any variant.
MODELS = {
    "nano": {
        "repo": "allenai/OlmoEarth-v1_2-Nano",
        "revision": "e1f693ae2a7d5b57871a978e9d09e22d05206747",
        "sha256": "2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378",
    },
    "tiny": {
        "repo": "allenai/OlmoEarth-v1_2-Tiny",
        "revision": "12a9fdbfeff905d7e147e7497f9f7a95c518eefc",
        "sha256": "835c0b21ab010c4c4515faafa44dc1a41c9bc512d3a30af184803c4f4257697d",
    },
    "small": {
        "repo": "allenai/OlmoEarth-v1_2-Small",
        "revision": "5dd894799741b60f59dc4b1dbbb758d8afd89062",
        "sha256": "459796ed5680bc85926f9a0e023476d14cb637bc19f826575c43836c909a5fa6",
    },
    "base": {
        "repo": "allenai/OlmoEarth-v1_2-Base",
        "revision": "2e99a734a30e9aeb993aaa39946c6dcf554739e0",
        "sha256": "57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec",
    },
}
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
MONTH_BANDS = S2_BANDS + ["VV", "VH", "day", "month0", "year"]
INPUT_BANDS = [f"m{month:02d}_{band}" for month in range(1, 13) for band in MONTH_BANDS]


def model_name(variant: str) -> str:
    if variant not in MODELS:
        raise ValueError(f"Unknown OlmoEarth variant {variant!r}; choose from {list(MODELS)}")
    return f"olmoearth-v1_2-{variant}-annual"


def output_bands(embedding_size: int) -> list[str]:
    return [f"embedding_{i:03d}" for i in range(embedding_size)]


def validate_contract(stac: dict, variant: str) -> list[str]:
    """Validate the pinned source and both backend/MLM tensor declarations."""
    assert stac["id"] == model_name(variant), "STAC variant does not match the selection"
    source = MODELS[variant]
    url = f"https://huggingface.co/{source['repo']}"
    assert any(link["rel"] == "derived_from" and
               link["href"] == f"{url}/tree/{source['revision']}" for link in stac["links"])
    assert any(link["rel"] == "license" and
               link["href"] == f"{url}/blob/{source['revision']}/LICENSE.txt" for link in stac["links"])
    inp, out = stac["properties"]["mlm:input"], stac["properties"]["mlm:output"]
    assert len(inp) == len(out) == 1
    shape = out[0]["shape"]
    assert len(shape) == 4 and shape[0] == 1 and shape[2:] == [8, 8]
    assert isinstance(shape[1], int) and shape[1] > 0
    bands = output_bands(shape[1])
    for declared, nested, name, labels, expected in [
        (inp[0], "input", "raster", INPUT_BANDS, [1, 204, 32, 32]),
        (out[0], "result", "annual_embeddings", bands, shape),
    ]:
        assert declared["name"] == name and declared["bands"] == labels
        assert declared["shape"] == declared[nested]["shape"] == expected
        assert declared["data_type"] == declared[nested]["data_type"] == "float32"
        assert declared["dim_order"] == declared[nested]["dim_order"] == ["batch", "bands", "height", "width"]
    return bands


def validate_model(model_path: Path, embedding_size: int) -> None:
    """Check the exported ONNX interface before uploading it to the backend."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    for tensors, name, shape in [
        (session.get_inputs(), "raster", [1, 204, 32, 32]),
        (session.get_outputs(), "annual_embeddings", [1, embedding_size, 8, 8]),
    ]:
        if len(tensors) != 1:
            raise ValueError(f"Expected one ONNX tensor for {name}")
        tensor = tensors[0]
        if tensor.name != name or tensor.shape != shape or tensor.type != "tensor(float)":
            raise ValueError(f"Expected ONNX {name}: float32 {shape}")


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


def check_results(folder: Path, model_path: Path, extent: dict, year: int, variant: str = "nano") -> None:
    """Check actual grids/masks and every output patch against local ONNX Runtime."""
    bands = validate_contract(json.loads(model_path.with_suffix(".stac.json").read_text()), variant)
    embedding_size = len(bands)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == len(session.get_outputs()) == 1
    assert session.get_inputs()[0].name == "raster"
    assert session.get_inputs()[0].shape == [1, 204, 32, 32]
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].name == "annual_embeddings"
    assert session.get_outputs()[0].shape == [1, embedding_size, 8, 8]
    assert session.get_outputs()[0].type == "tensor(float)"
    rasters, bounds = {}, {}
    for path in folder.glob("*.tif*"):
        with rasterio.open(path) as src:
            assert src.crs is not None and src.crs.to_epsg() == extent["crs"]
            assert all(dtype == "float32" for dtype in src.dtypes)
            assert src.count in {204, embedding_size} and src.count not in rasters
            assert src.transform.b == src.transform.d == 0 and src.transform.e < 0
            if src.count == embedding_size:
                assert src.descriptions == tuple(bands)
            else:
                # CDSE's stitched writer drops descriptions; assert input order before submission.
                assert src.descriptions in {tuple(INPUT_BANDS), (None,) * 204}
            assert src.res == ((10, 10) if src.count == 204 else (40, 40))
            rasters[src.count] = src.read(masked=True).filled(np.nan)
            bounds[src.count] = src.bounds
            print(f"{path.name}: {src.count} bands, {src.width}x{src.height}, {src.res[0]:g} m, {tuple(src.bounds)}")
    assert set(rasters) == {204, embedding_size}, "Expected packed monthly inputs and annual embeddings"
    inputs, actual = rasters[204], rasters[embedding_size]
    outer, inner = bounds[204], bounds[embedding_size]
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
    size = 32
    input_height, input_width = inputs.shape[-2:]
    assert input_height % size == input_width % size == 0, "Export full source tiles for parity"
    predicted = np.empty((embedding_size, input_height // 4, input_width // 4), dtype=np.float32)
    for row in range(0, input_height, size):
        for col in range(0, input_width, size):
            expected = session.run(None, {"raster": inputs[None, :, row:row + size, col:col + size]})[0]
            assert isinstance(expected, np.ndarray)
            predicted[:, row // 4:(row + size) // 4, col // 4:(col + size) // 4] = expected[0]
    expected = predicted[:, window[0], window[1]]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    maximum = np.nanmax(np.abs(actual - expected))
    print(f"Verified all {height * width} saved patches with full tile context; maximum ONNX error {maximum:.6g}")
