"""Small calendar/validation helpers; the openEO processing lives in the notebook.

OlmoEarth artifacts and derivatives retain the restrictions described in
README.md#licensing and https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE.
This module does not import PyTorch, OlmoEarth, or the offline exporter.
"""

from datetime import date
from pathlib import Path

import onnxruntime as ort

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
