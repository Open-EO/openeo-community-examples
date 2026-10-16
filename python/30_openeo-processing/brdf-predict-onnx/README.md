# BRDF normalization with `predict_onnx` (no UDF)

Runs custom per-pixel math on openEO as an **ONNX model** instead of a Python UDF.
The math is written with numpy-style code in [ndonnx](https://github.com/Quantco/ndonnx),
exported to ONNX, tested locally with `onnxruntime`, and run on the backend with the
experimental `predict_onnx` process.

The example implements the Roy et al. (2016/2017) c-factor BRDF normalization (NBAR) used by
[sen2like](https://github.com/senbox-org/sen2like), intended to reduce the effects of sun and
viewing geometry on Sentinel-2 reflectance.

> **Scientific limitation: ONNX demonstration, not validated NBAR.** The CDSE
> Sentinel-2 angle bands used here currently contain tile-level mean sun and viewing
> angles, not spatially resolved geometry. The same viewing angles are used for all
> reflectance bands, rather than band/detector-specific angles. This limits the
> accuracy of the BRDF correction; do not treat the output as scientifically validated
> NBAR. Accurate viewing-angle handling is tracked in
> [openeo-geotrellis-extensions#916](https://github.com/Open-EO/openeo-geotrellis-extensions/issues/916).

## Files

| File | Content |
|---|---|
| [`brdf-predict-onnx.ipynb`](./brdf-predict-onnx.ipynb) | Build the model with ndonnx, test it locally, run it with `predict_onnx`, and compare with the same algorithm as a Python UDF |
| [`roy_brdf.py`](./roy_brdf.py) | Plain numpy version of the algorithm (local reference) plus the Python UDF entrypoint (`apply_datacube`) |
| [`roy_brdf_ndonnx.stac.json`](./roy_brdf_ndonnx.stac.json) | STAC MLM Item describing the generated ONNX model and its input/output contract |

## Requirements

- An openEO backend that supports `predict_onnx` (this example targets the
  [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)).
- Locally: `pip install "openeo[artifacts]" ndonnx "onnxruntime>=1.18" rasterio matplotlib` (ndonnx needs `numpy>=2`).

## When to choose `predict_onnx` over a UDF

- The computation is fixed per-pixel or per-tile math that ONNX can express.
- You want to test exactly what will run on the backend before spending credits.
- You don't want to ship Python dependencies to the backend.

Stay with a UDF when you need arbitrary Python, per-job parameters (`predict_onnx` has no
`context`), or access to band names and coordinates.

## Model contract

- Input `(11, 32, 32)` float64: `B02 B03 B04 B8A B11 B12`, `sunZenithAngles`,
  `viewZenithMean`, `sunAzimuthAngles`, `viewAzimuthMean`, `theta_s` (reference sun zenith).
- Output `(6, 32, 32)` float64: the corrected reflectance for the 6 bands.
- Bands are matched by position. The notebook uploads its locally exported ONNX file as a
  temporary CDSE Artifact, replaces the STAC model asset's placeholder with a presigned URL,
  and passes the STAC Item as JSON to `predict_onnx`. Rerun the upload for later jobs; do not
  share the presigned URL.
- The STAC file includes both the nested MLM tensor fields and the flat fields required by
  the current CDSE backend.
