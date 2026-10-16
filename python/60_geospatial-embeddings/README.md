# 60 · Geospatial Embeddings

These notebooks show how to turn Earth observation data into features for classification, clustering and similarity search. They cover PCA, learned compression, and embeddings from models trained on Sentinel-1 and Sentinel-2 imagery.

## Notebooks

| Example | Data source | Key openEO features | Description |
| --- | --- | --- | --- |
| [corsa/corsa-processes.ipynb](./corsa/corsa-processes.ipynb) | Sentinel-2 | custom backend processes | Apply CORSA deep-learning compression (100×) and decompression. |
| [dimensionality-reduction/dimensionality-reduction.ipynb](./dimensionality-reduction/dimensionality-reduction.ipynb) | Sentinel-2 | `apply_dimension`, UDF (PCA) | PCA-based dimensionality reduction for ML preprocessing. See the [subfolder README](./dimensionality-reduction/README.md). |
| [tessera/tessera-embedding.ipynb](./tessera/tessera-embedding.ipynb) | Sentinel-1 + Sentinel-2 | PyTorch UDF | Generate 128-band TESSERA pixel embeddings. See the [subfolder README](./tessera/README.md). |
| [terramind/terramind-embedding.ipynb](./terramind/terramind-embedding.ipynb) | Sentinel-1 + Sentinel-2 | ONNX UDF | Generate 768-band TerraMind pixel embeddings for downstream land-cover and geospatial ML workflows. See the [subfolder README](./terramind/README.md). |
| [olmoearth/olmoearth-annual-embeddings.ipynb](./olmoearth/olmoearth-annual-embeddings.ipynb) | Sentinel-1 + Sentinel-2 | monthly composites, native `predict_onnx`, STAC MLM | Make an annual OlmoEarth v1.2 feature map at 40 m from twelve monthly composites. Nano is the default with 128 bands; Tiny, Small and Base return 192, 384 and 768. See the [README](./olmoearth/README.md) for setup and the model's restricted-use license. |

## Choosing an example

- Start with [dimensionality-reduction](./dimensionality-reduction/) for a small, explainable feature transform using PCA.
- Try [corsa](./corsa/) for learned compression through a custom backend process.
- [tessera](./tessera/) and [terramind](./terramind/) extract features for use across different tasks.
- [olmoearth](./olmoearth/) combines a year's monthly Sentinel-1/2 data into one feature cube. It shows the preprocessing and uses native ONNX inference.

## Native `predict_onnx` or a UDF?

Use native `predict_onnx` if the backend supports the model's input layout, data types and output grid. Prepare the inputs with openEO processes and, where supported, describe the model with STAC MLM. That's what the [OlmoEarth example](./olmoearth/) does, so there's no Python UDF or dependency archive to package.

Use a UDF for tensor layouts, model calls or libraries that the native processes don't support. It can run ONNX Runtime, PyTorch or other libraries available in the backend runtime or packaged with the UDF. [TerraMind](./terramind/) and [TESSERA](./tessera/) show two ways to do this. Check the library versions and make sure the output bands and coordinates are correct.

For models that need overlapping windows, check how the process handles them. `apply_neighborhood` can pass surrounding pixels to a UDF. Check how the result is cropped and placed on the output grid. The CDSE `predict_onnx` process used here has no overlap option. Native inference isn't automatically faster or more accurate.

## Typical workflow

1. Load and prepare the satellite data in openEO.
2. Run the feature transform or embedding model with a UDF or backend process.
3. Save the feature cube for later use.
4. Use it for classification, clustering or another analysis.
