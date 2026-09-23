# 60 · Geospatial Embeddings

This section collects notebook examples for turning Earth observation data into compact, learned feature spaces. The examples span from classical dimensionality reduction to modern foundation-model style embeddings used as inputs for downstream ML tasks such as classification, clustering, and similarity search.

## What is included

These notebooks show different ways to compress or transform EO data:

- dimensionality reduction for feature selection and preprocessing
- custom backend processes for deep compression/decompression
- learned embeddings for pixel- or patch-level downstream tasks
- foundation-model-inspired feature extraction from Sentinel-1 and Sentinel-2 imagery

## Notebooks

| Example | Data source | Key openEO features | Description |
|---|---|---|---|
| [corsa/corsa-processes.ipynb](./corsa/corsa-processes.ipynb) | Sentinel-2 | custom backend processes | Apply CORSA deep-learning compression (100×) and decompression. |
| [dimensionality-reduction/dimensionality-reduction.ipynb](./dimensionality-reduction/dimensionality-reduction.ipynb) | Sentinel-2 | `apply_dimension`, UDF (PCA) | PCA-based dimensionality reduction for ML preprocessing. See the [subfolder README](./dimensionality-reduction/README.md). |
| [tessera/tessera-embedding.ipynb](./tessera/tessera-embedding.ipynb) | Sentinel-1 + Sentinel-2 | PyTorch UDF | Generate 128-band TESSERA pixel embeddings. See the [subfolder README](./tessera/README.md). |
| [terramind/terramind-embedding.ipynb](./terramind/terramind-embedding.ipynb) | Sentinel-1 + Sentinel-2 | ONNX UDF | Generate 768-band TerraMind pixel embeddings for downstream land-cover and geospatial ML workflows. See the [subfolder README](./terramind/README.md). |

## Choosing an example

- If you want a lightweight and explainable feature transform, start with [dimensionality-reduction](./dimensionality-reduction/).
- If you want a very compact learned representation with a custom backend process, use [corsa](./corsa/).
- If you want a task-agnostic embedding product built from EO data, use [tessera](./tessera/) or [terramind](./terramind/).

## Typical workflow

1. load and prepare EO data in openEO
2. build a feature cube or embedding cube with a UDF or backend process
3. materialize the result as a reusable intermediate product
4. feed it into classification, clustering, or downstream geospatial analysis

This folder is intended as a practical collection of embedding and feature-space examples that can be reused as building blocks in larger geospatial ML pipelines.
