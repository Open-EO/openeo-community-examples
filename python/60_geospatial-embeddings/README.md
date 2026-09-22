# 60 · Geospatial Embeddings

Notebooks about compressing, reducing, or embedding EO data into learned feature spaces — either as a preprocessing step for ML or as a first-class product.

## Notebooks

| Notebook | Data source | Key openEO features | Description |
|---|---|---|---|
| [corsa/corsa-processes.ipynb](./corsa/corsa-processes.ipynb) | Sentinel-2 | custom backend processes | Apply CORSA deep-learning compression (100×) and decompression. |
| [dimensionality-reduction/dimensionality-reduction.ipynb](./dimensionality-reduction/dimensionality-reduction.ipynb) | Sentinel-2 | `apply_dimension`, UDF (PCA) | PCA-based dimensionality reduction as ML preprocessing. See [subfolder README](./dimensionality-reduction/README.md). |
| [tessera/tessera-embedding.ipynb](./tessera/tessera-embedding.ipynb) | Sentinel-1 + Sentinel-2 | PyTorch UDF | Generate 128-band TESSERA pixel embeddings for downstream land-cover ML. See [subfolder README](./tessera/README.md). |
