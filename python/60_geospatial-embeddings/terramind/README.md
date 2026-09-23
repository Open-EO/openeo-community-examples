# TerraMind patch embeddings from openEO-loaded Sentinel-1/2

Computes [TerraMind 1.0](https://github.com/IBM/terramind) (IBM + ESA's any-to-any generative
foundation model for Earth Observation) patch embeddings for a custom AOI and time period
directly from openEO. The TerraMind encoder runs inside an openEO UDF as an **ONNX model**,
so no PyTorch or TerraTorch is needed in the UDF sandbox — only `onnxruntime`.

The output is a **768-band embedding cube** on the input pixel grid that can be materialised
to NetCDF and reused as a task-agnostic feature source for land-cover classification, crop-type
mapping, change detection, similarity search, etc.

## Requirements

- An openEO backend with `SENTINEL2_L2A` and `SENTINEL1_GRD` collections (this example targets
  the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) openEO endpoint).
- A backend that supports the `udf-dependency-archives` job option (check with your backend
  provider).
- Offline: a Python environment with `terratorch` + `torch` + `onnx` for the one-off ONNX export.

## Architecture

```
SENTINEL2_L2A (12 L2A bands + SCL cloud mask, temporal median)  ─┐
                                                                 ├─ merge_cubes ─► apply_neighborhood(224x224, UDF)
SENTINEL1_GRD (VV, VH sigma0-ellipsoid, temporal median)        ─┘                                │
                                                                                                  ▼
                                                                        ONNX TerraMind encoder (standardization baked in)
                                                                                                  │
                                                                                                  ▼
                                                                    768-band embedding cube on the input pixel grid
```

## Two-step workflow

1. **Offline, once**: export the TerraMind encoder to ONNX with
   [`export_terramind_to_onnx.py`](./export_terramind_to_onnx.py). The script:
   - downloads the pretrained weights from Hugging Face (`ibm-esa-geospatial/TerraMind-1.0-*`),
   - builds the backbone via `terratorch`,
   - bakes S2 L2A and S1 GRD pre-training mean/std into the ONNX graph as constants, so the
     UDF hands raw DN and linear power directly to the model,
   - exports to `.onnx` + `.onnx.data` (external weights).

   Zip the two files together (matching the `onnx_dir` / `onnx_filename` layout the UDF expects)
   and upload the zip to somewhere the backend can reach over HTTPS.

2. **Online**: [`TerraMindEmbedding.ipynb`](./TerraMindEmbedding.ipynb) loads S1/S2 via openEO,
   runs [`udf_terramind_embedding.py`](./udf_terramind_embedding.py) with the ONNX archive plus
   the shared `onnxruntime` archive supplied through `udf-dependency-archives`, saves the
   768-band cube as NetCDF, and sanity-checks it with an unsupervised K-means clustering.

## Why materialise the embedding as its own cube?

- **Reusable intermediate.** The 768-band cube is task-agnostic. Once produced, the same cube
  can feed a land-cover classifier, a crop-type model, a change-detection routine, or a
  similarity-search index.


## Files

- `export_terramind_to_onnx.py` — offline ONNX export helper (run cell-by-cell).
- `udf_terramind_embedding.py` — the openEO UDF: loads the ONNX session, encodes 224×224 tiles,
  upsamples the 14×14 token grid back to the input pixel grid so `apply_neighborhood`'s
  pixel-index overlap trim keeps working.
- `TerraMindEmbedding.ipynb` — end-to-end example: load S2 L2A + S1 GRD, run the UDF, cluster
  the embedding with K-means, overlay on the S2 RGB.

## Caveats

- **TerraMind is single-timestep**: the notebook reduces S2 and S1 to a temporal median before
  running the encoder. A BAP composite works equally well and can be swapped in.
- **Standardization is baked into the ONNX graph** using TerraMind's published pretraining
  mean/std for the `untok_sen2l2a@224` / `untok_sen1grd@224` modalities. Do not rescale the
  inputs in the UDF — hand raw S2 L2A DN and S1 GRD linear power straight to the model.
- **Native spatial support is ~160 m** (14×14 tokens over a 224×224 tile). The UDF nearest-
  neighbour upsamples tokens back to the input pixel grid so `apply_neighborhood` overlap trim
  works; downstream steps can `resample_spatial` back down to ~160 m to shrink the cube.
- **Fine-tuning**: retrain / LoRA-adapt with `terratorch fit` offline, re-export with the same
  script, point the `TERRAMIND_ONNX_ARCHIVE` URL at the new zip: nothing else changes.
