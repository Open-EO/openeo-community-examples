# OlmoEarth annual Sentinel-1/2 embeddings with native `predict_onnx`

A step-by-step [notebook](./olmoearth-annual-embeddings.ipynb) that builds twelve monthly Sentinel-1 and Sentinel-2 composites, runs Ai2's pretrained **OlmoEarth v1.2** through openEO's native **`predict_onnx`**, and produces **one annual embedding cube at 40 m**. **Nano is the default (128 bands)**; Tiny, Small and Base produce 192, 384 and 768 bands. Defaults cover a 2.56 × 2.56 km area near VITO in Mol, Belgium, for 2024.

An embedding is a reusable learned feature vector, **not a land-cover label**. All months and both modalities attend jointly and bidirectionally within each spatial tile before valid-only annual pooling. The notebook shows monthly RGB/radar observations, missing-data coverage, and an annual PCA map with a georeferenced RGB overlay. No retraining or downstream accuracy claim is involved.

## About OlmoEarth

[OlmoEarth](https://github.com/allenai/olmoearth_pretrain) is Ai2's family of **multimodal, spatiotemporal foundation models for Earth observation**. Its pretrained encoders transform satellite images or time series into reusable learned features for downstream tasks such as classification and segmentation. The official models support Sentinel-1, Sentinel-2 and Landsat imagery; this example uses **Sentinel-1 and Sentinel-2 only**.

The [official v1.2 model overview](https://github.com/allenai/olmoearth_pretrain#model-summary) lists the following encoder sizes:

| v1.2 variant | Encoder parameters | Output bands | Approx. ONNX size |
|---|---:|---:|---:|
| **Nano — default** | **1.7M** | **128** | 5.8 MB |
| Tiny | 12.5M | 192 | 31 MB |
| Small | 35.6M | 384 | 100 MB |
| Base | 114M | 768 | 368 MB |

Encoder parameter counts are upstream totals; ONNX sizes refer to this Sentinel-1/2 annual export. We default to **Nano**, the smallest v1.2 variant, for a compact CPU-based ONNX demonstration—not because it is necessarily the best model for every downstream task. Select another variant using the exporter/checker flags and notebook configuration below; use its matching ONNX and STAC pair. The older v1 Large model is outside this example's supported family.

We retain the pretrained weights. The openEO-compatible input interface and valid-only annual pooling are adaptations in this example, **not a separately trained official annual OlmoEarth model**. They turn twelve monthly Sentinel-1/2 observations into one feature vector per 40 m patch; the notebook visualizes those features with PCA rather than predicting land-cover classes.

For model details, see the [Nano model card](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano) and [OlmoEarth v1.2 technical report](https://huggingface.co/papers/2605.20804). The [restricted-use licensing terms](#licensing) apply to these model artifacts and derivatives.

## Requirements

- Python **3.12** and the dependencies below.
- A [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) account with openEO access and batch-job capacity.
- The demonstrated CDSE backend, with `SENTINEL2_L2A`, `SENTINEL1_GRD`, `to_scl_dilation_mask`, `sar_backscatter`, `aggregate_temporal`, `convert_data_type`, and experimental **STAC-aware `predict_onnx`**, plus artifact uploads.
- Read the [licensing section](#licensing) before using or sharing OlmoEarth artifacts or derivatives.

Other backends are **not** validated merely because they expose similar process names. No UDF, `udf-dependency-archives`, external-weight archive, GPU, or backend change is needed. Public hosting of the exported model is optional: the notebook uploads the local model as a temporary CDSE artifact.

## Quick start

From the repository root:

```bash
cd python/60_geospatial-embeddings/olmoearth
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install CPU PyTorch first so the one-off export does not pull a GPU stack.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-export.txt
python export_olmoearth_to_onnx.py
python check_olmoearth_onnx.py
jupyter lab olmoearth-annual-embeddings.ipynb
```

The no-argument commands still select **Nano**. The first export downloads only the selected variant's pinned official `config.json` and `weights.pth`, checks the source weight SHA-256, and writes a **single self-contained ONNX model** (~5.7 MB for Nano) plus matching STAC JSON. Later runs reuse the Hugging Face cache. Hashing streams files rather than reading the roughly 1 GB Base source checkpoint into another memory buffer. No dependency upgrade or external weights sidecar is needed.

### Select a variant

For example, to use Tiny:

```bash
python export_olmoearth_to_onnx.py --variant tiny
python check_olmoearth_onnx.py --variant tiny
```

Use `nano`, `tiny`, `small` or `base` for both commands, then set **`VARIANT = 'tiny'`** (or your selection) in the notebook configuration. Default filenames are `olmoearth-v1_2-<variant>-annual.onnx` and `.stac.json`. The exporter derives output width from the actual checkpoint, generates both flat/nested metadata and labels from the shared Nano baseline, and retains that variant's pinned provenance/license links. The runtime remains PyTorch-free; it validates the selected descriptor against ONNX metadata before upload.

`--output PATH` is supported by the exporter, and `--model PATH` by the checker **alongside the same `--variant`**. Put the resulting files at the notebook's configured paths if using non-default filenames. A mismatched variant or tensor declaration fails validation. Generated Tiny/Small/Base descriptors and all binary models are ignored by Git; only the shared Nano baseline is committed.

**Changing variants changes the feature space.** Do not mix vectors or reuse a classifier trained on another variant; retrain downstream models. All variants retain the same 32×32 input tiles, 320 m context, 4×4 patches, 40 m spacing and twelve-month preprocessing/pooling. Larger encoders do not attend beyond tile boundaries. Larger models require more download, memory and CPU resources; the notebook's 6 GB executor setting is not a guarantee for every variant. See the actual validation scope below.

The notebook itself uses only **`requirements.txt`**. To run it in a separate lightweight environment after exporting elsewhere:

```bash
python3.12 -m venv .venv-runtime
source .venv-runtime/bin/activate
python -m pip install -r requirements.txt
jupyter lab olmoearth-annual-embeddings.ipynb
```

Keep the ONNX file and matching STAC descriptor next to the notebook. Neither its helper nor its processing cells import PyTorch or OlmoEarth. Activate the matching environment/kernel in Jupyter. Windows users should use their virtual environment's `Scripts` activation path instead of `source`.

### Submit, pause, resume

1. Keep Mol/2024 defaults for the first run and leave `JOB_ID = None`.
2. Run the setup, OIDC authentication and processing cells in order. The graph is lazy; loading and compositing are deferred until execution.
3. Run upload/submission **once**, note your printed job ID, and **pause**. Submission does not wait. Do not expect a first uninterrupted “Run All” to finish the download cells while the asynchronous job is still running.
4. Rerun status later. When it is `finished`, continue to download, checks and plots.
5. In a new session, configure **your own** saved `JOB_ID`. Rerun setup/graph cells; upload and submission will be skipped. Reset the identifier if changing the **variant**, year or area, and rerun configuration and subsequent cells. The author's validation job cannot be reused by other accounts.

The model's presigned URL is kept only in memory and can expire. Upload again immediately before a new submission. **Do not print/save the populated STAC descriptor or process graph, or commit tokens, signed URLs, ONNX binaries, model caches or downloaded rasters.** Results and generated binaries are ignored by Git.

## What the graph teaches

The processing is visible in the notebook rather than hidden in a large helper:

1. Define **twelve half-open calendar-month intervals** and composite midpoint dates `[day, zero-based month, year]`, including leap-year February.
2. Load all twelve Sentinel-2 model bands in order: `B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`. Build a WorldCereal-style SCL erosion/dilation mask using a 1 km halo, apply it **before** the monthly median, and align bands to 10 m.
3. Load descending-orbit Sentinel-1 VV/VH, process sigma0-ellipsoid with COPERNICUS_30 and noise removal, align to the optical grid, compute monthly **linear-power means**, then convert to **dB**.
4. Pack each month as **12 S2 + 2 S1 + 3 date channels**, so twelve months become **204 float32 bands**. Assert positional order. The model reconstructs time internally; this is not twelve independent inference calls.
5. Read the selected STAC tensor contract, cross-check ONNX metadata, upload the artifact, explicitly call `cube.process('predict_onnx', ...)`, restore that model's output labels, and submit a variant-titled batch job.
6. Save and verify a reusable annual feature product, inspect observations/coverage, and visualize relative embedding variation with PCA.

CDSE already harmonizes Sentinel-2 baseline DN offsets: **do not subtract another 1000, convert DN to reflectance, or normalize again**. The official OlmoEarth computed normalization is baked into ONNX. There is **no WorldCereal uint16 radar encoding**. Composite midpoint dates are representative dates, not individual capture dates.

## Tensor contract and missing data

| Tensor | Shape | Dtype |
|---|---|---|
| Input `raster` | `[1, 204, 32, 32]` | float32 |
| Output `annual_embeddings` | `[1, D, 8, 8]` | float32 |

`D` is **128/192/384/768** for **Nano/Tiny/Small/Base**. Output names run from `embedding_000` to `embedding_<D-1>` (zero-padded to three digits); the selected descriptor supplies the exact label list.

The input grid is 10 m; a 4×4 patch produces a native 40 m feature vector. **STAC is required for downsampling on the demonstrated backend path**; a bare ONNX URL does not have the same contract. Nested MLM tensor metadata is intentionally duplicated in flat fields for the deployed parser. The descriptor labels the continuous feature output as generic `regression`, not as land-cover predictions.

One invalid pixel or band invalidates its complete modality/month patch. Its contents are safely filled before patch interpolation, excluded as attention keys, and excluded from temporal pooling. We average valid months within each modality, then available modalities; no evidence in either modality produces nodata. The fixed-slot attention adapter uses the official encoder components and is checked against the **unmodified** encoder's missing-token handling.

Default **32×32 input tiles** keep memory manageable. There is **no spatial overlap**, so context stops at 320 m tile boundaries and seams may be visible.

## Validation and grid handling

- **Local check:** after exporting, run `python check_olmoearth_onnx.py --variant <variant>` to check the tensor/STAC contract, normalization, missing-data handling and parity with the stock PyTorch encoder.
- **Notebook checks:** saved rasters are checked for CRS, dtype, labels, dates, nodata and numerical agreement with local ONNX Runtime. Packed inputs are exported with **`stitch=True` and `strict_cropping=False`** to retain full boundary inference context for this comparison.
- **Grid alignment:** the native 40 m output may extend less than one cell beyond the requested AOI. Use the saved raster's actual transform and bounds, not the requested AOI corner or an assumed array size. The plots crop inputs to that footprint without resampling.
- **Tested scope:** Nano on the default Mol 2024 footprint; all four variants on a 320 m subset. The saved notebook figures are Nano results. This does not establish large-area performance or downstream accuracy.

## Limits and interpretation

- **Scope limits:** arbitrary AOIs/years and an entirely absent acquisition month have not been validated end-to-end. Empty observations must stay missing, never be replaced by a neighbouring month's data.
- **Retrospective, not causal:** all months interact before annual pooling; do not use it as an as-of-date forecast without a separately designed strategy.
- **No seamlessness guarantee:** there is no tile overlap. Coarser-band/radar upsampling adds no independent detail.
- **Not a large-area export recipe:** stitched inputs, local full-raster parity and plots are for this small demonstration. A large production workflow needs a different export/validation strategy.
- **PCA is a visualization:** colors are not classes or accuracy evidence, and separately fitted AOI/year PCAs are not directly comparable. Classification requires a separate model and independent evaluation.

For download problems, check the network and trusted certificate configuration; **never disable certificate verification**. If a firewall blocks Hugging Face's Xet transport, `HF_HUB_DISABLE_XET=1` selects its HTTP path, but valid TLS trust is still required. For backend job failures, inspect logs in the openEO editor before resubmitting.

## Files

- `olmoearth-annual-embeddings.ipynb` — explicit pedagogical openEO workflow and plots.
- `openeo_olmoearth.py` — lightweight pinned model/band/contract/calendar/validity/result-check helpers; no PyTorch imports or hidden preprocessing.
- `export_olmoearth_to_onnx.py` — offline adapter, pinned official download, SHA-256 verification and ONNX export.
- `check_olmoearth_onnx.py` — independent stock-PyTorch/ONNX self-check.
- `olmoearth-v1_2-nano-annual.stac.json` — safe shared baseline; the exporter generates variant-specific contracts with relative artifact links and correct upstream license/provenance.
- `requirements.txt`, `requirements-export.txt`, `.gitignore` — separate notebook/export dependencies and generated-file exclusions.

## Provenance

- Official sources: [Nano](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/tree/e1f693ae2a7d5b57871a978e9d09e22d05206747), [Tiny](https://huggingface.co/allenai/OlmoEarth-v1_2-Tiny/tree/12a9fdbfeff905d7e147e7497f9f7a95c518eefc), [Small](https://huggingface.co/allenai/OlmoEarth-v1_2-Small/tree/5dd894799741b60f59dc4b1dbbb758d8afd89062), [Base](https://huggingface.co/allenai/OlmoEarth-v1_2-Base/tree/2e99a734a30e9aeb993aaa39946c6dcf554739e0).
- All immutable revisions and official `weights.pth` SHA-256 values are in the small [`MODELS` table](./openeo_olmoearth.py); every export verifies its selection. Default Nano SHA-256: `2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378`.
- Export: `olmoearth-pretrain==0.1.2`, CPU `torch==2.9.1`, opset 18; all weights embedded.
- Runtime: `openeo[artifacts]==0.53.0`, `onnxruntime==1.30.0`; tested with Python 3.12.
- Preprocessing reference: [WorldCereal's monthly composite graph](https://github.com/WorldCereal/worldcereal-classification/blob/temporary-global-udp-host/src/worldcereal/udp/global-production-udp-no-export-workspace-onnx.json). We reuse its masking/compositing approach, not its model or integer encoding.

## Licensing

This example uses **Ai2 OlmoEarth**. The OlmoEarth encoder, checkpoint, exported ONNX model and their derivatives retain the **[OlmoEarth Artifact License](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/blob/e1f693ae2a7d5b57871a978e9d09e22d05206747/LICENSE.txt)** and its restricted-use terms; they are **not relicensed under the community repository's Apache-2.0 license**.

The exported descriptor links to the selected variant's pinned license; switching variants does not remove these restrictions. In particular, the upstream terms restrict military/defense/intelligence/human-surveillance/policing uses and extractive activities including oil/gas/mineral extraction and deforestation. Attribution and downstream notice/restriction obligations also apply. Review the full license before using or distributing weights, adapted models or derived products. A repository license does not remove those upstream conditions.
