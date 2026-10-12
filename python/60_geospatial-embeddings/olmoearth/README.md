# OlmoEarth annual Sentinel-1/2 embeddings with native `predict_onnx`

A step-by-step [notebook](./olmoearth-annual-embeddings.ipynb) that builds twelve monthly Sentinel-1 and Sentinel-2 composites, runs Ai2's pretrained **OlmoEarth v1.2 Nano** through openEO's native **`predict_onnx`**, and produces **one annual 128-band embedding cube at 40 m**. Defaults cover a 2.56 × 2.56 km area near VITO in Mol, Belgium, for 2024.

An embedding is a reusable learned feature vector, **not a land-cover label**. All months and both modalities attend jointly and bidirectionally within each spatial tile before valid-only annual pooling. The notebook shows monthly RGB/radar observations, missing-data coverage, and an annual PCA map with a georeferenced RGB overlay. No retraining or downstream accuracy claim is involved.

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

The first export downloads only the pinned official `config.json` and `weights.pth`, checks the source weight SHA-256, and writes a **single self-contained ~5.7 MB ONNX model** plus its STAC JSON. Later runs reuse the Hugging Face cache. `--output PATH` is supported by the exporter, and `--model PATH` by the checker; put the resulting files at the notebook's configured paths if using non-default filenames.

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
5. In a new session, configure **your own** saved `JOB_ID`. Rerun setup/graph cells; upload and submission will be skipped. Reset the identifier if changing the year or area. The author's validation job cannot be reused by other accounts.

The model's presigned URL is kept only in memory and can expire. Upload again immediately before a new submission. **Do not print/save the populated STAC descriptor or process graph, or commit tokens, signed URLs, ONNX binaries, model caches or downloaded rasters.** Results and generated binaries are ignored by Git.

## What the graph teaches

The processing is visible in the notebook rather than hidden in a large helper:

1. Define **twelve half-open calendar-month intervals** and composite midpoint dates `[day, zero-based month, year]`, including leap-year February.
2. Load all twelve Sentinel-2 model bands in order: `B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09`. Build a WorldCereal-style SCL erosion/dilation mask using a 1 km halo, apply it **before** the monthly median, and align bands to 10 m.
3. Load descending-orbit Sentinel-1 VV/VH, process sigma0-ellipsoid with COPERNICUS_30 and noise removal, align to the optical grid, compute monthly **linear-power means**, then convert to **dB**.
4. Pack each month as **12 S2 + 2 S1 + 3 date channels**, so twelve months become **204 float32 bands**. Assert positional order. The model reconstructs time internally; this is not twelve independent inference calls.
5. Read the committed STAC tensor contract, upload the ONNX artifact, explicitly call `cube.process('predict_onnx', ...)`, restore 128 output labels, and submit a batch job.
6. Save and verify a reusable annual feature product, inspect observations/coverage, and visualize relative embedding variation with PCA.

CDSE already harmonizes Sentinel-2 baseline DN offsets: **do not subtract another 1000, convert DN to reflectance, or normalize again**. The official OlmoEarth computed normalization is baked into ONNX. There is **no WorldCereal uint16 radar encoding**. Composite midpoint dates are representative dates, not individual capture dates.

## Tensor contract and missing data

| Tensor | Shape | Dtype |
|---|---|---|
| Input `raster` | `[1, 204, 32, 32]` | float32 |
| Output `annual_embeddings` | `[1, 128, 8, 8]` | float32 |

The input grid is 10 m; a 4×4 patch produces a native 40 m feature vector. **STAC is required for downsampling on the demonstrated backend path**; a bare ONNX URL does not have the same contract. Nested MLM tensor metadata is intentionally duplicated in flat fields for the deployed parser. The descriptor labels the continuous feature output as generic `regression`, not as land-cover predictions.

One invalid pixel or band invalidates its complete modality/month patch. Its contents are safely filled before patch interpolation, excluded as attention keys, and excluded from temporal pooling. We average valid months within each modality, then available modalities; no evidence in either modality produces nodata. The fixed-slot attention adapter uses the official encoder components and is checked against the **unmodified** encoder's missing-token handling.

Default **32×32 input tiles** keep memory manageable. A 64×64 annual input required about 6.4 GiB in an earlier ONNX-only check; reducing output to one annual map does not remove the twelve-month attention workload. There is **no spatial overlap**, so context stops at 320 m tile boundaries and seams may be visible.

## Validation and grid handling

`check_olmoearth_onnx.py` asserts the calendar, tensor/STAC contract, official normalization, valid-only pooling, complete/partial/single-modality/all-missing cases, and invalid-patch-content invariance. It compares the dense adapter and ONNX Runtime with stock PyTorch encoder outputs using independent pooling (`rtol=1e-3`, `atol=1e-4`).

The notebook saves both packed inputs and embeddings, then checks the actual CRS, dtype, labels, date channels, nodata and every output patch against local ONNX Runtime. Input export uses **`stitch=True` plus `strict_cropping=False`**: cropped inputs lose boundary inference context, and disabling strict cropping alone did not preserve full tiles on the tested writer. The stitched writer can produce `.tiff` and omit input descriptions; the graph order and raster dates are checked independently, while output labels remain mandatory.

The native 40 m output grid may extend less than one cell beyond the requested AOI. Do not assign the requested AOI corner to it, assume a 64×64 result, or resample inputs just to make arrays match. Validation reconstructs full source-tile predictions and selects the saved output's real grid window. The plots crop inputs to that footprint without resampling.

The standalone example was validated on CDSE on **2026-10-12**, after export/checks in a fresh Python 3.12 environment and notebook execution in a separate environment **without PyTorch or OlmoEarth installed**. The fresh batch job produced **128 × 65 × 65** embeddings in EPSG:32631; all **4,225** saved patches matched local ONNX Runtime, with maximum absolute difference **1.37091e-6**. Local adapter/stock-encoder comparisons passed all five cases (maximum **2.58287e-7**). The saved notebook contains the four figures and defaults to `JOB_ID = None`, not the author's job.

An uncached Hugging Face CDN download was blocked by this test environment's certificate chain, so the export check used the official cached checkpoint after SHA-256 verification, without disabling TLS checks. The pinned configuration download succeeded; a fresh end-to-end weight download was not verified here.

## Limits and interpretation

- **Verified scope:** Mol 2024 defaults. Arbitrary AOIs/years and an entirely absent acquisition month have not been validated end-to-end. Empty observations must stay missing, never be replaced by a neighbouring month's data.
- **Retrospective, not causal:** all months interact before annual pooling; do not use it as an as-of-date forecast without a separately designed strategy.
- **No seamlessness guarantee:** there is no tile overlap. Coarser-band/radar upsampling adds no independent detail.
- **Not a large-area export recipe:** stitched inputs, local full-raster parity and plots are for this small demonstration. A large production workflow needs a different export/validation strategy.
- **PCA is a visualization:** colors are not classes or accuracy evidence, and separately fitted AOI/year PCAs are not directly comparable. Classification requires a separate model and independent evaluation.

For download problems, check the network and trusted certificate configuration; **never disable certificate verification**. If a firewall blocks Hugging Face's Xet transport, `HF_HUB_DISABLE_XET=1` selects its HTTP path, but valid TLS trust is still required. For backend job failures, inspect logs in the openEO editor before resubmitting.

## Files

- `olmoearth-annual-embeddings.ipynb` — explicit pedagogical openEO workflow and plots.
- `openeo_olmoearth.py` — lightweight band/calendar/validity/result-check helpers; no PyTorch imports or hidden preprocessing.
- `export_olmoearth_to_onnx.py` — offline adapter, pinned official download, SHA-256 verification and ONNX export.
- `check_olmoearth_onnx.py` — independent stock-PyTorch/ONNX self-check.
- `olmoearth-v1_2-nano-annual.stac.json` — safe committed model contract, relative artifact link, upstream license/provenance links.
- `requirements.txt`, `requirements-export.txt`, `.gitignore` — separate notebook/export dependencies and generated-file exclusions.

## Provenance

- Source: [Ai2 OlmoEarth v1.2 Nano](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano), revision `e1f693ae2a7d5b57871a978e9d09e22d05206747`.
- Official `weights.pth` SHA-256: `2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378`.
- Export: `olmoearth-pretrain==0.1.2`, CPU `torch==2.9.1`, opset 18; all weights embedded.
- Runtime: `openeo[artifacts]==0.53.0`, `onnxruntime==1.30.0`; tested with Python 3.12.
- Preprocessing reference: [WorldCereal's monthly composite graph](https://github.com/WorldCereal/worldcereal-classification/blob/temporary-global-udp-host/src/worldcereal/udp/global-production-udp-no-export-workspace-onnx.json). We reuse its masking/compositing approach, not its model or integer encoding.

## Licensing

This example uses **Ai2 OlmoEarth**. The OlmoEarth encoder, checkpoint, exported ONNX model and their derivatives retain the **[OlmoEarth Artifact License](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/blob/e1f693ae2a7d5b57871a978e9d09e22d05206747/LICENSE.txt)** and its restricted-use terms; they are **not relicensed under the community repository's Apache-2.0 license**.

In particular, the upstream terms restrict military/defense/intelligence/human-surveillance/policing uses and extractive activities including oil/gas/mineral extraction and deforestation. Attribution and downstream notice/restriction obligations also apply. Review the full license before using or distributing weights, adapted models or derived products. A repository license does not remove those upstream conditions.
