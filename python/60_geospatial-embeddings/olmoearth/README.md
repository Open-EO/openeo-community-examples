# OlmoEarth annual Sentinel-1/2 embeddings with native `predict_onnx`

This [notebook](./olmoearth-annual-embeddings.ipynb) makes an annual map of OlmoEarth features from twelve monthly Sentinel-1 and Sentinel-2 composites. It runs Ai2's pretrained OlmoEarth v1.2 through openEO's native `predict_onnx`. The default example uses Nano over a 2.56 × 2.56 km area in **Mol, Belgium, in 2024**, with output pixels at **40 m**.

Each output pixel holds a feature vector, not a land-cover label. The plots show monthly data, coverage and a PCA view of the annual features. The feature cube can be used for classification, clustering or similarity analysis. Models built on these features need training and evaluation for the intended task.

## About OlmoEarth

[OlmoEarth](https://github.com/allenai/olmoearth_pretrain) is a family of Earth-observation models from Ai2. The models use observations from different dates and support Sentinel-1, Sentinel-2 and Landsat. This example uses the two Sentinel sensors.

| v1.2 variant | Encoder parameters | Output bands | Approx. ONNX size |
|---|---:|---:|---:|
| **Nano — default** | **1.7M** | **128** | 5.8 MB |
| Tiny | 12.5M | 192 | 31 MB |
| Small | 35.6M | 384 | 100 MB |
| Base | 114M | 768 | 368 MB |

The parameter counts are from the [official overview](https://github.com/allenai/olmoearth_pretrain#model-summary). File sizes are for the ONNX exports used here. Nano is the smallest export and the default for this CPU example. Larger variants see the same input tiles but need more memory and compute. Switching variants requires retraining any model built on their features.

The pretrained weights remain unchanged. The example adds an openEO input layout and averages valid monthly features to make annual vectors. This is not an official annual-output model trained by Ai2. Before averaging, the encoder lets all months and both sensors attend to each other within a tile. See the [Nano model card](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano) and [v1.2 report](https://huggingface.co/papers/2605.20804) for more on the pretrained model.

## Requirements

This example needs Python **3.12** and a [CDSE](https://dataspace.copernicus.eu/) account with openEO access and enough capacity to run a batch job. The example uses CDSE's Sentinel-1/2 collections, cloud masking, SAR processing, experimental STAC support in `predict_onnx`, and artifact uploads. Check those capabilities before using another backend.

No UDF, Python dependency archive or GPU is required. The notebook uploads the ONNX file as a temporary CDSE artifact. Read [Licensing](#licensing) before using or sharing the model or anything derived from it.

## Quick start

### 1. Export and check the model once

From the repository root:

```bash
cd python/60_geospatial-embeddings/olmoearth
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-export.txt
python export_olmoearth_to_onnx.py --variant nano
python check_olmoearth_onnx.py --variant nano
```

The export script downloads a fixed revision of the selected checkpoint, checks its SHA-256 hash, and writes an ONNX file with all weights included. It also writes the matching STAC description. The check script compares the export with the original PyTorch encoder and tests normalization, missing data, calendar handling and input/output shapes. Both scripts run locally, not as UDFs on the backend.

To use Tiny, Small or Base, pass `--variant tiny`, `small` or `base` to both commands, then choose the same variant in the notebook. The files are named `olmoearth-v1_2-<variant>-annual.onnx` and `.stac.json`. For different paths, use `--output PATH` for the export and `--model PATH` for the check, keeping `--variant` the same.

### 2. Run the notebook

The export environment can also run the notebook. For a separate environment with just the notebook dependencies:

```bash
python3.12 -m venv .venv-runtime
source .venv-runtime/bin/activate
python -m pip install -r requirements.txt
jupyter lab olmoearth-annual-embeddings.ipynb
```

Start Jupyter in this directory and select that environment's kernel. Keep the ONNX file and its matching STAC JSON beside the notebook. The notebook doesn't import PyTorch or OlmoEarth. On Windows, activate the environment through its `Scripts` directory.

### 3. Submit, pause, resume

1. For a first run, keep the defaults and leave `JOB_ID = None`.
2. Run through the submission cell once, copy the printed job ID, then wait. The job continues on the backend after the cell returns.
3. Check its status later. When it says `finished`, continue with the download and plots. If it says `error` or `canceled`, read the logs before trying again.
4. To pick up the job in a new session, put its ID in `JOB_ID` and rerun the setup and graph cells. They won't upload or submit again. Clear the ID when changing the model, area or year.

On the first run, “Run All” will normally stop at the download cell because the job hasn't finished yet. Model URLs expire, so upload just before submitting a new job. **Don't print or save the filled-in STAC descriptor, process graph, tokens or signed URLs.** Git ignores generated models, caches and downloaded rasters.

## Processing

The notebook includes the code for each step:

1. Split the year into calendar months and give each a midpoint date, written as `[day, zero-based month, year]`.
2. Load all twelve Sentinel-2 bands, mask them with SCL erosion/dilation before taking monthly medians, and align them to 10 m.
3. Load descending Sentinel-1 VV/VH with sigma0-ellipsoid, COPERNICUS_30 and noise removal. Take monthly means in linear power, then convert to dB.
4. Put each month's **12 S2 bands, 2 S1 bands and 3 date bands** into one raster: 204 float32 bands, in calendar order.
5. Check the model against its STAC description, upload it, run `predict_onnx`, and set the output band labels.
6. Save the annual features and monthly inputs, then plot coverage and PCA.

Keep the input units as they are. CDSE has already corrected Sentinel-2 DN offsets, so **don't subtract another 1000 or divide by 10000**. Don't normalize again either: that's already in the ONNX model. Sentinel-1 inputs should be dB values, not integer-encoded radar data. The dates mark composite midpoints, not individual image captures.

## Tensor contract and missing data

| Tensor | Shape | Dtype |
|---|---|---|
| Input `raster` | `[1, 204, 32, 32]` | float32 |
| Output `annual_embeddings` | `[1, D, 8, 8]` | float32 |

`D` is 128, 192, 384 or 768 for Nano, Tiny, Small or Base. Pass the matching STAC description to `predict_onnx` so the backend applies the correct output grid. Each 40 m output pixel represents a 4×4 patch of 10 m input pixels.

Patches with any missing pixel or band are skipped for that sensor and month. Annual pooling averages valid months within each sensor, then the sensors with data. Patches with no data from either sensor produce nodata.

## Checks and limitations

Backend tests cover Nano on the Mol 2024 example and all four variants on a 320 m subset.

- Inference uses fixed 320 m tiles without overlap, so seams are possible.
- Full-year features include future observations for predictions made partway through the year.
- PCA plots show feature variation, not land-cover labels or accuracy.
- The resource settings are for a small example; larger areas may need adjustments.

## Files and provenance

| File | Purpose |
|---|---|
| `olmoearth-annual-embeddings.ipynb` | openEO workflow and plots |
| `export_olmoearth_to_onnx.py` | Download the model, adapt its inputs and export it locally |
| `check_olmoearth_onnx.py` | Compare the export with the original encoder and run tests |
| `openeo_olmoearth.py` | Band names, dates and checks shared by the notebook and scripts |
| `olmoearth-v1_2-nano-annual.stac.json` | Nano model description; the exporter writes matching files for other variants |
| `requirements.txt` / `requirements-export.txt` | Notebook dependencies / additional export dependencies |

The [`MODELS` table](./openeo_olmoearth.py) lists the exact upstream revision and weight hash for each variant. Exported STAC files link to that revision and its license. The masking and compositing follow [this reference graph](https://github.com/WorldCereal/worldcereal-classification/blob/temporary-global-udp-host/src/worldcereal/udp/global-production-udp-no-export-workspace-onnx.json).

Tested with Python 3.12, `olmoearth-pretrain==0.1.2`, CPU `torch==2.9.1`, ONNX opset 18, `openeo[artifacts]==0.53.0` and `onnxruntime==1.30.0`. Binary models and results aren't committed.

## Licensing

This example uses **Ai2 OlmoEarth**. The OlmoEarth encoder, checkpoint, exported ONNX model and their derivatives retain the **[OlmoEarth Artifact License](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/blob/e1f693ae2a7d5b57871a978e9d09e22d05206747/LICENSE.txt)** and its restricted-use terms; they are **not relicensed under the community repository's Apache-2.0 license**.

The exported descriptor links to the selected variant's pinned license; switching variants does not remove these restrictions. In particular, the upstream terms restrict military/defense/intelligence/human-surveillance/policing uses and extractive activities including oil/gas/mineral extraction and deforestation. Attribution and downstream notice/restriction obligations also apply. Review the full license before using or distributing weights, adapted models or derived products. A repository license does not remove those upstream conditions.
