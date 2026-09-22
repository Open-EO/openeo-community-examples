# 40 · Machine Learning

This section shows how to train and apply machine learning models in openEO workflows. The focus is on feature engineering, backend-side inference, and reusable model deployment patterns using UDFs and UDPs.

## What this section covers

- training ML models on EO data without moving everything locally
- building task-specific feature cubes from Sentinel imagery and ancillary data
- deploying models with ONNX or UDP-based inference patterns
- reusing trained models for large-scale prediction jobs

## Notebooks

| Notebook | Data source | Key openEO features | Description |
|---|---|---|---|
| [dynamic-land-cover-mapping/dynamic-land-cover-mapping.ipynb](./dynamic-land-cover-mapping/dynamic-land-cover-mapping.ipynb) | Sentinel-2 + SAR | UDFs for features, `MlModel` API | Train a Random Forest for dynamic land cover and run inference. |
| [parcel-delineation/parcel-delineation.ipynb](./parcel-delineation/parcel-delineation.ipynb) | Sentinel-2 | UDF, pretrained U-Net | Deploy a U-Net to delineate agricultural parcel boundaries. |
| [onnx-inference/onnx-ml-inference.ipynb](./onnx-inference/onnx-ml-inference.ipynb) | Sentinel-2 | UDF with ONNX runtime | Run a pretrained CNN in ONNX format at datacube scale. |
| [random-forest-forest-fire/random-forest-training.ipynb](./random-forest-forest-fire/random-forest-training.ipynb) | Sentinel-2 + SAR | GLCM UDFs, RF training, model persistence | Full training workflow for a forest-fire Random Forest model. |
| [random-forest-forest-fire/random-forest-inference-udp.ipynb](./random-forest-forest-fire/random-forest-inference-udp.ipynb) | Sentinel-2 + SAR | UDP creation, model reuse | Wrap the trained model as a shareable UDP for scalable inference. |

## Typical workflow

1. build or select a feature cube from EO data
2. train a model, or load a pretrained one
3. run inference with a UDF, ONNX session, or UDP
4. deploy the model for reuse across multiple areas or time periods

Additional background is available in:

- [random-forest-forest-fire/README.md](./random-forest-forest-fire/README.md)
- [dynamic-land-cover-mapping/README.md](./dynamic-land-cover-mapping/README.md)
- [parcel-delineation/README.md](./parcel-delineation/README.md)

This section is a good fit when you want to move from exploratory processing to actual predictive workflows in openEO.
