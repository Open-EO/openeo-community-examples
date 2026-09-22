# 40 · Machine Learning

Train or apply ML models on Earth observation data through openEO. Feature engineering, training, and inference all run on the backend via UDFs and (optionally) UDPs — no local downloads required.

## Notebooks

| Notebook | Data source | Key openEO features | Description |
|---|---|---|---|
| [dynamic-land-cover-mapping/dynamic-land-cover-mapping.ipynb](./dynamic-land-cover-mapping/dynamic-land-cover-mapping.ipynb) | Sentinel-2 + SAR | UDFs for features, `MlModel` API | Train a Random Forest for dynamic land cover and run inference. |
| [parcel-delineation/parcel-delineation.ipynb](./parcel-delineation/parcel-delineation.ipynb) | Sentinel-2 | UDF, pretrained U-Net | Deploy a U-Net to delineate agricultural parcel boundaries. |
| [onnx-inference/onnx-ml-inference.ipynb](./onnx-inference/onnx-ml-inference.ipynb) | Sentinel-2 | UDF with ONNX runtime | Run a pretrained CNN in ONNX format at datacube scale. |
| [random-forest-forest-fire/random-forest-training.ipynb](./random-forest-forest-fire/random-forest-training.ipynb) | Sentinel-2 + SAR | GLCM UDFs, RF training, model persistence | Full training workflow for a forest-fire Random Forest model. |
| [random-forest-forest-fire/random-forest-inference-udp.ipynb](./random-forest-forest-fire/random-forest-inference-udp.ipynb) | Sentinel-2 + SAR | UDP creation, model reuse | Wrap the trained model as a shareable UDP for scalable inference. |

See [`random-forest-forest-fire/README.md`](./random-forest-forest-fire/README.md), [`dynamic-land-cover-mapping/README.md`](./dynamic-land-cover-mapping/README.md), and [`parcel-delineation/README.md`](./parcel-delineation/README.md) for extra background on those notebooks.
