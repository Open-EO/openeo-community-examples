# Random Forest Model for Forest Fire Mapping

This folder contains a Random Forest model implementation for mapping forest fires using Sentinel-1 and Sentinel-2 data. This usecase can be useful for users interested in:

- Data preprocessing and feature extraction from Sentinel-1 and Sentinel-2 datasets.
- Training a model using openEO's Random Forest machine learning algorithm for forest fire mapping.
- Defining custom functions into the workflow as openEO's User-Defined-Functions (UDF).
- Saving the workflow as a User-Defined-Process (UDP) for future use.

The files/folders in this `RandomForest-ForestFire` include:
- [`Dataset/`](././Dataset/): Contains the training and validation datasets. It also includes the geojson files of the training and inference areas.
- [`RandomForestModelTraining.ipynb`](RandomForestModelTraining.ipynb): A Jupyter notebook that demonstrates the entire workflow, including data preprocessing, feature extraction, model training and evaluating the performance of the trained Random Forest model.
- [`eo_extractor.py`](eo_extractor.py): Contains functions for extracting features from Sentinel-1 and Sentinel-2 data.
- [`helper_functions.py`](helper_functions.py): Provides utility functions for data processing and model training.
- [`feature_udf_ndfi_glcm.py`](feature_udf_ndfi_glcm.py): Implements feature extraction methods using NDFI and GLCM.
- [`fire_using_RF.json`](fire_using_RF.json): The openEO process graph for the Random Forest model inference workflow.
- [`RandomForestModelInference_AsUDP.ipynb`](RandomForestModelInference_AsUDP.ipynb): A Jupyter notebook that demonstrates how to save the inference workflow as a User-Defined-Process (UDP) for future use.