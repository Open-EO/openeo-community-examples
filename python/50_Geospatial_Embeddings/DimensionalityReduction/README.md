# Dimensionality Reduction (UDF PCA)

This directory demonstrates how to perform **dimensionality reduction** using **Principal Component Analysis (PCA)** within the [OpenEO](https://openeo.org) framework.  
The example implements PCA as a **User-Defined Function (UDF)** in Python, showing how custom transformations can be integrated into OpenEO process graphs.

---

## Directory Structure

DimensionalityReduction/

### **├── udf_apply_pca.py**

This User-Defined Function (UDF) applies a **dimensionality reduction model** (such as PCA or another ONNX-compatible technique) to an input **xarray-based data cube** within an openEO processing workflow. It reduces the spectral dimensionality of multi-band imagery (e.g., satellite data) into a smaller set of **components**, while preserving spatial alignment (`x`, `y` coordinates). The UDF is fully compatible with **ONNX Runtime**, allowing the same trained model to be reused efficiently in different openEO environments.

1. **Locate the ONNX model file**  
   Searches for the specified ONNX model (e.g., `my_model.onnx`) in standard openEO working directories such as  
   `/opt/**/work-dir/onnx_models/`. This can be a scikit-learn model converted to ONNX, e.g. with `skl2onnx` package.

2. **Load and initialize the model**  
   Uses `onnxruntime` to create a runtime session for the model.  
   This enables efficient and hardware-independent inference.

3. **Extract model metadata**  
   Reads custom ONNX metadata entries (`input_features`, `output_features`)  
   to determine which bands to expect and how many components to produce.

4. **Prepare and preprocess the input cube**  
   Converts the input `xarray.DataArray` into a NumPy array, ensuring it is properly ordered  
   (`y`, `x`, `bands`) and reshaped for model inference.

5. **Run inference for dimensionality reduction**  
   Applies the loaded ONNX model to transform the multi-band input data into a reduced set of component features.

6. **Normalize the model outputs**  
   Scales each resulting component between `[0, 1]` using `MinMaxScaler` for consistent output values.

7. **Reconstruct a spatially aligned output cube**  
   Converts the inference results back into an `xarray.DataArray`, preserving the original `x` and `y` coordinates.  
   Each new band corresponds to a component (e.g., `COMP1`, `COMP2`, …).

8. **Update cube metadata**  
   Renames and filters band labels to match the output components and updates cube metadata accordingly.

9. **Cache for performance**  
   Uses `functools.lru_cache` to store the loaded ONNX model in memory,  
   avoiding repeated loading during multi-tile or multi-chunk processing.
   
### **├── udf_select_sifnificant_bands_by_pca_loadings.py**

This UDF selects the **most significant spectral bands** based on **PCA loadings**.  
It processes an `xarray.DataArray` resulting from a PCA transformation and identifies the bands that contribute most to the first few principal components.  

1. **Locate and load the PCA model**  
   Searches for the specified ONNX model (e.g., `my_model.onnx`) in standard openEO working directories such as  
   `/opt/**/work-dir/onnx_models/`. This can be a scikit-learn model converted to ONNX, e.g. with `skl2onnx` package.

2. **Compute band contributions**  
   - Uses an identity input to probe the PCA model and compute **loadings** for each band.  
   - Calculates **normalized contribution scores** per band to quantify their influence on the PCA components.

3. **Select significant bands**  
   - Either selects bands exceeding a **threshold** or the **top-k most contributing bands**.  
   - Returns **indices, contribution scores, and original band labels** for downstream filtering.

4. **Update cube metadata**  
   - Optionally used in `apply_metadata` to filter cube metadata to match the selected bands.  

### **└── Dimensionality Reduction.ipynb**

A Jupyter Notebook providing a **step-by-step demonstration** of PCA-based dimensionality reduction using the above UDFs.  

1. **Load example satellite data**  
   Loads an Open Datacube with Sentinel-2 data and derives extra bands to create over 100 input features.

2. **Visualize RGB**  
   Downloads RGB data from the Sentinel-2 collection (mean value over spatial extent) to get a view of the area.

3. **Explore the data and model**  
   Uses a pre-trained PCA model to identify the most important features.

4. **Select significant bands via PCA loadings**  
   Applies `udf_select_significant_bands_by_pca_loadings.py` to filter out less informative bands.

5. **Apply PCA dimensionality reduction**  
   Uses `udf_apply_pca.py` to reduce dimensionality while preserving spatial structure.

6. **Visualize resulting components**  
   Plots and analyzes the principal components to understand the captured variance.



### **└── img/*.png**

Contains **example figures** used in the notebook and documentation:

- `AbsoluteSignificantBandLoadingsHeatmap.png` — Heatmap of band contributions to principal components  
- `BandSignificanceCurve.png` — Curve showing cumulative band significance  
- `CumulativeExplainedVariance.png` — Cumulative explained variance by PCA components  
- `ExplainedVarianceRatio.png` — Explained variance ratio per component

