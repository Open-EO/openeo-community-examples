# 30 · openEO Processing

Notebooks that show **how to build a datacube pipeline in openEO**: masking, merging, compositing, terrain, gap-filling, deriving new products, and wrapping reusable logic as a UDP.

If you just want to *load* an existing product, look in [20_data-discovery](../20_data-discovery/) instead.

## Notebooks

| Notebook | Data source | Key openEO features | Description |
|---|---|---|---|
| [basic-sentinel-merge/sentinel-merge.ipynb](./basic-sentinel-merge/sentinel-merge.ipynb) | Sentinel-1 + Sentinel-2 | `merge_cubes`, temporal interpolation | Merge S1 and S2 time series into one datacube. |
| [hillshade/hillshade.ipynb](./hillshade/hillshade.ipynb) | Copernicus 30 m DEM | `apply`, `slope`, `aspect`, trigonometry | Compute a Lambert-model hillshade from a DEM. |
| [rank-composites/bap-composite.ipynb](./rank-composites/bap-composite.ipynb) | Sentinel-2 L2A | scoring, masking, temporal aggregation | Best-Available-Pixel monthly composite. |
| [rank-composites/rank-composites.ipynb](./rank-composites/rank-composites.ipynb) | Sentinel-2 L2A | `aggregate_temporal`, `apply_neighborhood` | Gap-free max-NDVI rank composite. |
| [scl-dilation-mask/scl-dilation-mask.ipynb](./scl-dilation-mask/scl-dilation-mask.ipynb) | Sentinel-2 SCL | morphological ops, convolution | Conservative cloud/shadow mask via Gaussian dilation. |
| [sentinel1-stats/sentinel1-stats.ipynb](./sentinel1-stats/sentinel1-stats.ipynb) | Sentinel-1 GRD | `apply_dimension`, UDP publishing | Aggregate SAR statistics and publish them as a UDP. |
| [statistical-data-fill/statistical-data-fill.ipynb](./statistical-data-fill/statistical-data-fill.ipynb) | Sentinel-2 L2A | UDF (Lowess), cloud masking | Fill time-series gaps with Lowess smoothing in a UDF. |
| [biopar/biopar-service.ipynb](./biopar/biopar-service.ipynb) | Sentinel-2 L2A | UDP execution | Derive biophysical parameters (LAI, FAPAR, FCOVER, …) via the BioPAR UDP. |
