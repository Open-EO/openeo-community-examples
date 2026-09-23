# 30 · openEO Processing

This section moves beyond loading datasets and focuses on building actual processing pipelines in openEO: masking, merging, compositing, terrain processing, gap-filling, and creating reusable process logic.

If you want to access an existing product rather than derive a new one, start with [20 · Data Discovery](../20_data-discovery/README.md).

## What this section covers

- combining multiple time series into a single datacube
- deriving new products with masks and spatial operators
- temporal compositing and rank-based aggregation
- building reusable logic as a UDP
- applying UDFs for custom spatial or temporal processing

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

## Typical workflow

1. load the relevant collections
2. apply masking or quality filtering
3. merge, reduce, or aggregate over time and space
4. derive a new product or wrap logic as a reusable UDP
5. save or export the result for analysis or downstream modeling

This section is best for users who want to turn raw EO data into richer, custom products inside openEO.

## Suggested next steps

A good learning path is:


3. [40 · Machine Learning](../40_machine-learning/README.md) if you want to train or run ML models on top of the data.
4. [50 · Thematic Notebooks](../50_thematic-notebooks/README.md) or [60 · Geospatial Embeddings](../60_geospatial-embeddings/README.md) if you want domain-specific workflows or learned feature spaces.
5. [70 · Platform and Large Scale](../70_platform-and-large-scale/README.md) if you want to scale up processing, batch orchestration, and platform-level workflows.

The right next step depends on whether you want to go from custom product engineering to ML, feature embedding, or large-scale execution.