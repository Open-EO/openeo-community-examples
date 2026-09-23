# 20 · Data Discovery

This section is about finding and loading ready-made datasets into an openEO datacube. It is the place to start when you want to work with an existing product rather than derive a new one from raw inputs.

If you want to create a custom product by masking, merging, compositing, or deriving new bands, use [30 · openEO Processing](../30_openeo-processing/README.md) instead.

Each subfolder is named after the product it accesses, with the exception of [access-stac-catalogs](./access-stac-catalogs/), which groups workflows that use the general `load_stac` mechanism.

## What this section covers

- discovery of available collections and STAC-backed products
- loading product data into an openEO datacube
- combining external datasets with Sentinel-based workflows
- working with product-specific use cases such as flood mapping, biomass, and land-cover products

## Notebooks

| Notebook | Data source | Key openEO features | Description |
|---|---|---|---|
| [proba-v/proba-v.ipynb](./proba-v/proba-v.ipynb) | PROBA-V NDVI (Terrascope via federation) | `load_collection`, federation | Federated access to PROBA-V NDVI. |
| [modis/modis-data-using-openeo.ipynb](./modis/modis-data-using-openeo.ipynb) | MODIS (CDSE STAC) | catalogue browsing, `load_collection` | Explore MODIS collections available in CDSE. |
| [modis/carbon-dynamics.ipynb](./modis/carbon-dynamics.ipynb) | MODIS NDVI / GPP / LAI | `load_collection`, temporal aggregation | Carbon dynamics time series over Antwerp. |
| [modis/environmental-productivity.ipynb](./modis/environmental-productivity.ipynb) | MODIS NDVI / GPP / LAI | `load_collection`, correlation | Environmental productivity analysis. |
| [clms/clms-layers.ipynb](./clms/clms-layers.ipynb) | CLMS burnt area, soil moisture | `load_collection`, masking | Access CLMS layers for a wildfire case study. |
| [global-flood-monitoring/global-flood-monitoring.ipynb](./global-flood-monitoring/global-flood-monitoring.ipynb) | Copernicus GFM + GHSL | `load_collection`, spatial ops | Load GFM and estimate flood-affected population. |
| [worldcereal/worldcereal.ipynb](./worldcereal/worldcereal.ipynb) | ESA WorldCereal | `merge_cubes`, temporal reduce | Combine maize and winter cereal WorldCereal layers. |
| [access-stac-catalogs/load-stac-item.ipynb](./access-stac-catalogs/load-stac-item.ipynb) | Custom GeoTIFF via STAC | `load_stac` | Build a STAC item for your own file and load it. |
| [access-stac-catalogs/load-biomass-stac.ipynb](./access-stac-catalogs/load-biomass-stac.ipynb) | External biomass STAC | `load_stac`, `merge_cubes` | Combine an external biomass dataset with Sentinel-2. |
| [access-stac-catalogs/load-landsat-stac.ipynb](./access-stac-catalogs/load-landsat-stac.ipynb) | Landsat 8 STAC | `load_stac` | Access Landsat 8 through a STAC catalogue. |

## Typical workflow

1. find a suitable collection or external STAC catalogue
2. load the product into an openEO datacube
3. combine it with other datasets if needed
4. continue with analysis, masking, reduction, or downstream ML tasks

This section is best for users who want to work with existing EO products without building their own processing pipeline from scratch.


## Suggested next steps

A good learning path is:

1. [30 · openEO Processing](../30_openeo-processing/README.md) if you want to build custom processing pipelines, masks, composites, and derived products.
2. [40 · Machine Learning](../40_machine-learning/README.md) if you want to train or run ML models on top of the data.
3. [50 · Thematic Notebooks](../50_thematic-notebooks/README.md) or [60 · Geospatial Embeddings](../60_geospatial-embeddings/README.md) if you want domain-specific workflows or learned feature spaces.
4. [70 · Platform and Large Scale](../70_platform-and-large-scale/README.md) if you want to scale up processing, batch orchestration, and platform-level workflows.
