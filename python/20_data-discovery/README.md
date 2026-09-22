# 20 · Data Discovery

Notebooks that show **how to find and load a ready-made product** into an openEO datacube. If you want to *derive* a new product from raw inputs, look in [30_openeo-processing](../30_openeo-processing/) instead.

Each subfolder is named after the product it accesses. The one exception is [`access-stac-catalogs/`](./access-stac-catalogs/), which groups notebooks about the general `load_stac` mechanism rather than a specific product.

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
