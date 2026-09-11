# Python Client Examples

This folder contains community notebooks and code snippets for the openEO Python client, grouped by intent.

## Classification Rule

- Data Discovery: notebooks mainly about finding, loading, and inspecting data (for example load_collection/load_stac and product exploration).
- OpenEO Processing: notebooks mainly about transforming datacubes in openEO (for example mask, merge, reduce, UDFs, compositing, or multi-step process graphs).
- If a notebook starts by loading data but primarily builds a processing graph, it belongs to OpenEO Processing.

## Folder Taxonomy

### 00_Getting_Started
- [Getting Started](./00_Getting_Started/1.%20GettingStarted/GettingStarted.ipynb)

### 10_Data_Discovery
- [Access PROBA-V](./10_Data_Discovery/AccessPROBA-V/PROBA_V.ipynb)
- [Access CLMS product using openEO](./10_Data_Discovery/Access_CLMS/CLMS_layers_using_openEO.ipynb)
- [Air Quality](./10_Data_Discovery/AirQuality/AirQuality.ipynb)
- [BioPAR openEO Service](./10_Data_Discovery/BioPAR/biopar_service.ipynb)
- [Global Flood Monitoring](./10_Data_Discovery/GlobalFloodMonitoring/Global_Flood_Monitoring.ipynb)
- [Load STAC item example](./10_Data_Discovery/LoadStac/load-stac-item-example.ipynb)
- [Load Biomass STAC](./10_Data_Discovery/LoadStac/LoadBiomassSTAC.ipynb)
- [Load Landsat STAC](./10_Data_Discovery/LoadStac/LoadLandsatSTAC.ipynb)
- [Access MODIS data in CDSE](./10_Data_Discovery/MODIS/MODIS_data_using_openEO.ipynb)
- [Analyse Carbon Dynamic](./10_Data_Discovery/MODIS/CarbonProduces.ipynb)
- [Analyse Environmental Productivity](./10_Data_Discovery/MODIS/Environmental_productivity.ipynb)
- [WorldCereal](./10_Data_Discovery/WorldCereal/WorldCereal.ipynb)

### 20_OpenEO_Processing
- [Basic Sentinel Merge](./20_OpenEO_Processing/BasicSentinelMerge/sentinel_merge.ipynb)
- [Hillshade](./20_OpenEO_Processing/Hillshade/Hillshade.ipynb)
- [Rank Composites](./20_OpenEO_Processing/RankComposites/rank_composites.ipynb)
- [BAP Composite](./20_OpenEO_Processing/RankComposites/bap_composite.ipynb)
- [SCL Dilation Mask](./20_OpenEO_Processing/SCLDilationMask/to_scl_dilation_mask.ipynb)
- [Publishing a UDP (Sentinel-1 stats)](./20_OpenEO_Processing/Sentinel1_Stats/Sentinel1_Stats.ipynb)
- [Statistical Data Fill](./20_OpenEO_Processing/StatisticalDataFill/StatisticalDataFill.ipynb)

### 30_Machine_Learning
- [CORSA processes](./30_Machine_Learning/Corsa/CORSA%20processes.ipynb)
- [Dimensionality Reduction](./30_Machine_Learning/DimensionalityReduction/Dimensionality%20Reduction.ipynb)
- [Dynamic Land Cover Mapping](./30_Machine_Learning/DynamicLandCoverMapping/Dynamic%20land%20cover%20mapping.ipynb)
- [ONNX ML Inference](./30_Machine_Learning/OnnxMLInference/Onnx_ML_Inference.ipynb)
- [Parcel Delineation](./30_Machine_Learning/ParcelDelineation/Parcel%20delineation.ipynb)
- [Forest Fire Mapping Using Random Forest](./30_Machine_Learning/RandomForest-ForestFire/RandomForestModelTraining.ipynb)
- [Random Forest Inference as UDP](./30_Machine_Learning/RandomForest-ForestFire/RandomForestModelInference_AsUDP.ipynb)
- [Tessera Embedding](./30_Machine_Learning/TesseraEmbedding/TesseraEmbedding.ipynb)

### 40_Thematic_Notebooks
- [Anomaly Detection](./40_Thematic_Notebooks/Anomaly_Detection/Anomaly_Detection.ipynb)
- [Burnt Mapping](./40_Thematic_Notebooks/BurntMapping/burntmapping.ipynb)
- [Flood Detection with NDWI](./40_Thematic_Notebooks/FloodNDWI/flood_ndwi.ipynb)
- [Flood Mapping using SAR](./40_Thematic_Notebooks/FloodNDWI/flood_SAR.ipynb)
- [Forest Fire Analysis](./40_Thematic_Notebooks/ForestFire/ForestFire.ipynb)
- [Heatwave](./40_Thematic_Notebooks/Heatwave/HeatwaveNL.ipynb)
- [Land Cover Statistics](./40_Thematic_Notebooks/LandCoverStatistics/land_cover_country_lcfm.ipynb)
- [Landslide NDVI](./40_Thematic_Notebooks/LandslideNDVI/LandslidesNDVI.ipynb)
- [NDDI](./40_Thematic_Notebooks/NDDI/NDDI_Drought.ipynb)
- [Oil Spill Mapping](./40_Thematic_Notebooks/OilSpill/OilSpillMapping.ipynb)
- [RVI](./40_Thematic_Notebooks/RVI/RVI.ipynb)
- [Surface Soil Moisture](./40_Thematic_Notebooks/SurfaceSoilMoisture/SoilMoisture.ipynb)

### 50_Platform_And_Large_Scale
- [Federated Processing](./50_Platform_And_Large_Scale/Federation/FederatedProcessing.ipynb)
- [Managing Multiple Large-Scale Jobs](./50_Platform_And_Large_Scale/ManagingMultipleLargeScaleJobs/ManagingMultipleLargeScaleJobs.ipynb)
- [Visualising Multiple openEO Jobs](./50_Platform_And_Large_Scale/ManagingMultipleLargeScaleJobs/VisualisingMultipleOpeneoJobs.ipynb)

## Contributing

- Please provide each contribution in a separate folder.
