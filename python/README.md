# Python Client - Examples and Code Snippets

This folder contains examples and code snippets for the openEO Python client.

## Overview

Environments: `Python` (plain Python code), `Jupyter` (e.g. Notebooks)

The `Demonstrates` column summarizes the key openEO functionality used in each community example.


| Title | Environment | Description | Demonstrates |
|-|-|-|-|
| [Getting Started](./1.%20GettingStarted/) | `Jupyter`  | A beginner's sample notebook , featuring a straightforward workflow that illustrates the fundamental steps of accessing and utilizing available collections. | Loading collection from a backend; openEO process `load_collection` |
| [Anomaly_Detection](./RescaleChunks/) | `Jupyter`  | Check the crop growth on your field and compare it with similar fields in the region. | Loading data from **WFS**; openEO process `Anomaly_Detection` |
| [BasicSentinelMerge](./BasicSentinelMerge/) | `Jupyter`   | Merging Sentinel 1 and 2 in a single datacube for further processing. | openEO processes `merge_cubes`, `mask_scl_dilation`, `aggregate_temporal_period`, `array_interpolate_linear`, `sar_backscatter`, `filter_bbox` |
| [BurntMapping](./BurntMapping/)             | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). | openEO processes `run_udf`, `chunk_polygon` with polygon loaded from JSON, `reduce_dimension` |
| [FloodNDWI](./FloodNDWI/)                   | `Jupyter`    | Comparative study between pre and post image for Cologne during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/) | **Adding metadata** to a datacube; openEO processes `datacube_from_process`, `merge_datacube`, `reduce_dimension`|
| [FloodSAR](./FloodSAR/)                     | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). | **Thresholding** using `udf`; openEO processes `divide` |
| [GlobalFloodMonitoring](./GlobalFloodMonitoring/) | `Jupyter`    | Exploring the Global Flood Monitoring (GFM) product over the Pakistan flooding of 2022. [Reference](https://extwiki.eodc.eu/GFM) | Loading `GFM` data; saving data in specific `tile_grids`; plotting with additional data |
| [LoadStac](./LoadStac/)                     | `Jupyter`    | Load an external file in openEO by creating a Stac Item for it. | creating a simple stac item, openEO processes `load_stac` |
| [ParcelDelineation](./ParcelDelineation/)   | `Jupyter` | Delineates parcels with Sentinel-2 data using [ONNX](https://onnx.ai/) models. The example focuses on the inference step, using pre-trained models. It demonstrates data loading and preprocessing, inference, post-processing and finally producing vector data as a result. | Selection of best tiles; Running **ONNX models** using `udf`; postprocessing using **sobel filter** and **Felzenszwalb's algoritm** in `udf`, openEO processes `aggregate_spatial`, `build_child_callback`, `filter_labels`, `apply_neighborhood`, `raster_to_vector`, `filter_spatial`|
| [Publishing a UDP (S1 statistics)](./Sentinel1_Stats/) | `Jupyter`   | Computes various statistics for Sentinel-1 data and publishes it as a user-defined process (UDP) that can be re-used by others across multiple languages/environments. | Creating a `udp` with `ProcessBuilder`; Saving `udp`for public reuse with `save_user_defined_process`; Publishing a service; credit usage; openEO processes `rename_labels`, `apply_dimension`, `datacube_from_process` |
| [RankComposites](./RankComposites/)         | `Jupyter`   | Rank composites: max-NDVI & Best Available Pixel. | openEO processes `apply_neigborhood`, `array_apply`, `filter_bbox`, `mask`, `aggregate_temporal_period` |
| [RescaleChunks](./RescaleChunks/)           | `Jupyter`   | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF. | openEO processes `run_udf`, `chunk_polygon`, `reduce_dimension` |
| [WorldCereal](./WorldCereal/)               | `Jupyter`   | WorldCereal data extraction sample. | openEO processes `merge_cubes`, loading **WorldCereal** data |
| [RVI](./RVI/)               | `Jupyter`   | Calculate Radar Vegetation Index | openEO processes `sar_backscatter`, `spectral_nidices.compute_indices`; **plotting** mean result and timeseries; **Awesome Spectral Indices** |
| [OilSpill](./OilSpill/)               | `Jupyter`   | Oil Spill mapping with Sentinel-1 layer. | openEO processes `sar_backscatter`, `apply`, `apply_kernel`, `rename_labels`, `merge_cubes`; **plotting** binary image |
| [ForestFire](./ForestFire/)               | `Jupyter`   | Wildfire mapping using Sentinel-2 | openEO processes `apply_kernel`,`ndvi` `spectral_nidices.compute_indices`; **plotting** comparative visualisation; **Awesome Spectral Indices** |
| [Heatwave](./Heatwave/)               | `Jupyter`   | Heatwave mapping using LST layer. | openEO processes `mask`, `apply_dimension`, `reduce_dimension`; **plotting** Total number of days |
| [AirQuality](./AirQuality/)               | `Jupyter`   | Explore Sentinel-5P air quality products | openEO processes `merge_cubes`, `aggregate_temporal_period`; **plotting** mean result and timeseries; product's correlation  |
| [StatisticalDataFill](./StatisticalDataFill/)               | `Jupyter`   | Uses Lowess Regression in openEO for Filling Missing Time Series Data | openEO processes `aggregate_temporal_period`, `apply_dimension`; **plotting** mean result and timeseries ; `SENTINEL_5P_L2`|
| [DynamicLandCoverMapping](./DynamicLandCoverMapping/)               | `Jupyter`   | A usecase of using Random Forest from openEO¶ | openEO processes `MlModel`, `compute_and_rescale_indices`, `array_create`, `array_concat`, `array_interpolate_linear`; **plotting** predicted land cover mapping ; `SENTINEL2_L2A`, `SENTINEL1_GRD`|


## Contributing

* Please provide each contribution in a separate folder.
