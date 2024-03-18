# Python Client - Examples and Code Snippets

This folder contains examples and code snippets for the openEO Python client.

## Overview

Environments: `Python` (plain Python code), `Jupyter` (e.g. Notebooks)

The `Demonstrates` column summarizes the key openEO functionality used in each community example.


| Title | Environment | Description | Demonstrates |
|-|-|-|-|
| [Anomaly_Detection](./RescaleChunks/) | `Jupyter`  | Check the crop growth on your field and compare it with similar fields in the region. | Loading data from **WFS**; openEO process `Anomaly_Detection` |
| [BasicSentinelMerge](./BasicSentinelMerge/) | `Jupyter`   | Merging Sentinel 1 and 2 in a single datacube for further processing. | openEO processes `merge_cubes`, `mask_scl_dilation`, `aggregate_temporal_period`, `array_interpolate_linear`, `sar_backscatter`, `filter_bbox` |
| [BurntMapping](./BurntMapping/)             | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). | openEO processes `run_udf`, `chunk_polygon` with polygon loaded from JSON, `reduce_dimension` |
| [FloodNDWI](./FloodNDWI/)                   | `Jupyter`    | Comparative study between pre and post image for Cologne during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/) | **Adding metadata** to a datacube; openEO processes `datacube_from_process`, `merge_datacube`, `reduce_dimension`|
| [FloodSAR](./FloodSAR/)                     | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). | **Thresholding** using `udf`; openEO processes `divide` |
| [ParcelDelineation](./ParcelDelineation/)   | `Jupyter` | Delineates parcels with Sentinel-2 data using [ONNX](https://onnx.ai/) models. The example focuses on the inference step, using pre-trained models. It demonstrates data loading and preprocessing, inference, post-processing and finally producing vector data as a result. | Selection of best tiles; Running **ONNX models** using `udf`; postprocessing using **sobel filter** and **Felzenszwalb's algoritm** in `udf`, openEO processes `aggregate_spatial`, `build_child_callback`, `filter_labels`, `apply_neighborhood`, `raster_to_vector`, `filter_spatial`|
| [Publishing a UDP (S1 statistics)](./Sentinel1_Stats/) | `Jupyter`   | Computes various statistics for Sentinel-1 data and publishes it as a user-defined process (UDP) that can be re-used by others across multiple languages/environments. | Creating a `udp` with `ProcessBuilder`; Saving `udp`for public reuse with `save_user_defined_process`; Publishing a service; credit usage; openEO processes `rename_labels`, `apply_dimension`, `datacube_from_process` |
| [RankComposites](./RankComposites/)         | `Jupyter`   | Max NDVI compositing in openEO. | openEO processes `apply_neigborhood`, `array_apply`, `filter_bbox`, `mask`, `aggregate_temporal_period` |
| [RescaleChunks](./RescaleChunks/)           | `Jupyter`   | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF. | openEO processes `run_udf`, `chunk_polygon`, `reduce_dimension` |
| [WorldCereal](./WorldCereal/)               | `Jupyter`   | WorldCereal data extraction sample. | openEO processes `merge_cubes` |

| Title                                       | Environment | Description                                                                                                                                                                                                                                                                                                                           |
|---------------------------------------------| ----------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Anomaly_Detection](./RescaleChunks/)       | `Jupyter`  | Check the crop growth on your field and compare it with similar fields in the region. |
| [BasicSentinelMerge](./BasicSentinelMerge/) | `Jupyter`   | Merging Sentinel 1 and 2 in a single datacube for further processing. |
| [BurntMapping](./BurntMapping/)             | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). |
| [FloodNDWI](./FloodNDWI/)                   | `Jupyter`    | Comparative study between pre and post image for Cologne during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/) |
| [FloodSAR](./FloodSAR/)                     | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). |
| [ParcelDelineation](./ParcelDelineation/)   | `Jupyter` | Delineates parcels with Sentinel-2 data using ONNX models. The example focuses on the inference step, using pre-trained models. It demonstrates data loading and preprocessing, inference, and finally producing vector data as a result. |
| [Publishing a UDP (S1 statistics)](./Sentinel1_Stats/) | `Jupyter`   | Computes various statistics for Sentinel-1 data and publishes it as a user-defined process (UDP) that can be re-used by others across multiple languages/environments. |
| [RankComposites](./RankComposites/)         | `Jupyter`   | Max NDVI compositing in openEO. |
| [RescaleChunks](./RescaleChunks/)           | `Jupyter`   | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF. |
| [WorldCereal](./WorldCereal/)               | `Jupyter`   | WorldCereal data extraction sample. |
| [RVI](./RVI/)               | `Jupyter`   | Calculate Radar Vegetation Index |



## Contributing

* Please provide each contribution in a separate folder.
