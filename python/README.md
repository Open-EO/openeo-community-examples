# Python Client - Examples and Code Snippets

This folder contains examples and code snippets for the openEO Python client.

## Overview

Environments: `Python` (plain Python code), `Jupyter` (e.g. Notebooks)


| Title | Environment | Description                                                                                                                                                                                                                                                                                                                           |
| ----- | ----------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Anomaly_Detection](./RescaleChunks/)   | `Jupyter`  | Check the crop growth on your field and compare it with similar fields in the region.                                                                                                                                                                                                                                                 |
| [BurntMapping](./BurntMapping/)   | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). |
| [FloodNDWI](./FloodNDWI/)   | `Jupyter`    | Comparative study between pre and post image for Cologne during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/)                                                                                                                                       |
| [FloodSAR](./FloodSAR/)   | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping).                                                                                                                                        |
| [RescaleChunks](./RescaleChunks/)   | `Jupyter`   | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF.                                                                                                                                                                                        |
| [BasicSentinelMerge](./BasicSentinelMerge/) | `Jupyter`   | Merging Sentinel 1 and 2 in a single datacube for further processing.                                                                                                                                                                                                                                                                 |



## Contributing

* Please provide each contribution in a separate folder.