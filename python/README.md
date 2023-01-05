# Python Client - Examples and Code Snippets

This folder contains process graphs and code snippets for the openEO Python client.

## Overview

Environments: `Python` (plain Python code), `Jupyter` (e.g. Notebooks)

**Furthermore in usecase_notebooks repository:**


| Title | Environment | Description |
| ----- | ----------- | ----------- |
| biomass_basic   | `Jupyter`    | Executes simple process like biomass already available in EOplaza. Service available [here](https://portal.terrascope.be/catalogue/app-details/17) |
| burntmapping_chunks   | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). |
| flood_ndwi   | `Jupyter`    | Comparative study between pre and post image for Colong during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/) |
| flood_sar_udf   | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping) using openeo process.|
| ndwi_basic   | `Jupyter`    | Executes simple process like ndwi already available in EOplaza. Service available [here](https://portal.terrascope.be/catalogue/app-details/13). |
| rescale_chunks   | `Jupyter`    | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF. |



## Contributing

* OpenEO/EOplaza users can also send a pull request in this repository as either a python file/notebook for their usecase. Showcasing their usecases are highly encouraged.