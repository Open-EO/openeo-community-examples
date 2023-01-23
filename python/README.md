# Python Client - Examples and Code Snippets

This folder contains code snippets for the openEO Python client.

## Overview

Environments: `Python` (plain Python code), `Jupyter` (e.g. Notebooks)


| Title | Environment | Description |
| ----- | ----------- | ----------- |
| BurntMapping (./BurntMapping/)   | `Jupyter`    | Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping). |
| FloodNDWI (./FloodNDWI/)   | `Jupyter`    | Comparative study between pre and post image for Colong during 2021 flood using NDWI. [Refernce](https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/) |
| FloodSAR (./FloodSAR/)   | `Jupyter`    | Flood extent detection following [UN SPIDER's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping) using openeo process.|
| RescaleChunks (./RescaleChunks/)   | `Jupyter`    | The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon apply with a (User Defined Function) UDF. |



## Contributing

* OpenEO users can send a pull request to this repository as either a Python file or a Jupyter notebook for their usecase. Submissions should be in separate folders. Showcasing their usecases are highly encouraged.