# User-defined Processes

This folder contains user-defined processes that can be run by any openEO client (e.g. Web Editor, Python or R).

## Overview

| ID                                                  | Categories           | Summary                                                     |
| --------------------------------------------------- | -------------------- | ----------------------------------------------------------- |
| [anomaly_detection](anomaly_detection.json)         | cubes                | Regional Benchmarking using CropSAR                         |
| [array_contains_nodata](array_contains_nodata.json) | arrays               | Check for no-data values in an array                        |
| [array_find_nodata](array_find_nodata.json)         | arrays               | Find no-data values in an array                             |
| [burntmapping_chunks](burntmapping_chunks.json)     | math > indices       | Burnt area mapping                                          |
| [crop_polygon](crop_polygon.json)                   | cubes, filter, masks | Crop (filter and mask) using a polygon                      |
| [flood_ndwi](flood_ndwi.json)                       | vegetation indices   | Comparing pre and post flood NDWI                           |
| [flood_sar_udf](flood_sar_udf.json)                 | udf                  | Flood extent visualization applying threshold to SAR images |
| [rescale_chunks](rescale_chunks.json)               | cubes                | Rescaling of RGB within chunk of polygons                   |
| [rvi](rvi.json)                                     | math > indices       | Calculate radar vegetation index using Sentinel-1           |

## Contributing

* Please provide one JSON file per user-defined process directly into this folder (no sub-folders).
* All processes must be parametrized (e.g. no hard-coded collection names).
* All processes must have appropriate metadata included and follow common best practices. These are enforced by a CI tool that checks the processes for compliance.
  After installing nodejs, you can run `npm install` (once to install dependencies) and `npm test` to check your contributon for compliance.