# User-defined Processes

This folder contains user-defined processes that can be run by any openEO client (e.g. Web Editor, Python or R).

## Overview

| ID | Categories | Summary |
| -- | ---------- | ------- |
| [array_contains_nodata](array_contains_nodata.json) | arrays | Check for no-data values in an array |
| [array_find_nodata](array_find_nodata.json)         | arrays | Find no-data values in an array |

## Contributing

* Please provide one JSON file per user-defined process directly into this folder (no sub-folders).
* All processes must be parametrized (e.g. no hard-coded collection names).
* All processes must have appropriate metadata included and follow common best practices. These are enforces by a CI tool that checks the processes for compliance.
  Run `npm test` to check your contributon for compliance.