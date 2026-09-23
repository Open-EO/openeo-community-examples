# Python Client Examples

Community notebooks and code snippets for the [openEO Python client](https://open-eo.github.io/openeo-python-client/), organised by intent so you can jump straight to the kind of workflow you want to learn.

## How the folders are organised

Each numbered folder groups notebooks by what the user is trying to *do*, not by data source. Read the section README for a full notebook-by-notebook breakdown.

| Folder | What lives here |
|---|---|
| [10_getting-started](./10_getting-started/) | Start here. Your first openEO connection, authentication, a datacube, and a download. |
| [20_data-discovery](./20_data-discovery/) | **Access** existing products (MODIS, PROBA-V, CLMS, WorldCereal, Global Flood Monitoring, external STAC catalogs). |
| [30_openeo-processing](./30_openeo-processing/) | **Produce** and transform datacubes: composites, masks, merges, UDFs, UDPs, terrain, gap-filling. |
| [40_machine-learning](./40_machine-learning/) | Train and apply ML models (Random Forest, ONNX, U-Net) inside openEO via UDFs. |
| [50_thematic-notebooks](./50_thematic-notebooks/) | End-to-end use cases (floods, fires, heatwaves, droughts, land cover, soil moisture, …). |
| [60_geospatial-embeddings](./60_geospatial-embeddings/) | Compression and representation learning workflows (CORSA, PCA, TESSERA). |
| [70_platform-and-large-scale](./70_platform-and-large-scale/) | Federation, batch orchestration, and monitoring across backends. |

### Classification rule

- If a notebook mainly shows **how to *get* a ready-made product** into a datacube, it belongs in **`20_data-discovery`**.
- If it mainly shows **how to *build* something new** with openEO processes, it belongs in **`30_openeo-processing`**.
- If it wraps a full **applied use case** (data + processing + interpretation), it belongs in **`50_thematic-notebooks`**.
- ML training/inference workflows belong in **`40_machine-learning`** even if they also touch discovery or processing.


