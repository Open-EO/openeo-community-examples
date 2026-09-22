# 50 · Thematic Notebooks

End-to-end applied use cases. Each notebook stitches data access, processing, and interpretation together to answer a real question (Did it flood? Where did the fire burn? How dry is the soil?).

## Notebooks

| Notebook | Theme | Data source | Description |
|---|---|---|---|
| [air-quality/air-quality.ipynb](./air-quality/air-quality.ipynb) | Air quality | Sentinel-5P | Atmospheric product visualisation from Sentinel-5P. |
| [burnt-mapping/burnt-mapping.ipynb](./burnt-mapping/burnt-mapping.ipynb) | Burnt area | Sentinel-2 | Pre/post-fire NBR difference with `chunk_polygon` + UDF. |
| [flood-mapping/flood-ndwi.ipynb](./flood-mapping/flood-ndwi.ipynb) | Flood | Sentinel-2 | NDWI pre/post change detection (Cologne 2021 flood). |
| [flood-mapping/flood-sar.ipynb](./flood-mapping/flood-sar.ipynb) | Flood | Sentinel-1 | SAR backscatter change to map flooded areas. |
| [forest-fire/forest-fire.ipynb](./forest-fire/forest-fire.ipynb) | Wildfire | Sentinel-2 | Pre/near-real-time/post-fire NBR using `MultiResult`. |
| [heatwave/heatwave-nl.ipynb](./heatwave/heatwave-nl.ipynb) | Heatwave | Sentinel-3 LST | Detect Dutch-plan heatwaves from thermal data. |
| [land-cover-statistics/land-cover-country-lcfm.ipynb](./land-cover-statistics/land-cover-country-lcfm.ipynb) | Land cover | LCFM + Eurostat | Zonal land-cover statistics per region. |
| [landslide-ndvi/landslide-ndvi.ipynb](./landslide-ndvi/landslide-ndvi.ipynb) | Landslide | Sentinel-2 | NDVI difference + thresholding for landslide detection. |
| [drought-nddi/drought-nddi.ipynb](./drought-nddi/drought-nddi.ipynb) | Drought | Sentinel-2 | Normalised Difference Drought Index via awesome-spectral-indices. |
| [oil-spill/oil-spill-mapping.ipynb](./oil-spill/oil-spill-mapping.ipynb) | Oil spill | Sentinel-1 | Detect slicks from SAR backscatter anomalies. |
| [radar-vegetation-index/radar-vegetation-index.ipynb](./radar-vegetation-index/radar-vegetation-index.ipynb) | Vegetation | Sentinel-1 | Radar Vegetation Index from dual-pol SAR. |
| [surface-soil-moisture/surface-soil-moisture.ipynb](./surface-soil-moisture/surface-soil-moisture.ipynb) | Soil moisture | Sentinel-1 | Estimate surface soil moisture from 3-year backscatter. |
