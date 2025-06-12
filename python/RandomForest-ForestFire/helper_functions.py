import json
import folium
from typing import Any
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import geopandas as gpd
import numpy as np

import openeo
from openeo import Connection
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import array_create

# read geojson file as a dict
def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def view_aoi(aoi: dict[str, Any]) -> folium.Map:
    """
    Create a Folium map from a GeoJSON aoi.
    Args:
        aoi (dict): A GeoJSON FeatureCollection.
    Returns:
        folium.Map: A Folium map centered on the bounding box of the first feature.
    """
    first_feature = aoi["features"][0]
    coords = first_feature["geometry"]["coordinates"][0]
    
    lats = [pt[1] for pt in coords]
    lons = [pt[0] for pt in coords]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

    m = folium.Map(zoom_start=10)
    folium.GeoJson(aoi).add_to(m)
    m.fit_bounds(bounds)

    return m

def plot_pred(pred,aoi):
    # Load GeoJSON and reproject to match the raster
    aoi = gpd.read_file(aoi)
    
    with rasterio.open(pred) as src:
        aoi = aoi.to_crs(src.crs)  # Reproject GeoJSON to raster's CRS
        # Clip raster using geometry
        out_image, out_transform = mask(src, aoi.geometry, crop=True)
        bounds = transform_bounds(src.crs,'EPSG:4326',*src.bounds)

    m = folium.Map([(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2], zoom_start=10)

    folium.TileLayer(
          tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attr = 'Esri',
          name = 'Esri Satellite',
          overlay = False,
          control = True
          ).add_to(m)
    
    
    folium.raster_layers.ImageOverlay(
        image=np.nan_to_num(out_image[0]),
        bounds=[[bounds[1]-0.0002,bounds[0]],[bounds[3],bounds[2]]],
        # origin="lower",
        colormap=lambda x: (1, 0, 0, x),
        opacity=0.5).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

    

def s1_features(connection: Connection, date, aoi, reducer):

    """
    Preprocess Sentinel-1 SAR data by applying backscatter correction,
    computing VH/VV ratio and log transformations, then reducing over time.

    Args:
        connection (Connection): An openEO connection.
        date (Tuple[str, str]): Temporal extent as (start_date, end_date)
        aoi (dict): Spatial extent 
        reducer (Any): Reducer (first, last, mean, median)

    Returns:
        DataCube: The processed and temporally reduced Sentinel-1 data cube.
    """
    # load S2 pre-collection
    s1_cube = connection.load_collection(
        "SENTINEL1_GRD",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["VH", "VV"]
    )
    
    # apply SAR backscatter processing to the collection
    s1_cube = s1_cube.sar_backscatter(coefficient="sigma0-ellipsoid")

    # apply band-wise processing to create a ratio and log-transformed bands
    s1_cube = s1_cube.apply_dimension(
        dimension="bands",
        process=lambda x: array_create(
            [
                30.0 * x[0] / x[1],  # Ratio of VH to VV
                30.0 + 10.0 * x[0].log(base=10),  # Log-transformed VH
                30.0 + 10.0 * x[1].log(base=10),  # Log-transformed VV
            ]
        ),
    )

    s1_cube = s1_cube.rename_labels("bands", ["ratio"] + s1_cube.metadata.band_names)

    # scale to int16
    s1_cube = s1_cube.linear_scale_range(0, 30, 0, 30000)

    return s1_cube.reduce_dimension(reducer=reducer, dimension="t")

def s2_features(connection: Connection, date, aoi, reducer):
    """
   Preprocess Sentinel-2 data by loading relevant bands, applying scaling,
   and reducing over time using a specified reducer.

    Args:
        connection (Connection): An openEO connection.
        date: Temporal extent as (start_date, end_date)
        aoi: Spatial extent 
        reducer (Any): Reducer (first, last, mean, median)

    Returns:
        DataCube: The processed and temporally reduced Sentinel-2 datacube.
    """
    # load S2 pre-collection
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["B02", "B03", "B04", "B08","B12"],
        max_cloud_cover=80,
    )
    
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["SCL"],
        max_cloud_cover=80,
    )
    
    mask = scl.process(
        "to_scl_dilation_mask", 
        data=scl,
        kernel1_size=17,
        kernel2_size=77,
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3
    )
    
    # Create a cloud-free mosaic
    masked_cube = s2_cube.mask(mask)
    cf_cube = masked_cube.reduce_dimension(reducer=reducer, dimension="t")

    # calculate all indices
    indices_list = ["NBR", "BAI"]
    indices = compute_indices(cf_cube, indices_list)

    # calculate texture features
    features_udf = openeo.UDF.from_file("features.py")
    features = cf_cube.apply_neighborhood(
        process=features_udf,
        size=[
            {"dimension": "x", "value": 128, "unit": "px"},
            {"dimension": "y", "value": 128, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 32, "unit": "px"},
            {"dimension": "y", "value": 32, "unit": "px"},
        ],
    )
    
    # combine the original bands with the computed indices,
    merged_cube = cf_cube.merge_cubes(indices)
    merged_cube = merged_cube.merge_cubes(features)
    return merged_cube


