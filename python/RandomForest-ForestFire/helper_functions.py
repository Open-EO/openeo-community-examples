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


