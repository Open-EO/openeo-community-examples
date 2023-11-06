import openeo
import shapely
from ipyleaflet import (
    Map,
    Marker,
    TileLayer, ImageOverlay,
    Polygon, Rectangle,
    GeoJSON,
    DrawControl,
    LayersControl,
    WidgetControl,
    basemaps,
    basemap_to_tiles,
    WMSLayer,
    FullScreenControl
)

class openeoMap:
    def __init__(self,center,zoom):
        self.map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, interpolation='nearest')
        self.bbox = []
        self.point_coords = []
        self.figure = None
        self.figure_widget = None
        feature_collection = {
            'type': 'FeatureCollection',
            'features': []
        }

        draw = DrawControl(
            circlemarker={}, polyline={}, polygon={},
            marker= {"shapeOptions": {
                       "original": {},
                       "editing": {},
            }},
            rectangle = {"shapeOptions": {
                       "original": {},
                       "editing": {},
            }})

        self.map.add_control(draw)
        def handle_draw(target, action, geo_json):
            feature_collection['features'] = []
            feature_collection['features'].append(geo_json)
            if feature_collection['features'][0]['geometry']['type'] == 'Point':
                self.point_coords = feature_collection['features'][0]['geometry']['coordinates']
            else:
                coords = feature_collection['features'][0]['geometry']['coordinates'][0]
                polygon = shapely.geometry.Polygon(coords)
                self.bbox = polygon.bounds
        
        layers_control = LayersControl(position='topright')
        self.map.add_control(layers_control)
        self.map.add_control(FullScreenControl())
        self.map.add_layer(basemap_to_tiles(basemaps.Esri.WorldImagery));
        draw.on_draw(handle_draw)
    
    def getBbox(self):
        if(len(self.bbox) == 0):
            mapBox = self.map.bounds     
            return [ mapBox[0][1],mapBox[0][0],mapBox[1][1],mapBox[1][0]]
        else:
            return self.bbox
        