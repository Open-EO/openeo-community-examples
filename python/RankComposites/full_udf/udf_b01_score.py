import numpy as np
import time, math
import xarray as xr



from openeo.udf import XarrayDataCube
from xarray.ufuncs import isnan as ufuncs_isnan



def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:

    cube_array: xr.DataArray = cube.get_array()
    cube_array = cube_array.transpose('t', 'bands', 'y', 'x')

    weight = -1.5
    
    # Divide by 10000 and replace all zero values by NaN
    nan_cube_array = np.where(cube_array.sel(bands='B01') == 0, np.nan, cube_array.sel(bands='B01')/10000)
    # Take the average over the spatial dimensions
    average_b01 = np.nanmean(nan_cube_array, axis=(1, 2))
    
    # Convert the average B01 to a score. Cap and floor this score. Apply the weight.
    score = weight * ((average_b01 - 0.06) / 0.02).clip(min=-0.5, max=1)
    # score_hot_mean = np.where(np.isnan(score_hot_mean), 0, score_hot_mean)

    score = np.broadcast_to(score[:, np.newaxis, np.newaxis],
                                [cube_array.sizes['t'], cube_array.sizes['y'], cube_array.sizes['x']])

    score_da = xr.DataArray(
        score,
        coords={
            't': cube_array.coords['t'],
            'y': cube_array.coords['y'],
            'x': cube_array.coords['x'],
        },
        dims=['t', 'y', 'x']
    )

    score_da = score_da.expand_dims(
        dim={
            "bands": cube_array.coords["bands"],
        },
    )

    score_da = score_da.transpose('t', 'bands', 'y', 'x')

    return XarrayDataCube(score_da)