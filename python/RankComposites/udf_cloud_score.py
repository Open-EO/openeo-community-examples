import numpy as np
import time, math
import xarray as xr
import datetime

from scipy.ndimage import distance_transform_cdt
from skimage.morphology import footprints
from skimage.morphology import binary_erosion, binary_dilation
from openeo.udf import XarrayDataCube
from xarray.ufuncs import isnan as ufuncs_isnan



def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:

    cube_array: xr.DataArray = cube.get_array()
    cube_array = cube_array.transpose('t', 'bands', 'y', 'x')

    clouds = np.logical_or(np.logical_and(cube_array < 11, cube_array >= 8), cube_array == 3).isel(bands=0)

    weights = [1, 0.8, 0.5]

    # Calculate the Distance To Cloud score
    # Erode
    # Source: https://github.com/dzanaga/satio-pc/blob/e5fc46c0c14bba77e01dca409cf431e7ef22c077/src/satio_pc/preprocessing/clouds.py#L127
    e = footprints.disk(3)
    # Define a function to apply binary erosion
    def erode(image, selem):
        return ~binary_erosion(image, selem)

    # Use apply_ufunc to apply the erosion operation
    eroded = xr.apply_ufunc(
        erode,  # function to apply
        clouds,  # input DataArray
        input_core_dims=[['y', 'x']],  # dimensions over which to apply function
        output_core_dims=[['y', 'x']],  # dimensions of the output
        vectorize=True,  # vectorize the function over non-core dimensions
        dask="parallelized",  # enable dask parallelization
        output_dtypes=[np.int32],  # data type of the output
        kwargs={'selem': e}  # additional keyword arguments to pass to erode
    )

    # Distance to cloud = dilation
    d_min = 0
    d_req = 50
    d = xr.apply_ufunc(
        distance_transform_cdt,
        eroded,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32]
    )
    d = xr.where(d == -1, d_req, d)
    score_clouds = 1 / (1 + np.exp(-0.2 * (np.minimum(d, d_req) - (d_req - d_min) / 2)))

    # Calculate the Coverage score
    score_cov = 1 - clouds.sum(dim='x').sum(dim='y') / (
            cube_array.sizes['x'] * cube_array.sizes['y'])
    score_cov = np.broadcast_to(score_cov.values[:, np.newaxis, np.newaxis],
                                [cube_array.sizes['t'], cube_array.sizes['y'], cube_array.sizes['x']])

    # Final score is weighted average
    score = (weights[0] * score_clouds + weights[2] * score_cov) / sum(weights)
    score = np.where(cube_array.values[:,0,:,:]==0, 0, score)

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