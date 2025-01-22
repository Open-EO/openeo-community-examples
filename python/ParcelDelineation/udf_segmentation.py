import functools
import gc
import sys
from typing import Dict

import random
import xarray as xr
from openeo.udf import inspect
from xarray import DataArray

import numpy as np
from xarray.ufuncs import isnan as ufuncs_isnan


# Add the onnx dependencies to the path
sys.path.insert(0, "onnx_deps")
import onnxruntime as ort

model_names = frozenset([
    "BelgiumCropMap_unet_3BandsGenerator_Network1.onnx",
    "BelgiumCropMap_unet_3BandsGenerator_Network2.onnx",
    "BelgiumCropMap_unet_3BandsGenerator_Network3.onnx",
])

@functools.lru_cache(maxsize=25)
def load_ort_sessions(names):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.

    @param modeldir: Model directory
    @return: Loaded model sessions
    """
    # inspect(message="Loading convolutional neural networks as ONNX runtime sessions ...")
    return [
        ort.InferenceSession(f"onnx_models/{model_name}")
        for model_name in names
    ]

def process_window_onnx_old(ndvi_stack, patch_size=128):
    ## check whether you actually supplied 3 images or more
    nr_valid_bands = ndvi_stack.shape[2]

    ## if the stack doesn't have at least 3 bands, we cannot process this window
    if nr_valid_bands < 3:
        inspect(message="Not enough input data for this window -> skipping!")
        # gc.collect()
        return None
    
    ## we'll do 12 predictions: use 3 networks, and for each randomly take 3 NDVI bands and repeat 4 times
    ort_sessions = load_ort_sessions(model_names)
    number_of_models = len(ort_sessions)
    number_per_model = 4

    prediction = np.zeros((patch_size, patch_size, number_of_models * number_per_model))

    for model_counter in range(number_of_models):
        ## load the current model
        ort_session = ort_sessions[model_counter]

        ## make 4 predictions per model
        for i in range(number_per_model):
            # np.random.seed(i)
            random.seed(i)
            ## define the input data
            # input_data = ndvi_stack[:, :,  np.random.choice(np.arange(nr_valid_bands), size=3, replace=False)].reshape(1, patch_size * patch_size, 3)
            input_data = ndvi_stack[:, :, random.choices(np.arange(nr_valid_bands), k=3)].reshape(1, patch_size * patch_size, 3)
            ort_inputs = {
                ort_session.get_inputs()[0].name: input_data
            }

            ## make the prediction
            ort_outputs = ort_session.run(None, ort_inputs)
            prediction[:, :, i + number_per_model*model_counter] =  np.squeeze(
                ort_outputs[0]\
                    .reshape((patch_size, patch_size))
            )

    ## free up some memory to avoid memory errors
    gc.collect()

    ## final prediction is the median of all predictions per pixel
    final_prediction = np.median(prediction, axis=2)
    return final_prediction


def process_window_onnx(ndvi_stack: DataArray, patch_size=128):
    """Compute predictions.

    Compute predictions using ML models. ML models takes three inputs images and predicts
    one image. Four predictions are made per model using three random images. Three images
    are considered to save computational time. Final result is median of these predictions.

    Parameters
    ----------
    ndvi_stack : DataArray
        ndvi data
    patch_size : Int
        Size of the sample

    """
    # we'll do 12 predictions: use 3 networks, and for each random take 3 NDVI bands and repeat 4 times
    ort_sessions = load_ort_sessions(model_names)    # get models
    predictions_per_model = 4
    no_images = ndvi_stack.t.shape[0]

    # Range of index of images
    _range = range(no_images)
    # List of all predictions
    prediction = []
    for model_index, ort_session in enumerate(ort_sessions):
        ## make 4 predictions per model
        for i in range(predictions_per_model):
            # initialize a predicter array
            random.seed(i)   # without seed we will have random number leading to non-reproducable results.
            _idx = random.choices(_range, k=3)
            # re-shape the input data for ML input 
            input_data = ndvi_stack.isel(t=_idx).data.reshape(1, patch_size * patch_size, 3)
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}

            # Run ML to predict
            ort_outputs = ort_session.run(None, ort_inputs)
            # reshape ort_outputs and append it to prediction list
            prediction.append(ort_outputs[0].reshape((patch_size, patch_size)))

    ## free up some memory to avoid memory errors
    gc.collect()

    # Create a DataArray of all predictions
    all_predictions = xr.DataArray(prediction, dims=["predict", "x", "y"],
                                   coords={"predict": range(len(prediction)),
                                           "x": ndvi_stack.coords["x"],
                                           "y": ndvi_stack.coords["y"]}
                                   )
    ## final prediction is the median of all predictions per pixel
    return all_predictions.median(dim="predict")


def preprocess_datacube(cubearray: DataArray):
    # check if bands is in the dims and select the first index
    if "bands" in cubearray.dims:
        nvdi_stack = cubearray.isel(bands=0)
    else:
        nvdi_stack = cubearray

    # Clamp out of range NDVI values
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack < 0.92, 0.92)
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack > -0.08)       # No data exists id less than -0.08
    nvdi_stack += 0.08

    # Fill the no data with value 0
    nvdi_stack_data = nvdi_stack.fillna(0)

    # Count the amount of invalid data per acquisition
    sum_invalid = nvdi_stack.isnull().sum(dim=['x', 'y'])

    # Select all clear images (without ANY missing values)
    # or select the 3 best ones (contain nans)
    if (sum_invalid.data == 0).sum() > 3:
        good_data = nvdi_stack_data.sel(t = sum_invalid[sum_invalid.data == 0].t)
    else:
        good_data = nvdi_stack_data.sel(t = sum_invalid.sortby(sum_invalid).t[:3])
    return good_data.transpose("x", "y", "t")


def preprocess_datacube_old(cubearray):
    ## xarray nan to np nan (we will pass a numpy array)
    cubearray = cubearray.where(~ufuncs_isnan(cubearray), np.nan)

    ## Transpose to format accepted by model and get the values
    cubearray = cubearray.transpose("x", "bands", "y", "t")
    cubearray = cubearray[:, 0, :, :]
    ndvi_stack = cubearray.transpose("x", "y", "t").values

    ## Clamp out of range NDVI values
    ndvi_stack = np.where(ndvi_stack < 0.92, ndvi_stack, 0.92)
    ndvi_stack = np.where(ndvi_stack >-0.08, ndvi_stack, np.nan)
    ndvi_stack = (ndvi_stack + 0.08)

    ## Create a mask where all valid values are 1 and all nans are 0
    ## This will be used for selecting the best images
    mask_stack = np.ones_like(ndvi_stack, dtype=np.uint8)
    mask_stack = np.where(np.isnan(ndvi_stack), 0, 1)

    ## Fill the NaN values with 0
    ndvi_stack[mask_stack == 0] = 0

    ## Count the amount of invalid data per acquisition and sort accordingly
    sum_invalid = np.sum(mask_stack == 0, axis=(0, 1))

    ## If we have enough clear images (without ANY missing values), we're good to go, 
    ## and we will use all of them (could be more than 3!)
    if len(np.where(sum_invalid == 0)[0]) > 3:
        ndvi_stack = ndvi_stack[:, :, np.where(sum_invalid == 0)[0]]

    ## else we need to add some images that do contain some nan's; 
    ## in this case we will select just the 3 best ones
    else:
        # inspect(f"Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> appending some bad images as well!")
        idxsorted = np.argsort(sum_invalid)
        ndvi_stack = ndvi_stack[:, :, idxsorted[:3]]

    ## return the stack
    return ndvi_stack


def apply_datacube_old(cube: DataArray, context: Dict) -> DataArray:
    ## preprocess the datacube
    ndvi_stack = preprocess_datacube_old(cube)

    ## process the window
    result = process_window_onnx_old(ndvi_stack)

    # ## transform your numpy array predictions into an xarray
    result = result.astype(np.float64)
    result_xarray = xr.DataArray(
        result,
        dims=["x", "y"],
        coords={"x": cube.coords["x"], "y": cube.coords["y"]},
    )

    ## Reintroduce time and bands dimensions
    result_xarray = result_xarray.expand_dims(
        dim={
            "t": [np.datetime64(str(cube.t.dt.year.values[0]) + "-01-01")], 
            "bands": ["prediction"],
        },
    )

    ## Return the resulting xarray
    return result_xarray


def apply_datacube(cube: DataArray, context: Dict) -> DataArray:
    ## preprocess the datacube
    ndvi_stack = preprocess_datacube(cube)

    # check number of images after preprocessing
    # if the stack doesn't have at least 3 bands, we cannot process this window
    nr_valid_bands = ndvi_stack.t.shape[0]
    if nr_valid_bands < 3:
        inspect(message="Not enough input data for this window -> skipping!")
        return None

    ## process the window
    result = process_window_onnx(ndvi_stack)

    ## Reintroduce time and bands dimensions
    result_xarray = result.expand_dims(dim={"t": [(cube.t.dt.year.values[0])], "bands": ["prediction"]})

    ## Return the resulting xarray
    return result_xarray
