import functools
import gc
import sys
from typing import Dict

import numpy as np
import xarray as xr
from openeo.udf import inspect
from xarray.ufuncs import isnan as ufuncs_isnan
from xarray import DataArray

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

def process_window_onnx(ndvi_stack, patch_size=128):
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
            ## define the input data
            input_data = ndvi_stack[
                                :,
                                :,
                                np.random.choice(
                                    np.arange(nr_valid_bands), size=3, replace=False
                                ),
                            ].reshape(1, patch_size * patch_size, 3)
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

def preprocess_datacube(cubearray):
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

def apply_datacube(cube: DataArray, context: Dict) -> DataArray:
    ## preprocess the datacube
    ndvi_stack = preprocess_datacube(cube)

    ## process the window
    result = process_window_onnx(ndvi_stack)

    ## transform your numpy array predictions into an xarray
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
