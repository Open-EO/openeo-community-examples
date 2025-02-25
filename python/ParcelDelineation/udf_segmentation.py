import functools
import gc
import sys
from typing import Dict
import random
import xarray as xr
from openeo.udf import inspect

# Add the onnx dependencies to the path
sys.path.insert(1, "onnx_deps")
import onnxruntime as ort


model_names = frozenset([
    "BelgiumCropMap_unet_3BandsGenerator_Network1.onnx",
    "BelgiumCropMap_unet_3BandsGenerator_Network2.onnx",
    "BelgiumCropMap_unet_3BandsGenerator_Network3.onnx",
])


@functools.lru_cache(maxsize=1)
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


def process_window_onnx(ndvi_stack: xr.DataArray, patch_size=128):
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
    # we'll do 12 predictions: use 3 networks, and for each random take 3 NDVI images and repeat 4 times
    ort_sessions = load_ort_sessions(model_names)    # get models

    predictions_per_model = 4
    no_rand_images = 3          # Number of random images that are needed for input
    no_images = ndvi_stack.t.shape[0]
    
    # Range of index of images
    _range = range(no_images)
    # List of all predictions
    prediction = []
    for ort_session in ort_sessions:
        # make 4 predictions per model
        for i in range(predictions_per_model):
            # initialize a predicter array
            random.seed(i)   # without seed we will have random number leading to non-reproducible results.
            _idx = random.choices(_range, k=no_rand_images) # Random selection of 3 images for input
            # re-shape the input data for ML input 
            input_data = ndvi_stack.isel(t=_idx).data.reshape(1, patch_size * patch_size, no_rand_images)
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}

            # Run ML to predict
            ort_outputs = ort_session.run(None, ort_inputs)
            # reshape ort_outputs and append it to prediction list
            prediction.append(ort_outputs[0].reshape((patch_size, patch_size)))

    # free up some memory to avoid memory errors
    gc.collect()

    # Create a DataArray of all predictions
    all_predictions = xr.DataArray(prediction, dims=["predict", "x", "y"],
                                   coords={"predict": range(len(prediction)),
                                           "x": ndvi_stack.coords["x"],
                                           "y": ndvi_stack.coords["y"]}
                                   )
    # final prediction is the median of all predictions per pixel
    return all_predictions.median(dim="predict")


def preprocess_datacube(cubearray: xr.DataArray, min_images: int):
    # check if bands is in the dims and select the first index
    if "bands" in cubearray.dims:
        nvdi_stack = cubearray.isel(bands=0)
    else:
        nvdi_stack = cubearray

    # Clamp out of range NDVI values
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack < 0.92, 0.92)
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack > -0.08)       # No data exists id less than -0.08
    nvdi_stack += 0.08
    
    # Count the amount of invalid data per acquisition
    sum_invalid = nvdi_stack.isnull().sum(dim=['x', 'y'])
    # Check % of invalid data per acquisition by using mean
    sum_invalid_mean = nvdi_stack.isnull().mean(dim=['x', 'y'])
    
    # Fill the no data with value 0
    nvdi_stack_data = nvdi_stack.fillna(0)

    # Check if valid number of acquisitions exist for machine learning
    # Should not contain all nan values
    # If not enough acquisitions return invalid data as True and an xarray of nan values
    if (sum_invalid_mean.data < 1).sum() <= min_images:
        inspect(message="Input data is invalid for this window -> skipping!")
        # create a nan dataset and return. This can be further returned by apply_datacube
        nan_data = xr.zeros_like(nvdi_stack.sel(t = sum_invalid_mean.t[0], drop=True))
        nan_data = nan_data.where(lambda nan_data: nan_data > 1)
        return True, nan_data   # invalid_data flag and data

    # Good data as machine learning is possible.
    # If possible select all clear images (without ANY missing values)
    # Else select the 4 best ones among the missing values
    if (sum_invalid.data == 0).sum() >= min_images:
        good_data = nvdi_stack_data.sel(t = sum_invalid[sum_invalid.data == 0].t)
    else:
        good_data = nvdi_stack_data.sel(t = sum_invalid.sortby(sum_invalid).t[:min_images])
    return False, good_data.transpose("x", "y", "t")         # invalid_data flag and data


def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    # select atleast best 4 temporal images of ndvi for ML
    min_images = 4
    
    # preprocess the datacube
    invalid_data, ndvi_stack = preprocess_datacube(cube, min_images)

    # if data is invalid
    if invalid_data:
        return ndvi_stack.expand_dims(dim={"t": [(cube.t.dt.year.values[0])], "bands": ["prediction"]})
    
    # process the window
    result = process_window_onnx(ndvi_stack)
    # Reintroduce time and bands dimensions
    result_xarray = result.expand_dims(dim={"t": [(cube.t.dt.year.values[0])], "bands": ["prediction"]})
    # Return the resulting xarray
    return result_xarray
