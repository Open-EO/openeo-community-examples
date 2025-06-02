from functools import lru_cache
import gc
import sys
from typing import Dict, Tuple
from random import seed, sample
from xarray import DataArray, zeros_like
from openeo.udf import inspect

# Add the onnx dependencies to the path
sys.path.insert(1, "onnx_deps")
import onnxruntime as ort


model_names = frozenset(
    [
        "BelgiumCropMap_unet_3BandsGenerator_Network1.onnx",
        "BelgiumCropMap_unet_3BandsGenerator_Network2.onnx",
        "BelgiumCropMap_unet_3BandsGenerator_Network3.onnx",
    ]
)


@lru_cache(maxsize=1)
def load_ort_sessions(names):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.

    @param modeldir: Model directory
    @return: Loaded model sessions
    """
    # inspect(message="Loading convolutional neural networks as ONNX runtime sessions ...")
    return [ort.InferenceSession(f"onnx_models/{model_name}") for model_name in names]


def process_window_onnx(ndvi_stack: DataArray, patch_size=128) -> DataArray:
    """Compute prediction.

    Compute predictions using ML models. ML models takes three inputs images and predicts
    one image. Four predictions are made per model using three random images. Three images
    are considered to save computational time. Final result is median of these predictions.

    Parameters
    ----------
    ndvi_stack : DataArray
        ndvi data
    patch_size : Int
        Size of the sample

    Returns
    -------
    xr.DataArray
        Machine learning prediction.
    """
    # Do 12 predictions: use 3 networks, and for each take 3 random NDVI images and repeat 4 times
    ort_sessions = load_ort_sessions(model_names)  # get models

    predictions_per_model = 4
    no_rand_images = 3  # Number of random images that are needed for input
    no_images = ndvi_stack.t.shape[0]

    # Range of index of images for random index selection
    images_range = range(no_images)
    # List of all predictions
    prediction = []
    for ort_session in ort_sessions:
        # make 4 predictions per model
        for i in range(predictions_per_model):
            # initialize a predicter array
            # Seed to lead to a reproducible results.
            seed(i)
            # Random selection of 3 images for input
            idx = sample(images_range, k=no_rand_images)
            # log a message that the selected indices are not at least a week away
            if len(set((ndvi_stack.isel(t=idx)).t.dt.isocalendar().week.data)) != no_rand_images:
                inspect(message="Time difference is not larger than a week for good parcel delineation")

            # re-shape the input data for ML input
            input_data = ndvi_stack.isel(t=idx).data.reshape(1, patch_size * patch_size, no_rand_images)
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}

            # Run ML to predict
            ort_outputs = ort_session.run(None, ort_inputs)
            # reshape ort_outputs and append it to prediction list
            prediction.append(ort_outputs[0].reshape((patch_size, patch_size)))

    # free up some memory to avoid memory errors
    gc.collect()

    # Create a DataArray of all predictions
    all_predictions = DataArray(
        prediction,
        dims=["predict", "x", "y"],
        coords={
            "predict": range(len(prediction)),
            "x": ndvi_stack.coords["x"],
            "y": ndvi_stack.coords["y"],
        },
    )
    # final prediction is the median of all predictions per pixel
    return all_predictions.median(dim="predict")


def get_valid_ml_inputs(nvdi_stack_data: DataArray, sum_invalid, min_images: int) -> DataArray:
    """Machine learning inputs

    Extract ML inputs based on how good the data is

    """
    if (sum_invalid.data == 0).sum() >= min_images:
        good_data = nvdi_stack_data.sel(t=sum_invalid[sum_invalid.data == 0].t)
    else:  # select the 4 best time samples with least amount of invalid pixels.
        good_data = nvdi_stack_data.sel(t=sum_invalid.sortby(sum_invalid).t[:min_images])
    return good_data


def preprocess_datacube(cubearray: DataArray, min_images: int) -> Tuple[bool, DataArray]:
    """Preprocess data for machine learning.

    Preprocess data by clamping NVDI values and first check if the
    data is valid for machine learning and then check if there is good
    data to perform machine learning.

    Parameters
    ----------
    cubearray : xr.DataArray
        Input datacube
    min_images : int
        Minimum number of samples to consider for machine learning.

    Returns
    -------
    bool
        True refers to data is invalid for machine learning.
    xr.DataArray
        If above bool is False, return data for machine learning else returns a
        sample containing nan (similar to machine learning output).
    """
    # Preprocessing data
    # check if bands is in the dims and select the first index
    if "bands" in cubearray.dims:
        nvdi_stack = cubearray.isel(bands=0)
    else:
        nvdi_stack = cubearray
    # Clamp out of range NDVI values
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack < 0.92, 0.92)
    nvdi_stack = nvdi_stack.where(lambda nvdi_stack: nvdi_stack > -0.08)
    nvdi_stack += 0.08
    # Count the amount of invalid pixels in each time sample.
    sum_invalid = nvdi_stack.isnull().sum(dim=["x", "y"])
    # Check % of invalid pixels in each time sample by using mean
    sum_invalid_mean = nvdi_stack.isnull().mean(dim=["x", "y"])
    # Fill the invalid pixels with value 0
    nvdi_stack_data = nvdi_stack.fillna(0)

    # Check if data is valid for machine learning. If invalid, return True and
    # an DataArray of nan values (similar to the machine learning output)
    # The number of invalid time sample less then min images
    if (sum_invalid_mean.data < 1).sum() <= min_images:
        inspect(message="Input data is invalid for this window -> skipping!")
        # create a nan dataset and return
        nan_data = zeros_like(nvdi_stack.sel(t=sum_invalid_mean.t[0], drop=True))
        nan_data = nan_data.where(lambda nan_data: nan_data > 1)
        return True, nan_data
    # Data selection: valid data for machine learning
    # select time samples where there are no invalid pixels
    good_data = get_valid_ml_inputs(nvdi_stack_data, sum_invalid, min_images)
    return False, good_data.transpose("x", "y", "t")


def apply_datacube(cube: DataArray, context: Dict) -> DataArray:
    # select atleast best 4 temporal images of ndvi for ML
    min_images = 4
    # preprocess the datacube
    invalid_data, ndvi_stack = preprocess_datacube(cube, min_images)
    # If data is invalid, there is no need to run prediction algorithm so
    # return prediction as nan DataArray and reintroduce time and bands dimensions
    if invalid_data:
        return ndvi_stack.expand_dims(dim={"t": [(cube.t.dt.year.values[0])], "bands": ["prediction"]})
    # Machine learning prediction: process the window
    result = process_window_onnx(ndvi_stack)
    # Reintroduce time and bands dimensions
    result_xarray = result.expand_dims(dim={"t": [(cube.t.dt.year.values[0])], "bands": ["prediction"]})
    # Return the resulting xarray
    return result_xarray
