import sys
import functools
from pathlib import Path

import numpy as np
import xarray as xr
from openeo.udf.debug import inspect
from openeo.metadata import CubeMetadata

# The onnx_deps folder contains the extracted contents of the dependencies archive provided in the job options
sys.path.append("onnx_deps")
import onnxruntime as ort

def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename the bands by using apply metadata
    :param metadata: Metadata of the input data
    :param context: Context of the UDF
    :return: Renamed labels
    """
    # rename band labels
    return metadata.rename_labels(dimension="bands", target=["probability"])


@functools.lru_cache(maxsize=5)
def _load_ort_session(model_name: str) -> ort.InferenceSession:
    """
    Loads an ONNX model from the onnx_models folder and returns an ONNX runtime session.

    Extracting the model loading code into a separate function allows us to cache the loaded model.
    This prevents the model from being loaded for every chunk of data that is processed, but only once per executor,
    which can save a lot of time, memory and ultimately processing costs.

    Should you have to download the model from a remote location, you can add the download code here, and cache the model.

    Make sure that the arguments of the method you add the @functools.lru_cache decorator to are hashable.
    Be careful with using this decorator for class methods, as the self argument is not hashable. 
    In that case you can use a static method or make sure your class is hashable (more difficult): https://docs.python.org/3/faq/programming.html#faq-cache-method-calls.

    More information on this functool can be found here: 
    https://docs.python.org/3/library/functools.html#functools.lru_cache
    """
    # The onnx_models folder contains the content of the model archive provided in the job options
    return ort.InferenceSession(f"onnx_models/{model_name}")

def _apply_model(input_xr: xr.DataArray) -> xr.DataArray:
    """
    Run the inference on the given input data using the provided ONNX runtime session.
    This method is called for each timestep in the chunk received by apply_datacube.
    """
    # Load the ONNX model
    ort_session = _load_ort_session("test_model.onnx") # name of the model in the archive

    # If there is still a 't' dimension, squeeze it, the _apply model works per time slice 
    if 't' in input_xr.dims:
        input_xr = input_xr.squeeze('t')  # remove time dimension

    # Make sure the input dimensions are in the expected order and save the original shape
    input_xr = input_xr.transpose("bands", "y", "x")
    input_shape = input_xr.shape

    # Get the underlying np.ndarray and reshape it to the expected shape.
    input_np = input_xr.values.reshape(ort_session.get_inputs()[0].shape) 

    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Return the output as an xarray DataArray
    return xr.DataArray(
        ort_outputs[0].reshape(input_shape),  # Reshape the output to the original shape (bands, y, x)
        dims=["bands", "y", "x"],
    )

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Function that is called for each chunk of data that is processed.
    The function name and arguments are defined by the UDF API.
    
    More information can be found here: 
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    CAVEAT: Some users tend to extract the underlying numpy array and preprocess it for the model using Numpy functions.
        The order of the dimensions in the numpy array might not be the same for each back-end or when running a udf locally, 
        which can lead to unexpected results. 

        It is recommended to use the named dimensions of the xarray DataArray to avoid this issue.
        The order of the dimensions can be changed using the transpose method.
        While it is a better practice to do preprocessing using openeo processes, most operations are also available in Xarray. 
    """
    # Define how you want to handle nan values
    cube = cube.fillna(0)

    # Apply the model for each timestep in the chunk
    output_data = cube.groupby("t").apply(_apply_model)

    return output_data
