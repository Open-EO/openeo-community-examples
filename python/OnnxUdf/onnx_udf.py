import functools
import sys
from typing import Dict
import xarray as xr
import numpy as np
from openeo.udf.debug import inspect

# The onnx_deps folder contains the extrcted contents of the dependencies archive provided in the job options
sys.path.insert(0, "onnx_deps") 
import onnxruntime as ort

inspect(message="UDF initialized")

@functools.lru_cache(maxsize=5)
def _load_ort_session(model_name: str) -> ort.InferenceSession:
    """
    Loads an ONNX model from the onnx_models folder and returns an ONNX runtime session.

    Extracting the model loading code into a separate function allows us to cache the loaded model.
    This prevents the model from being loaded for every chunk of data that is processed, but only once per executor.

    Should you have to download the model from a remote location, you can add the download code here, and cache the model.

    Make sure that the arguments of the method you add the @functools.lru_cache decorator to are hashable.
    Be carefull with using this decorator for class methods, as the self argument is not hashable. In that case, you can use a static method.

    More information on this functool can be found here: https://docs.python.org/3/library/functools.html#functools.lru_cache
    """
    inspect(message="Loading model")
    # the onnx_models folder contians the content of the model archive provided in the job options
    return ort.InferenceSession(f"onnx_models/{model_name}")

def _apply_model(input_xr: xr.DataArray) -> xr.DataArray:
    """
    Run the inference on the given input data using the provided ONNX runtime session.
    This method is called for each timestep in the chunk received by apply_datacube.
    """
    inspect(message="Applying model")
    ort_session = _load_ort_session("test_model.onnx") # name of the model in the archive

    # Make sure the input dimensions are in the expected order and save the original shape
    input_xr = input_xr.transpose("bands", "y", "x")
    input_shape = input_xr.shape

    # Get the underlying np.ndarray and reshape it to the expected shape.
    input_np = input_xr.values.reshape(ort_session.get_inputs()[0].shape) 

    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_outputs = ort_session.run(None, ort_inputs)

    return xr.DataArray(
        ort_outputs[0].reshape(input_shape),  # Reshape the output to the original shape (bands, y, x)
        dims=["bands", "y", "x"],
    )

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Function that is called for each chunk of data that is processed.
    The function name and arguments are defined by the UDF API.
    
    More information can be found here: https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures
    """
    inspect(message="apply_datacube called")
    
    output_data = cube.groupby("t").apply(_apply_model)

    return output_data
