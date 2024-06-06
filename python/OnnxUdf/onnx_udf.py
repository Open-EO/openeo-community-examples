import functools
import sys
from openeo.udf import XarrayDataCube
from typing import Dict
import xarray as xr
from openeo.udf.debug import inspect

sys.path.insert(0, "onnx_deps")
import onnxruntime as ort

@functools.lru_cache(maxsize=5)
def _load_ort_session(model_name: str):
    """
    Extracting the model loading code into a separate function allows us to cache the loaded model.
    This prevents the model from being loaded for every chunk of data that is processed.

    Should you have to download the model from a remote location, you can add the download code here, and cache the model.

    Make sure that the arguments of the method you add the @functools.lru_cache decorator to are hashable.
    More information on this functool can be found here: https://docs.python.org/3/library/functools.html#functools.lru_cache
    """

    # the onnx_models folder contians the content of the model archive provided in the job options
    return ort.InferenceSession(f"onnx_models/{model_name}") 

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    cube = cube.transpose("bands", "y", "x")  # Make sure the data is in the correct order
    input_np = cube.values  # Only perform inference for the first date and get the numpy array
    input_np = input_np.reshape(1,1,256,256)  # Neural network expects shape (1, 1, 256, 256)

    # Load the model
    ort_session = _load_ort_session("test_model.onnx") #name of the model in the archive

    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    output_np = ort_outputs[0].reshape(1,256,256)  # Reshape the output to the expected shape

    # Convert the output back to an xarray DataArray
    output_data = xr.DataArray(
        output_np,
        dims=["bands", "y", "x"],
    )
    inspect(output_data, "output data")

    return XarrayDataCube(output_data)
