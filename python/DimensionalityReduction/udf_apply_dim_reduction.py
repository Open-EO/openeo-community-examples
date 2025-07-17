import os
import sys
import functools
import glob
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from openeo.udf import inspect
from openeo.metadata import CubeMetadata
from typing import List, Tuple, Dict
import numpy as np

sys.path.append("onnx_deps")
import onnxruntime as ort


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename and filter the bands by using apply metadata to only keep dimensionality reduction model components
    
    :param metadata: Metadata of the input data
    :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "model_type" (str): model_type to run ("PCA")
    :return: Filtered & renamed components of labels
    """
    # Get model type
    model_type = (context or {}).get("model_type", "")

    # Get amount of components from the model in the context
    _, metadata_dict = load_dim_reduction_model(model_type=model_type)
    
    # rename and reduce band labels to component labels 
    metadata = metadata.rename_labels(dimension="bands", target=metadata_dict["output_features"])
    metadata = metadata.filter_bands(metadata_dict["output_features"])

    # rename band labels
    return metadata


def is_onnx_file(file_path: str) -> bool:
    """
    Determines if a file is an ONNX file based on its extension.

    This function checks the provided file path and determines whether the file
    is an ONNX file by checking if the file name ends with the `.onnx` file extension.

    :param file_path: The path to the file whose extension is to be verified.
    :return: True if the file has a `.onnx` extension, otherwise False.
    """
    if not file_path.endswith(".onnx") or not os.path.isfile(file_path):
        inspect(message=f'Not a valid ONNX file')
        return False
    else: 
        return True


def find_model_file(model_type_id: str) -> str:
    """
    Locates a serialized dimensionality reduction model file within common temporary directories.

    This function searches recursively through the working directory
    to locate a model file named according to the pattern `dim_reduction_<model_type_id>.onnx`.
    It assumes the file has been extracted from the job’s dependency archive into a subdirectory of 
    structure like `/opt/*/work-dir/models/`, coressponding to the driver's working directory.

    :param model_type_id: The type of dimensionality reduction model (e.g., 'PCA').
                          This determines the expected filename of the model.
    :return: The full file path to the located model file.
    :raises FileNotFoundError: If the model file cannot be found in any of the predefined directories.
    """
    # Look in likely temp dirs
    model_filename = f"dim_reduction_{model_type_id.lower()}.onnx"
    
    # Model file should always be unzipped from working-directory of the Driver
    for path in glob.glob(f"/opt/**/work-dir/onnx_models/**/{model_filename}", recursive=True):
        inspect(message=f"Found model file: {path}")
        return path

    raise FileNotFoundError(f"Model file {model_filename} not found in the working directory")


def run_inference(input_np: np.ndarray, ort_session: ort.InferenceSession) -> np.ndarray:
    """
    Executes inference using an ONNX Runtime session and input numpy array. This function
    returns the dimensionality-reduced output.

    :param input_np: Numpy array containing the input tensor data for inference.
    :param ort_session: ONNX Runtime inference session object used to execute the dimensionality reduction model.
    :return: Numpy array containing the dimensionality-reduced features.
    """
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    transformed = ort_outputs[0]
    return transformed


@functools.lru_cache(maxsize=1)
def load_dim_reduction_model(model_type_id: str) -> Tuple[ort.InferenceSession, Dict[str, List[str]]]:
    """
    Loads a dimensionality reduction from a given URL, caches the model locally, and initializes an dimensionality reduction session.

    The function ensures the dimensionality reduction model is locally stored in the specified driver directory
    to optimize repeated access. It also validates if the file is a dimensionality reduction model.

    :param model_type: The type of model to load. Must be "PCA".
    :param model_dir: Directory path where the model files are located.
    :return: A dimensionality reduction model
    :raises ValueError: If model_type is invalid or if the model file is not found or invalid.
    """   
    # Process the model file to ensure it's a valid dimensionality reduction technique model
    model_path = find_model_file(model_type_id.lower())
    inspect(message=f"Downloading model file from {model_path}...")

    if not is_onnx_file(model_path):
        raise ValueError(f"No valid {model_type_id} model file found in directory: {model_path}")
    inspect(message=f"Found valid model file: {model_path}")

    # Initialize the ONNX Runtime session
    inspect(message=f"Initializing ONNX Runtime session for model at {model_path}...")
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Extract metadata
    model_meta = ort_session.get_modelmeta()

    input_features = model_meta.custom_metadata_map.get("input_features", "")
    if input_features:
        if input_features.startswith('"') and input_features.endswith('"'):
            input_features = input_features[1:-1]
    input_features = [band.strip() for band in input_features.split(",")]

    output_features = model_meta.custom_metadata_map.get("output_features", "")
    if output_features:
        if output_features.startswith('"') and output_features.endswith('"'):
            output_features = output_features[1:-1]
    output_features = [band.strip() for band in output_features.split(",")]
    n_components = len(output_features)
    inspect(message=f"Dimensionality reduction components: {n_components}")

    metadata = {
        "input_features": input_features,
        "output_features": output_features,
        "n_components": n_components
    }

    inspect(message=f"Successfully extracted metadata from model at {model_path}...")
    return ort_session, metadata 


def preprocess_input(input_xr: xr.DataArray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Preprocesses input data for model inference using an ONNX runtime session. This
    function takes an xarray DataArray, rearranges its dimensions, and reshapes its
    values to match the input requirements of the ONNX model specified by the given
    ONNX InferenceSession.

    :param input_xr: Input data in the format of an xarray DataArray. The expected
        dimensions are "y", "x", and "bands", and the order of the dimensions will
        be transposed to match this requirement.
    :param ort_session: ONNX runtime inference session that specifies the model for
        inference. Used to determine the required input shape of the model.
    :return: A tuple containing:
        - A numpy array formatted to fit the input shape of the ONNX model.
        - The original shape of the input data as a tuple with the transposed "y",
          "x", and "bands" dimensions.
    """
    # Ensure input is in ('bands', 'y', 'x') order
    input_xr = input_xr.transpose("y", "x", "bands")
    input_shape = input_xr.shape
    inspect(message=f"Input dims: {input_xr.dims}, shape: {input_xr.shape}")

    # Make numpy array 
    input_np = input_xr.values.reshape(-1, input_xr.shape[-1])

    return input_np, input_shape
    

def create_output_xarray(transformed: np.ndarray, original_shape: Tuple[int, int, int], input_xr: xr.DataArray) -> xr.DataArray:
    """
    Generate an xarray.DataArray based on dimensionality reduction components and the 
    coordinate information from the input xarray.DataArray. This function structures 
    the component data into a 3D DataArray with spatial alignment inherited from the input.

    :param n_components: The number of components (e.g., principal components) generated 
        by a dimensionality reduction algorithm.
    :param components: A 3D array (xarray.DataArray or NumPy array) containing component values 
        with shape [n_components, y, x].
    :param input_xr: The input xarray.DataArray used to extract the 'x' and 'y' coordinates, 
        ensuring spatial alignment in the output.
    :return: An xarray.DataArray with dimensions ['bands', 'y', 'x'], where 'bands' are 
        labeled as "COMP1", "COMP2", etc., and spatial coordinates are taken from input_xr.
    """
    # Construct DataArray output
    y, x, _ = original_shape
    n_components = transformed.shape[1]

    # Normalize component weights
    scaler = MinMaxScaler()
    transformed_normalized = scaler.fit_transform(transformed)
    inspect(message=f"Normalizing ouput components...")

    # Reshape back to (n_components/back, y, x)
    reshaped = transformed_normalized.T.reshape((n_components, y, x))

    # Construct output array
    coords = {
        "bands": [f"COMP{i+1}" for i in range(n_components)],
        "y": input_xr.coords["y"],
        "x": input_xr.coords["x"],
    }

    result = xr.DataArray(reshaped, dims=("bands", "y", "x"), coords=coords)
    inspect(message=f"Output dims: {result.dims}, shape: {result.shape}")

    # Attach input cube attrs 
    result.attrs = input_xr.attrs
    result = result.rename('__xarray_dataarray_components__')

    return result


def apply_datacube(cube: xr.DataArray, context: dict = None) -> xr.DataArray:
    """
    Applies a dimensionality reduction model on a given data cube for dimensionality reduction.
    The function ensures that the input cube is processed to fill any missing values and 
    is in the correct data type to be compatible with the models. 

    Note: The function name and arguments are defined by the UDF API.
    More information can be found here:
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    :param cube: The data cube on which dimensionality reduction will be applied. It must be an `xr.DataArray`.
    :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "model_type" (str): model_type ro run ("PCA")
    :return: An `xr.DataArray` representing the processed output cube after successfully applying the model
    """  
    # fill nan in cube and make sure cube is in right dtype for dimensionality reduction
    cube = cube.fillna(0)
    cube = cube.astype("float32")

    # Get model type
    model_type = (context or {}).get("model_type", "")
    inspect(message=f"Running model: {model_type}")

    # Load the ONNX model and extract metadata
    ort_session, _ = load_dim_reduction_model(model_type=model_type)
   
    # preprocess input array to numpy array in correct shape
    input_np, input_shape = preprocess_input(cube)

    # Apply Dimensionality Reduction model by running inference
    inspect(message=f"Running inference ...")
    component_array = run_inference(input_np, ort_session)

    # Build coords and return xr.DataArray
    result = create_output_xarray(transformed=component_array, 
                                  original_shape=input_shape,
                                  input_xr=cube)
    
    # make sure output Xarray has the correct dtype
    result = result.astype("float32")
    return result
