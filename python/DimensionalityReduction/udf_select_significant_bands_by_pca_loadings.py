import os
import sys
import functools
import glob
import xarray as xr
import numpy as np
from openeo.metadata import CubeMetadata
from openeo.udf import inspect
from typing import Dict, List, Tuple

sys.path.append("onnx_deps")
import onnxruntime as ort 


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """
    Filter the significant bands of the dimensionality reduction model by using apply metadata
    
    :param metadata: Metadata of the input data
   
     :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "threshold" (float): The minimum absolute value a component loading must
                          have to consider its corresponding band significant.
    :return: Filtered bands
    """
    # Get bands to filter or get original bands
    idx, _, _ = significant_bands_from_pca(context)
    # Filter bands
    metadata = metadata.filter_bands(idx.tolist())
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


@functools.lru_cache(maxsize=1)
def load_dim_reduction_model(model_type_id: str) -> Tuple[ort.InferenceSession, Dict[str, List[str]]]:
    """
    Loads a dimensionality reduction from a given URL, caches the model locally, and initializes an dimensionality reduction session.

    The function ensures the dimensionality reduction model is locally stored in the specified driver directory
    to optimize repeated access. It also validates if the file is a dimensionality reduction model.

    :param model_type: The type of model to load. Must be either "PCA".
    :param model_dir: Directory path where the model files are located.
    :return: A dimensionality reduction model
    :raises ValueError: If model_type is invalid or if the model file is not found or invalid.
    """    
    # Process the model file to ensure it's a valid dimensionality reduction technique model
    model_path = find_model_file(model_type_id)
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


def significant_bands_from_pca(context: dict = None):
    """
    Return indices, scores and bands of the most influential input bands
    for a PCA model stored in ONNX format.

    :param context: Optional dictionary containing configuration values.
                    Expected key(s):
                        - threshold(float): Normalized contribution score threshold. Bands with scores below this are discarded.
                        Ignored if `top_k` is provided.
                        - top_k (int): If given, returns exactly the top_k bands by contribution score.
    :return: tuple
        A tuple of three elements:
        - indices (np.ndarray): 0-based indices of bands meeting the criterion.
        - scores (np.ndarray): normalized contribution scores for all bands (sum to 1).
        - labels (np.ndarray): band coordinate labels if `cube` is an xarray with coords.
    """
    # Get context parameters:
    threshold = (context or {}).get("threshold", None)
    top_k = (context or {}).get("top_k", None)
    if threshold is not None:
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = None
    if top_k is not None:
        try:
            top_k = int(top_k)
        except ValueError:
            top_k = None

    # Spin up ONNX Runtime session
    session, metadata = load_dim_reduction_model(model_type_id="PCA")
    input_name = session.get_inputs()[0].name
    n_bands = len(metadata["input_features"])

    # Probe the model to recover loadings
    I   = np.eye(n_bands, dtype=np.float32) # (bands, bands)
    Z   = np.zeros_like(I)

    Y_I = session.run(None, {input_name: I})[0] # (bands, n_components)
    Y_0 = session.run(None, {input_name: Z})[0]

    loadings = Y_I - Y_0 # cancel mean term

    # Contribution score per band
    scores = np.abs(loadings).sum(axis=1)
    scores = scores / scores.sum() # normalise to 1

    # Select significant bands
    if top_k is not None:
        idx = scores.argsort()[-top_k:][::-1]
    else:
        idx = np.where(scores >= threshold)[0]

    # keep original xarray band labels if present
    input_features_array = np.array(metadata["input_features"])
    labels = input_features_array[idx]

    return idx, scores, labels


def apply_datacube(cube: xr.DataArray, context: dict = None) -> xr.DataArray:
    """
    Applies PCA to explore the most significant dimensions.
    
    :param cube: The data cube on which dimensionality reduction will be applied. It must be an `xr.DataArray`.
    :param context: Optional dictionary containing configuration values.
                    Expected key(s):
                        - threshold(float): Normalized contribution score threshold. Bands with scores below this are discarded.
                        Ignored if `top_k` is provided.
                        - top_k (int): If given, returns exactly the top_k bands by contribution score.
    :return: An `xr.DataArray` representing the most significant dimensions of the input cube after successfully applying the model
    """
    # fill nan in cube and make sure cube is in right dtype for dimensionality reduction
    cube = cube.fillna(0)
    cube = cube.astype("float32")
   
    # Ensure input is in ('bands', 'y', 'x') order
    cube = cube.transpose('bands', 'y', 'x')
    inspect(message=f"Input dims: {cube.dims}, shape: {cube.shape}")

    # Apply Dimensionality Reduction model
    significant_bands_mask, _, _ = significant_bands_from_pca(context)
        
    # Safety check: ensure we are not selecting 0 bands
    if significant_bands_mask.size == 0:
        inspect(message="Warning: No significant bands found. Returning original cube.")
        return cube  # Or raise an error if strict handling is needed

    significant_band_cube = cube.isel(bands=significant_bands_mask)
    inspect(message=f"Filtered dims: {significant_band_cube.dims}, shape: {significant_band_cube.shape}")

    return significant_band_cube
