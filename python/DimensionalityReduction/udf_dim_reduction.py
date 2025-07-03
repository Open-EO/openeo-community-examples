import os
import functools
import joblib
import glob
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding as SE, LocallyLinearEmbedding as LLE
# from umap import UMAP #TODO implement umap
from sklearn.preprocessing import MinMaxScaler
from openeo.udf import inspect
from openeo.metadata import CubeMetadata
from typing import Dict, Union
import numpy as np


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename and filter the bands by using apply metadata to only keep dimensionality reduction model components
    
    :param metadata: Metadata of the input data
    :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "model_type" (str): model_type to run ("PCA" | "TSNE" | "LLE" | "SE")
    :return: Filtered & renamed components of labels
    """
    # Get model type
    model_type = (context or {}).get("model_type", "")

    # Get amount of components from the model in the context
    dim_reduction_model = load_dim_reduction_model(model_type=model_type)
    n_components = dim_reduction_model.n_components

    # Original list of band names
    bands = metadata.band_names  

    # Rename only the first `n_components` bands
    new_band_names = [f"COMP{i+1}" for i in range(n_components)] + bands[n_components:]
    
    # rename and reduce band labels to component labels 
    metadata = metadata.rename_labels(dimension="bands", target=new_band_names)
    metadata = metadata.filter_bands([f"COMP{i+1}" for i in range(n_components)])

    # rename band labels
    return metadata


def is_dim_reduction_model_file(file_path: str) -> bool:
    """
    Determines if a file is a pickle file that contains a dimensionality reduction model

    This function checks whether the file has a `.pkl` extension and attempts to load it,
    verifying that it contains a valid dimensionality reduction model object.

    :param file_path: The path to the file.
    :return: True if the file has a `.pkl` extension and contains a PCA, t-SNE, LLE or SE model, otherwise False.
    """
    if not file_path.endswith(".pkl") or not os.path.isfile(file_path):
        inspect(message=f'Not a valid pickle file')
        return False

    try:
        with open(file_path, 'rb') as f:
            model = joblib.load(f)
        
        return isinstance(model, (PCA, TSNE, SE, LLE))

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def find_model_file(model_type: str) -> str:
    """
    Locates a serialized dimensionality reduction model file within common temporary directories.

    This function searches recursively through a set of predefined directories (e.g., /tmp, /opt, /mnt, /home)
    to locate a model file named according to the pattern `dim_reduction_<model_type>.pkl`.
    It assumes the file has been extracted from the job’s dependency archive into a subdirectory of 
    structure like `*/work-dir/models/`, coressponding to the driver's working directory.

    :param model_type: The type of dimensionality reduction model (e.g., 'PCA', 'TSNE', 'SE', 'LLE').
                       This determines the expected filename of the model.
    :return: The full file path to the located model file.
    :raises FileNotFoundError: If the model file cannot be found in any of the predefined directories.
    """
    # Look in likely temp dirs
    possible_dirs = ["/tmp", "/opt", "/mnt", "/home"]  # backend-specific
    model_filename = f"dim_reduction_{model_type.lower()}.pkl"
    
    for base_dir in possible_dirs:
        # Model file should always be unzipped from working-drectory of the Driver
        for path in glob.glob(f"{base_dir}/**/work-dir/models/{model_filename}", recursive=True):
            inspect(message=f"Found model file: {path}")
            return path
    
    raise FileNotFoundError(f"Model file {model_filename} not found in any of {possible_dirs}")


@functools.lru_cache(maxsize=1)
def load_dim_reduction_model(model_type: str) -> Union[PCA, TSNE, LLE, SE]:
    """
    Loads a dimensionality reduction from a given URL, caches the model locally, and initializes an dimensionality reduction session.

    The function ensures the dimensionality reduction model is locally stored in the specified driver directory
    to optimize repeated access. It also validates if the file is a dimensionality reduction model.

    :param model_type: The type of model to load. Must be either "PCA", "T-SNE", "LLE" or "SE".
    :param model_dir: Directory path where the model files are located.
    :return: A dimensionality reduction model
    :raises ValueError: If model_type is invalid or if the model file is not found or invalid.
    """
    valid_types = {"PCA", "TSNE", "LLE", "SE"}
    if model_type not in valid_types:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {valid_types}")
    
    try:
        # Process the model file to ensure it's a valid dimensionality reduction technique model
        model_path = find_model_file(model_type)
        inspect(message=f"Downloading model file from {model_path}...")

        if not is_dim_reduction_model_file(model_path):
            raise ValueError(f"No valid {model_type} model file found in directory: {model_path}")
        
        inspect(message=f"Found valid model file: {model_path}")
        return joblib.load(model_path)
    
    except Exception as e:
        raise ValueError(f"Failed to load reduction model from {model_path}: {e}")


def apply_datacube(cube: xr.DataArray, context: Dict = None) -> xr.DataArray:
    """
    Applies a dimensionality reduction model on a given data cube for dimensionality reduction. The function ensures that the input
    cube is processed to fill any missing values and is in the correct data type to be compatible with the
    models. 

    Note: The function name and arguments are defined by the UDF API.
    More information can be found here:
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    :param cube: The data cube on which dimensionality reduction will be applied. It must be an `xr.DataArray`.
    :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "model_type" (str): model_type ro tun ("PCA" | "TSNE" | "LLE" | "SE")
    :return: An `xr.DataArray` representing the processed output cube after successfully applying the model
    """  
    # fill nan in cube and make sure cube is in right dtype for dimensionality reduction
    cube = cube.fillna(0)
    cube = cube.astype("float32")
   
    # Ensure input is in ('bands', 'y', 'x') order
    cube = cube.transpose('bands', 'y', 'x')
    inspect(message=f"Input dims: {cube.dims}, shape: {cube.shape}")

    # Reshape to (pixels, bands)
    bands, y, x = cube.shape
    data = cube.values.reshape((bands, y * x)).T  # shape: (pixels, bands)

   # Get model type
    model_type = (context or {}).get("model_type", "")

    # Apply Dimensionality Reduction model
    dim_reduction_model = load_dim_reduction_model(model_type=model_type)
    n_components = dim_reduction_model.n_components
    inspect(message=f"Dimensionality reduction components: {n_components}")
    inspect(message=f"Fitting dimensionality reduction model...")
    transformed = dim_reduction_model.fit_transform(data)  # shape: (pixels, n_components)

    # Normalize Dimensionality Reduction output
    inspect(message=f"Normalizing ouput components...")
    scaler = MinMaxScaler()
    transformed_normalized = scaler.fit_transform(transformed)
    
    # Reshape back to (n_components, y, x)
    result_data = transformed_normalized.T.reshape((n_components, y, x))

    # Build coords and return xr.DataArray
    coords = {
        "bands": [f"COMP{i+1}" for i in range(n_components)],
        "y": cube.coords["y"],
        "x": cube.coords["x"],
    }
    result = xr.DataArray(result_data, dims=("bands", "y", "x"), coords=coords)
    inspect(message=f"Output dims: {result.dims}, shape: {result.shape}")

    # Attach result cube attrs 
    result.attrs = cube.attrs
    result.rename('__xarray_dataarray_components__')
    
    # make sure output Xarray has the correct dtype
    result.astype("float32")
    return result
