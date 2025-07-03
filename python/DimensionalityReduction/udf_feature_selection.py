import os
import functools
import joblib
import glob
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from openeo.metadata import CubeMetadata
from openeo.udf import inspect
from typing import Union


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
    significant_bands_mask = get_significant_band_mask(context)
    significant_bands_list = np.where(significant_bands_mask)[0].tolist()

    # Filter bands
    metadata = metadata.filter_bands(significant_bands_list)
    # rename band labels
    return metadata


def is_dim_reduction_model_file(file_path: str) -> bool:
    """
    Determines if a file is a pickle file that contains a PCA model.

    This function checks whether the file has a `.pkl` extension and attempts to load it,
    verifying that it contains a scikit-learn PCA model.

    :param file_path: The path to the file.
    :return: True if the file has a `.pkl` extension and contains a PCA model, otherwise False.
    """
    if not file_path.endswith(".pkl") or not os.path.isfile(file_path):
        inspect(message=f'Not a valid pickle file')
        return False

    try:
        with open(file_path, 'rb') as f:
            model = joblib.load(f)

        return isinstance(model, (PCA))

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def find_model_file(model_type: str) -> str:
    """
    Locates a serialized dimensionality reduction model file within common temporary directories.

    This function searches recursively through a set of predefined directories (e.g., /tmp, /opt, /mnt, /home)
    to locate a model file named according to the pattern `dim_reduction_<model_type>.pkl`.
    It assumes the file has been extracted from the job’s dependency archive into a subdirectory of 
    structure like `*/work-dir/models/`, coressponding to the driver's working directory.

    :param model_type: The type of dimensionality reduction model (e.g., 'PCA').
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
def load_dim_reduction_model(model_type: str = "PCA") -> Union[PCA]:
    """
    Loads an PCA model from a given URL, caches the model locally, and initializes an dimensionality reduction session.

    The function ensures the dimensionality reduction model is locally stored in the specified driver directory
    to optimize repeated access. It also validates if the file is a valid PCA.

    :param model_type: The type of model to load. Only "PCA" allowed at the moment.
    :param model_dir: Directory path where the model files are located.
    :return: A PCA dimensionality reduction model
    :raises ValueError: If model_type is invalid or if the model file is not found or invalid.
    """
    valid_types = {"PCA"}
    if model_type not in valid_types:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {valid_types}")
    
    try:
        # Find the model file in the working directory of the driver
        model_path = find_model_file(model_type)
        inspect(message=f"Downloading model file from {model_path}...")

        # Process the model file to ensure it's a valid dimensionality reduction technique model
        if not is_dim_reduction_model_file(model_path):
            raise ValueError(f"No valid {model_type} model file found in directory: {model_path}")
        
        inspect(message=f"Found valid model file: {model_path}")
        return joblib.load(model_path)
    
    except Exception as e:
        raise ValueError(f"Failed to load reduction model from {model_path}: {e}")


def get_significant_band_mask(context: dict = None):
    """
    Identifies significant spectral bands based on a threshold applied to PCA component loadings.

    This function performs dimensionality reduction using PCA on a dataset (model must be pre-loaded)
    and extracts the spectral bands that contribute significantly to the principal components. A band
    is considered significant if the absolute value of any of its loadings in the PCA components is
    greater than or equal to the specified threshold.

    :param context: Optional dictionary containing configuration values.
                    Expected key:
                        - "threshold" (float): The minimum absolute value a component loading must
                          have to consider its corresponding band significant.
    :return: A NumPy array (mask) indicating which spectral bands are significant.
             Shape corresponds to the number of bands in the input data.
    :raises ValueError: If the dimensionality reduction model used is not PCA (e.g., t-SNE or UMAP).
    """ 
    
    # Get threshold significance
    threshold = (context or {}).get("threshold", "")

    # Apply Dimensionality Reduction model
    dim_reduction_model = load_dim_reduction_model(model_type= "PCA")
    n_components = dim_reduction_model.n_components
    inspect(message=f"Dimensionality reduction components: {n_components}")
    
    if hasattr(dim_reduction_model, 'components_') and isinstance(dim_reduction_model, PCA):
        components = dim_reduction_model.components_
        
        # Find significant bands
        significant_bands_mask = np.any(np.abs(components) >= threshold, axis=0)
        inspect(message=f"Significant bands: {np.where(significant_bands_mask)[0].tolist()}")
        
        return significant_bands_mask
    else:
        # t-SNE does not have components_, so we cannot select significant bands
        raise ValueError("Significance-based filtering is only supported with PCA, not T-SNE or UMAP")
        


def apply_datacube(cube: xr.DataArray, context: dict = None) -> xr.DataArray:
    """
    Applies PCA to explore the most significant dimensions.
    
    :param cube: The data cube on which dimensionality reduction will be applied. It must be an `xr.DataArray`.
    :param context: A dictionary that includes the model_type to run
    :return: An `xr.DataArray` representing the most significant dimensions of the input cube after successfully applying the model
    """
    # fill nan in cube and make sure cube is in right dtype for dimensionality reduction
    cube = cube.fillna(0)
    cube = cube.astype("float32")
   
    # Ensure input is in ('bands', 'y', 'x') order
    cube = cube.transpose('bands', 'y', 'x')
    inspect(message=f"Input dims: {cube.dims}, shape: {cube.shape}")

    # Apply Dimensionality Reduction model
    significant_bands_mask = get_significant_band_mask(context)
        
    # Apply Mask on bands
    significant_band_cube = cube.isel(bands=significant_bands_mask)
    inspect(message=f"Filtered dims: {significant_band_cube.dims}, shape: {significant_band_cube.shape}")

    return significant_band_cube
