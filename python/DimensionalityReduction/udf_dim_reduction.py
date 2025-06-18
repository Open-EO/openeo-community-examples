# /// script
# dependencies = [
#   'joblib==1.4.2',
#   'scikit-learn==1.3.2',
#   'netCDF4==1.6.5'
# ]
# ///
#
# This openEO UDF script implements dimension reduction models from scikit-learn. Packages installed are meanr for modelling and storage handling of the data.

import os
import functools
import joblib
import requests
import shutil
import tempfile
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urlparse
from openeo.udf import inspect
from openeo.metadata import CubeMetadata
from typing import Dict, Union


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename and filter the bands by using apply metadata
    
    :param metadata: Metadata of the input data
    :param context: Context of the UDF
    :return: Renamed labels
    """
    # Get amount of components from context
    n_components = (context or {}).get("n_components", 3)
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
    Determines if a file is a pickle file that contains a PCA or t-SNE model.

    This function checks whether the file has a `.pkl` extension and attempts to load it,
    verifying that it contains a scikit-learn PCA or t-SNE model.

    :param file_path: The path to the file.
    :return: True if the file has a `.pkl` extension and contains a PCA or t-SNE model, otherwise False.
    """
    if not file_path.endswith(".pkl") or not os.path.isfile(file_path):
        return False

    try:
        with open(file_path, 'rb') as f:
            model = joblib.load(f)
        
        return isinstance(model, (PCA, TSNE))

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def download_file(url: str, max_file_size_mb: int = 100, cache_dir: str = "/tmp/cache") -> str:
    """
    Downloads a file from the specified URL. The file is
    cached in a given directory, and downloads of the same file are prevented using a locking
    mechanism. If the file already exists in the cache, it will not be downloaded again.

    :param url: The URL of the file to download.
    :param max_file_size_mb: Maximum allowable file size in megabytes. Defaults to 100 MB.
    :param cache_dir: Directory where the downloaded file will be cached. Defaults to '/tmp/cache'.
    :return: The path to the downloaded file in the cache directory.

    :raises ValueError: If the file size exceeds the maximum limit, or if there is an issue during the
                        download process.
    """
    # Construct the file path within the cache directory (e.g., '/tmp/cache/PCA_model.pkl')
    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists

    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(cache_dir, file_name)

    if os.path.exists(file_path):
        inspect(message=f"File {file_path} already exists in cache.")
        return file_path

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error if the request fails

        file_size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_file_path = temp_file.name
            for chunk in response.iter_content(chunk_size=1024):
                temp_file.write(chunk)
                file_size += len(chunk)
                if file_size > max_file_size_mb * 1024 * 1024:
                    raise ValueError(f"Downloaded file exceeds the size limit of {max_file_size_mb} MB")

        shutil.move(temp_file_path, file_path)
        return file_path

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Cleanup if an error occurs
        raise ValueError(f"Error downloading file: {e}")


@functools.lru_cache(maxsize=1)
def load_dim_reduction_model(model_url: str, cache_dir: str = "/tmp/cache") -> Union[PCA, TSNE]:
    """
    Loads an PCA or t-SNE model from a given URL, caches the model locally, and initializes an dimensionality reduction session.

    The function ensures the dimensionality reduction model is downloaded and locally stored in the specified cache directory
    to optimize repeated access. It also validates if the file is a valid PCA or TSNE model.

    :param model_url: The URL pointing to the dimensionality reduction model to be downloaded and loaded. The URL must provide
                      a valid dimensionality reduction file.
    :param cache_dir: An optional directory path to store the cached dimensionality reduction model. Defaults to '/tmp/cache',
                      ensuring local caching for repeated access.
    :return: A PCA or t-SNE dimensionality reduction model
    :raises ValueError: If the dimensionality reduction model fails to load, initialize, or if the downloaded file
                        is not a valid PCA or t-SNE model
    """
    try:
        # Process the model file to ensure it's a valid dimensionality reduction technique model (PCA or t-SNE)
        inspect(message=f"downloading model file from {model_url}...")
        model_path = download_file(model_url, cache_dir=cache_dir)
        
        if not is_dim_reduction_model_file(model_path):
            os.remove(model_path)
            raise ValueError(f"Downloaded file is not a valid PCA or t-SNE pickle file.")
       
        return joblib.load(model_path)
    
    except Exception as e:
        raise ValueError(f"Failed to load reduction model from {model_url}: {e}")


def apply_datacube(cube: xr.DataArray, context: Dict = None) -> xr.DataArray:
    """
    Applies a PCA or t-SNE model on a given data cube for dimensionality reduction. The function ensures that the input
    cube is processed to fill any missing values and is in the correct data type to be compatible with the
    models. 

    Note: The function name and arguments are defined by the UDF API.
    More information can be found here:
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    :param cube: The data cube on which CPA will be applied. It must be an `xr.DataArray`.
    :param context: A dictionary that includes the amount of output components of the PCA model
    :return: An `xr.DataArray` representing the processed output cube after successfully applying the model
    """  
    # fill nan in cube and make sure cube is in right dtype for dimensionality reduction
    cube = cube.fillna(0)
    cube = cube.astype("float32")
   
    # Ensure input is in ('bands', 'y', 'x') order
    cube = cube.transpose('bands', 'y', 'x')
    inspect(f"Input dims: {cube.dims}, shape: {cube.shape}")

    # Reshape to (pixels, bands)
    bands, y, x = cube.shape
    data = cube.values.reshape((bands, y * x)).T  # shape: (pixels, bands)

   # Get model URL
    model_path = (context or {}).get("model_url", "")

    # Apply Dimensionality Reduction model
    dim_reduction_model = load_dim_reduction_model(model_url=model_path, cache_dir = "/tmp/cache")
    n_components = dim_reduction_model.n_components
    inspect(f"Dimensionality reduction components: {n_components}")
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
    inspect(f"Output dims: {result.dims}, shape: {result.shape}")

    # Attach result cube attrs 
    result.attrs = cube.attrs
    result.rename('__xarray_dataarray_components__')
    
    # make sure output Xarray has the correct dtype
    result.astype("float32")
    return result
