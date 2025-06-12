from openeo.udf import inspect
from xarray import DataArray
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from openeo.metadata import CubeMetadata
from typing import Dict


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
    new_band_names = [f"PC{i+1}" for i in range(n_components)] + bands[n_components:]
    
    # rename band labels
    metadata = metadata.rename_labels(dimension="bands", target=new_band_names)
    metadata = metadata.filter_bands([f"PC{i+1}" for i in range(n_components)])

    # rename band labels
    return metadata


def apply_datacube(cube: DataArray, context: Dict = None) -> DataArray:
    """
    Applies a PCA model on a given data cube for inference. The function ensures that the input
    cube is processed to fill any missing values and is in the correct data type to be compatible with the
    models. 

    Note: The function name and arguments are defined by the UDF API.
    More information can be found here:
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    :param cube: The data cube on which CPA will be applied. It must be an `xr.DataArray`.
    :param context: A dictionary that includes the amount of output components of the PCA model
    :return: An `xr.DataArray` representing the processed output cube after successfully applying the model
    """
    # Ensure input is in ('bands', 'y', 'x') order
    cube = cube.transpose('bands', 'y', 'x')
    inspect(f"Input dims: {cube.dims}, shape: {cube.shape}")
    
    # Get amount of components from context, Default 3 components
    n_components = (context or {}).get("n_components", 3)
    inspect(f"PCA components: {n_components}, Default 3")

    # Reshape to (pixels, bands)
    bands, y, x = cube.shape
    data = cube.values.reshape((bands, y * x)).T  # shape: (pixels, bands)

    # Impute NaNs
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_imputed)  # shape: (pixels, n_components)

    # Normalize PCA output
    scaler = MinMaxScaler()
    transformed_normalized = scaler.fit_transform(transformed)
    
    # Reshape back to (n_components, y, x)
    result_data = transformed_normalized.T.reshape((n_components, y, x))

    # Build coords and return DataArray
    coords = {
        "bands": [f"PC{i+1}" for i in range(n_components)],
        "y": cube.coords["y"],
        "x": cube.coords["x"],
    }

    result = DataArray(result_data, dims=("bands", "y", "x"), coords=coords)
    inspect(f"Output dims: {result.dims}, shape: {result.shape}")
    return result
