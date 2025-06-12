import xarray
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from openeo.metadata import CollectionMetadata


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    return metadata.rename_labels(
        dimension = "bands",
        target = ["contrast","variance","NDFI"]
    )


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    """
    Applies spatial texture analysis and spectral index computation to a Sentinel-2 data cube.

    Computes:
    - NDFI (Normalized Difference Fraction Index) from bands B08 and B12
    - Texture features (contrast and variance) using Gray-Level Co-occurrence Matrix (GLCM)

    Args:
        cube (xarray.DataArray): A 3D data cube with dimensions (bands, y, x) containing at least bands B08 and B12.
        context (dict): A context dictionary (currently unused, included for API compatibility).

    Returns:
        xarray.DataArray: A new data cube with dimensions (bands, y, x) containing:
                          - 'contrast': GLCM contrast
                          - 'variance': GLCM variance
                          - 'NDFI': Normalised Difference Fire Index
    """
    
    # Parameters
    window_size = 33
    pad = window_size // 2
    levels = 256  # For 8-bit images
    
    # Load Data
    # data = cube.values # shape: (t, bands, y, x)
    
    #first get NDFI
    b08 = cube.sel(bands="B08")
    b12 = cube.sel(bands="B12")

    # Compute mean values
    avg_b08 = b08.mean()
    avg_b12 = b12.mean()

    # Calculate NDFI
    ndfi = ((b12 / avg_b12) - (b08 / avg_b08)) / (b08 / avg_b08)
    
    # Padding the image to handle border pixels for GLCM
    padded = np.pad(b12, pad_width=pad, mode='reflect')

    # Normalize to 0–255 range
    img_norm = (padded - padded.min()) / (padded.max() - padded.min())
    padded = (img_norm * 255).astype(np.uint8)
    
    # Initialize feature maps
    shape = b12.shape
    contrast = np.zeros(shape)
    variance = np.zeros(shape)
    
    for i in range(pad, pad + shape[0]):
        for j in range(pad, pad + shape[1]):
            window = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            
            # Compute GLCM
            glcm = graycomatrix(window, distances=[5], angles=[0], levels=levels, symmetric=True, normed=True)
            
            # Texture features
            contrast[i - pad, j - pad] = graycoprops(glcm, 'contrast')[0, 0]
            variance[i - pad, j - pad] = np.var(window)

    all_texture = np.stack([contrast,variance,ndfi])
    # create a data cube with all the calculated properties
    textures = xarray.DataArray(
        data=all_texture,
        dims=["bands", "y", "x"],
        coords={"bands": ["contrast","variance","NDFI"], "y": cube.coords["y"], "x": cube.coords["x"]},
    )

    return textures