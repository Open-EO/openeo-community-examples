import xarray as xr
import numpy as np
from openeo.udf import XarrayDataCube
from skimage.feature import graycomatrix, graycoprops


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    
    # Parameters
    window_size = 33
    pad = window_size // 2
    levels = 256  # For 8-bit images
    
    # Load Data
    data = cube.get_array()  # shape: (t, bands, y, x)
    
    #first get NDFI
    b08 = data.sel(bands="B08")
    b12 = data.sel(bands="B12")

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
        coords={"bands": ["contrast","variance","NDFI"], "y": cube.array.y, "x": cube.array.x},
    )

    return XarrayDataCube(textures)