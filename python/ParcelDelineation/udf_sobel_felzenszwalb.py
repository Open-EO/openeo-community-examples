from xarray import DataArray
from skimage import segmentation, graph
from skimage.filters import sobel
from typing import Dict
from openeo.udf import inspect


def apply_datacube(cube: DataArray, context: Dict) -> DataArray:
    inspect(message=f"Dimensions of the final datacube {cube.dims}")
    # get the underlying array without the bands and t dimension
    image_data = cube.squeeze("t", drop=True).squeeze("bands", drop=True).values
    # compute edges
    edges = sobel(image_data)
    # Perform felzenszwalb segmentation
    segment = segmentation.felzenszwalb(image_data, scale=120, sigma=0.0, min_size=30, channel_axis=None)
    # Perform the rag boundary analysis and merge the segments
    bgraph = graph.rag_boundary(segment, edges)
    # merging segments
    mergedsegment = graph.cut_threshold(segment, bgraph, 0.15, in_place=False)
    # create a data cube and perform masking operations
    output_arr = DataArray(mergedsegment.reshape(cube.shape), dims=cube.dims, coords=cube.coords)
    output_arr = output_arr.where(cube >= 0.3)   # Mask the output pixels based on the cube values <0.3
    output_arr = output_arr.where(output_arr >= 0)  # Mask all values less than or equal to zero
    return output_arr
