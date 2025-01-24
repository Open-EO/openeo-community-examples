from xarray import DataArray
from skimage import segmentation, graph
from skimage.filters import sobel


def apply_datacube(cube: DataArray, context: dict) -> DataArray:
    # get the underlying array without the bands and t dimension
    _data = cube.squeeze("t", drop=True).squeeze("bands", drop=True).values
    # compute edges
    edges = sobel(_data)
    # Perform felzenszwalb segmentation
    segment = segmentation.felzenszwalb(_data, scale=120, sigma=0.0, min_size=30, channel_axis=None)
    # Perform the rag boundary analysis and merge the segments
    bgraph = graph.rag_boundary(segment, edges)
    # merging segments
    mergedsegment = graph.cut_threshold(segment, bgraph, 0.15, in_place=False)
    # create a data cube and perform masking operations
    output_arr = DataArray(mergedsegment.reshape(cube.shape), dims=cube.dims, coords=cube.coords)
    output_arr = output_arr.where(cube >= 0.3)   # Mask the output pixels based on the cube values <0.3
    output_arr = output_arr.where(output_arr >= 0)  # Mask all values less than or equal to zero
    return output_arr

