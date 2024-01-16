import numpy as np
import xarray
from openeo.udf import XarrayDataCube
from skimage import segmentation
from skimage.filters import sobel
from skimage import graph


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    # get the underlying numpy array
    inarray = cube.get_array().squeeze("t", drop=True).squeeze("bands", drop=True)
    inimage = inarray.values

    # compute edges
    edges = sobel(inimage)

    # Perform felzenszwalb segmentation
    segment = np.array(
        segmentation.felzenszwalb(inimage, scale=120, sigma=0.0, min_size=30, channel_axis=None)
    ).astype(np.int32)

    # Perform the rag boundary analysis and merge the segments
    bgraph = graph.rag_boundary(segment, edges)
    # merging segments
    mergedsegment = graph.cut_threshold(segment, bgraph, 0.15, in_place=False)

    # create random numbers for the segments
    unique_classes = np.unique(mergedsegment)
    random_numbers = np.random.randint(0, 1000000, size=len(np.unique(mergedsegment)))

    counter = 0
    for unique_class in unique_classes:
        if unique_class == 0:
            continue
        mergedsegment[mergedsegment == unique_class] = random_numbers[counter]
        counter += 1

    mergedsegment = mergedsegment.astype(float)
    mergedsegment[inimage < 0.3] = np.nan
    mergedsegment[mergedsegment < 0] = 0

    outarr = xarray.DataArray(
        mergedsegment.reshape(cube.get_array().shape),
        dims=cube.get_array().dims,
        coords=cube.get_array().coords,
    )
    outarr = outarr.astype(np.float64)
    outarr = outarr.where(outarr != 0, np.nan)

    return XarrayDataCube(outarr)
