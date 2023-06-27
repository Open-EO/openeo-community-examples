# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
from openeo.udf import XarrayDataCube
from typing import Dict


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:

    import numpy

    # set how much images to select in the order of highest number of clear pixels
    maxlayers=12

    # get the underlying xarray
    inputarray=cube.get_array()

    # prepare uniform coordinates
    trange=numpy.arange(numpy.datetime64(str(inputarray.t.dt.year.values[0])+'-01-01'),numpy.datetime64(str(inputarray.t.dt.year.values[0])+'-03-31'))

    # order the layers by decreasing number of clear pixels
    counts=list(sorted(zip(
        [i for i in range(inputarray.t.shape[0])],
        inputarray.count(dim=['x','y']).values.flatten()
    ), key=lambda i: i[1], reverse=True))

    # return the selected ones
    resultarray=inputarray[[i[0] for i in counts[:maxlayers]]]
    resultarray=resultarray.sortby(resultarray.t,ascending=True)
    resultarray=resultarray.assign_coords(t=trange[:maxlayers])
    return XarrayDataCube(resultarray)

