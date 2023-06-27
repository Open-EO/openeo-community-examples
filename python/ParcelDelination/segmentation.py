import functools
import os
from tensorflow.keras.models import load_model
import numpy as np
import gc
import logging
from tensorflow.keras.backend import clear_session
from xarray.core.common import ones_like
from xarray.ufuncs import isnan as ufuncs_isnan
import xarray
from openeo.udf import XarrayDataCube
from typing import Dict

# Needed because joblib hijacks root logger
logging.basicConfig(level=logging.INFO)


@functools.lru_cache(maxsize=25)
def load_models(modeldir):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.

    @param modeldir: Model directory
    @return: Loaded models
    """

    print('Loading convolutional neural networks ...')
    weightsmodel1 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network1.h5')
    weightsmodel2 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network2.h5')
    weightsmodel3 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network3.h5')
    return [
        load_model(weightsmodel1),
        load_model(weightsmodel2),
        load_model(weightsmodel3)
    ]

class Segmentation():
    
    
    def __init__(self, logger=None):
        if logger is None:
            self.log = logging.getLogger(__name__)
        else: self.log=logger
        self.models=None
    
    def processWindow(self, models, data, patch_size = 128):
    
        model1 = models[0]
        model2 = models[1]
        model3 = models[2]
        
        ## read the xarray data as a numpy array
        ndvi_stack = data['s2_ndvi'].values.copy()
        mask_stack = data['s2_mask'].values.copy()
    
        ## mask the ndvi data
        ndvi_stack[mask_stack == 1] = 255
        ndvi_stack[ndvi_stack > 250] = 255
    
        ## count the amount of invalid data per acquisition and sort accordingly
        sum_invalid = np.sum(ndvi_stack == 255, axis=(0, 1))
    
        ## if we have enough clear images (without ANY missing values), we're good to go, and we will use all of them (could be more than 3!)
        if len(np.where(sum_invalid == 0)[0]) > 3:
            allgood = 1
            self.log.debug((f'Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> good to go'))
            ndvi_stack = ndvi_stack[:, :, np.where(sum_invalid == 0)[0]]

        ## else we need to add some images that do contain some nan's; in this case we will select just the 3 best ones
        else:
            self.log.debug((f'Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> appending some bad images as well!'))
            allgood = 0
            idxsorted = np.argsort(sum_invalid)
            ndvi_stack = ndvi_stack[:, :, idxsorted[0:4]]
    
        ## fill the NaN values with 0
        ndvi_stack[ndvi_stack == 255] = 0
    
        ## convert to fractional number
        ndvi_stack = ndvi_stack / 250.
    
        ## check whether you actually supplied 3 images or more
        nrValidBands = ndvi_stack.shape[2]

        ## if the stack doesn't have at least 3 bands, we cannot process this window
        if nrValidBands < 3:
            self.log.warning('Not enough input data for this window -> skipping!')
            clear_session()
            gc.collect()
            return None
        ## else, let's do inference!
        else:
            ## we'll do 12 predictions: use 3 networks, and for each randomly take 3 NDVI bands and repeat 4 times
            prediction = np.zeros((patch_size, patch_size, 12))
            for i in range(4):
                prediction[:, :, i] = np.squeeze(
                    model1.predict(ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                                   .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))

            for i in range(4):
                prediction[:, :, i + 4] = np.squeeze(
                    model2.predict(
                        ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                        .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))

            for i in range(4):
                prediction[:, :, i + 8] = np.squeeze(
                    model3.predict(
                        ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                        .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))

            clear_session() # to avoid memory leakage executors
            gc.collect()

            ## final prediction is the median of all predictions per pixel
            final_prediction = np.median(prediction, axis = 2)

            return (final_prediction, allgood)
    


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:

    modeldir='/data/users/Public/driesseb/fielddelineation'

    ## if you would specify a context, you can use the method "get" to retrieve the value
    # if context is not None:
    #     startdate = context.get('startdate', "2019-01-01")

    ## get the array and transpose
    cubearray:xarray.DataArray = cube.get_array()
    cubearray = cubearray.transpose('x','bands','y','t')

    ## get nan indices
    nan_locations_NDVI = ufuncs_isnan(cubearray)

    ## clamp out of range NDVI values
    cubearray = cubearray.where(cubearray < 0.92, 0.92)
    cubearray = cubearray.where(cubearray > -0.08,np.nan)

    ## xarray nan to np nan (we will pass a numpy array)
    cubearray = cubearray.where(~nan_locations_NDVI,np.nan)

    ## rescale data and just take NDVI to get rid of the band dimension
    cubearray = ((cubearray +0.08)*250.)[:,0,:,:]

    ## transpose to format accepted by model
    inputarray = cubearray.transpose('x','y','t')

    ## create a mask where all valid values are 1 and all nans are 0
    inputmask = ones_like(inputarray)
    inputmask = inputmask.where(ufuncs_isnan(inputarray), 0.)

    ## wrap the input array and the mask in a dictionary to pass to the processWindow method of the Segmentation class
    inputdata={
        's2_ndvi': inputarray,
        's2_mask': inputmask
    }

    ## load in the pretrained U-Net keras models and do inference!
    s=Segmentation()
    models = load_models(modeldir)
    (result, allgood) = s.processWindow(models, inputdata)

    ## transform your numpy array predictions into an xarray
    result=result.astype(np.float64)
    result_xarray = xarray.DataArray(result,dims=['x','y'],coords=dict(x=cubearray.coords['x'],y=cubearray.coords['y']))
    result_xarray=result_xarray.expand_dims('t',0).assign_coords(t=[np.datetime64(str(cubearray.t.dt.year.values[0])+'-01-01')])

    #openEO assumes a fixed data order, so transpose your result and store it in a XarrayDataCube that you return
    result_xarray = result_xarray.transpose('t','y','x')
    return XarrayDataCube(result_xarray)
