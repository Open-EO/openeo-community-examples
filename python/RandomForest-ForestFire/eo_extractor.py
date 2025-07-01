import openeo
from openeo import Connection
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import array_create



def s1_features(connection: Connection, date, aoi, reducer):

    """
    Preprocess Sentinel-1 SAR data by applying backscatter correction,
    computing VH/VV ratio and log transformations, then reducing over time.

    Args:
        connection (Connection): An openEO connection.
        date (Tuple[str, str]): Temporal extent as (start_date, end_date)
        aoi (dict): Spatial extent 
        reducer (Any): Reducer (first, last, mean, median)

    Returns:
        DataCube: The processed and temporally reduced Sentinel-1 data cube.
    """
    # load S2 pre-collection
    s1_cube = connection.load_collection(
        "SENTINEL1_GRD",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["VH", "VV"]
    )
    
    # apply SAR backscatter processing to the collection
    s1_cube = s1_cube.sar_backscatter(coefficient="sigma0-ellipsoid")

    # apply band-wise processing to create a ratio and log-transformed bands
    s1_cube = s1_cube.apply_dimension(
        dimension="bands",
        process=lambda x: array_create(
            [
                30.0 * x[0] / x[1],  # Ratio of VH to VV
                30.0 + 10.0 * x[0].log(base=10),  # Log-transformed VH
                30.0 + 10.0 * x[1].log(base=10),  # Log-transformed VV
            ]
        ),
    )
    
    s1_cube = s1_cube.rename_labels("bands", ["ratio"] + s1_cube.metadata.band_names)

    # scale to int16
    s1_cube = s1_cube.linear_scale_range(0, 30, 0, 30000)

    return s1_cube.reduce_dimension(reducer=reducer, dimension="t")

def s2_features(connection: Connection, date, aoi, reducer):
    """
   Preprocess Sentinel-2 data by loading relevant bands, applying scaling,
   and reducing over time using a specified reducer.

    Args:
        connection (Connection): An openEO connection.
        date: Temporal extent as (start_date, end_date)
        aoi: Spatial extent 
        reducer (Any): Reducer (first, last, mean, median)

    Returns:
        DataCube: The processed and temporally reduced Sentinel-2 datacube.
    """
    # load S2 pre-collection
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["B02", "B03", "B04", "B08","B12"],
        max_cloud_cover=80,
    )
    
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=date,
        spatial_extent=aoi,
        bands=["SCL"],
        max_cloud_cover=80,
    )
    
    mask = scl.process(
        "to_scl_dilation_mask", 
        data=scl,
        kernel1_size=17,
        kernel2_size=77,
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3
    )
    
    # Create a cloud-free mosaic
    masked_cube = s2_cube.mask(mask)
    cf_cube = masked_cube.reduce_dimension(reducer=reducer, dimension="t")

    # calculate all indices
    indices_list = ["NBR", "BAI"]
    indices = compute_indices(cf_cube, indices_list)

    # calculate texture features
    features_udf = openeo.UDF.from_file("features.py")
    features = cf_cube.apply_neighborhood(
        process=features_udf,
        size=[
            {"dimension": "x", "value": 128, "unit": "px"},
            {"dimension": "y", "value": 128, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 32, "unit": "px"},
            {"dimension": "y", "value": 32, "unit": "px"},
        ],
    )
    
    # combine the original bands with the computed indices,
    merged_cube = cf_cube.merge_cubes(indices)
    merged_cube = merged_cube.merge_cubes(features)
    return merged_cube