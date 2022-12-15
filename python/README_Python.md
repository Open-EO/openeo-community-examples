# OpenEO Python Client - Examples and Code Snippets

This repository contains Jupyter Notebooks and process graphs showcasing on usecase of OpenEO features and EOplaza services. 

# Contributing

OpenEO/EOplaza users can also send a pull request in this repository as either a python file/notebook for their usecase. Showcasing their usecases are highly encouraged. 


# Using/Running the Notebooks
You can run the Notebooks locally but you need to take following things into consideration:

* Install OpenEO

    ``` !pip install OpenEO ```

* Authentication

    You can either use authenticate_oidc() as a simple practice with different backends or use your username and password=username123 for basic authentication, i.e., authenticate_basic(username=skywalker, password=skywalker123).

* Path to GeoJson

    Make sure that the aoi path is correctly assigned.


# Contents
This repository includes two folder:

1. process_graphs
    
    This folder includes the process graphs i.e chain of specific processes as json file. Each file is represent notebook with similar name in usecase_notebooks. 

2. usecase_notebook

    It includes several examples showcasing the use of OpenEO features and Eoplaza services as a jupyter notebook.
    
    * biomass_basic

        Executes simple process like biomass already available in EOplaza. Service available [here](https://portal.terrascope.be/catalogue/app-details/17)

    * burntmapping_chunks

        Classical Normalized Burnt Ratio(NBR) difference performed using VITO backend on a chunk polygon. You can find ways to develop your process and use chunk_polygon on a usecase. The method followed in this notebook to compute DNBR is inspired from [UN SPIDER's](https://github.com/UN-SPIDER/burn-severity-mapping-EO) recommended preactices.

    * flood_ndwi

        Comparative study between pre and post image for Colong during 2021 flood. A simple technique to subtract pre and post image is done to know the change in water content due to flood in that region. Refernce: https://labo.obs-mip.fr/multitemp/the-ndwi-applied-to-the-recent-flooding-in-the-central-us/

    * flood_sar_udf

        Flood extent can be determined using a change detection approach on Sentinel-1 data. In this notebook we have tried in adopting (UN SPIDER's)[https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping] recommendation practice for computing flood extent by implementing openeo process.

    * ndwi_basic

        Executes simple process like ndwi already available in EOplaza. Service available [here](https://portal.terrascope.be/catalogue/app-details/13).
    * rescale_chunks 

        The creation of a simple process to rescale Sentinel 2 RGB image along with the use of chunk_polygon  apply with a (User Defined Function) UDF.