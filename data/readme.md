Datasets: Folder Structure and Preparation
------------------------------------------

In the following, we present the folder structure required to use the different datasets with **Hy²DL**.

### CAMELS_DE

The CAMELS_DE dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.16755906).  
To add this dataset to the library, use the following folder structure:

```
CAMELS_DE/
  timeseries/ 
  CAMELS_DE_climatic_attributes.csv
  CAMELS_DE_humaninfluence_attributes.csv
  ...
  CAMELS_DE_xxx_attributes.csv
```

### CAMELS_GB

The CAMELS_GB dataset can be downloaded from [Zenodo](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9).
To add this dataset to the library, use the following folder structure:

```
CAMELS_GB/
  timeseries/
  CAMELS_GB_climatic_attributes.csv
  CAMELS_GB_humaninfluence_attributes.csv
  ...
  CAMELS_GB_xxx_attributes.csv
```

### CAMELS_US

The CAMELS_US dataset can be downloaded from [Zenodo](https://doi.org/10.5065/D6MW2F4D).
To add this dataset to the library, use the following folder structure:

```
CAMELS_US/
  basin_mean_forcing/
    daymet/
    maurer/
    nldas/ 
  camels_attributes_v2.0/
    camels_clim.txt
    camels_geol.txt
    camels_xxx.txt 
  usgs_streamflow/
```

### CAMELS_US (hourly resolution)

Hourly products from CAMELS_US can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.4072701).
To add this dataset, create an additional `hourly/` folder inside the CAMELS_US structure:

```
CAMELS_US/
  hourly/               
    nldas_hourly/
    usgs-streamflow/
```

### CARAVAN

The CARAVAN dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.10968468).
Support is provided only for the CSV files. To add this dataset to the library, use the following folder structure:

```
Caravan/
  attributes/
    camels/
    camelsaus/
    camelsbr/
    camelscl/
    camelsgb/
    hysets/
    lamah/
  code/
  licenses/
  shapefiles/
    camels/
    camelsaus/
    camelsbr/
    camelscl/
    camelsgb/
    hysets/
    lamah/
  timeseries/
    camels/
    camelsaus/
    camelsbr/
    camelscl/
    camelsgb/
    hysets/
    lamah/
```

For using **community extensions** from CARAVAN (`GitHub discussion <https://github.com/kratzert/Caravan/discussions/10>`_), the required extension dataset should be manually downloaded and added to the corresponding folders in the structure above.
