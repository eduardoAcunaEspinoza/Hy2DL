In the following, we present a scheme of the folder structure required to use the different datasets.

##### CAMELS_DE:
The CAMELS_DE dataset can be downloaded from https://doi.org/10.5281/zenodo.13837553. To add this dataset to the library, the following folder structure should be used:
```
CAMELS_DE/
  timeseries/ 
  CAMELS_DE_climatic_attributes.csv
  CAMELS_DE_humaninfluence_attributes.csv
  ...
  CAMELS_DE_xxx_attributes.csv
```
##### CAMELS_GB:
To use CAMELS_GB can be downloaded from https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9. To add this dataset to the library, the following folder structure should be used:

```
CAMELS_GB/
  timeseries/
  CAMELS_GB_climatic_attributes.csv
  CAMELS_GB_humaninfluence_attributes.csv
  ...
  CAMELS_GB_xxx_attributes.csv
```
#####  CAMELS_US:
To use CAMELS_US can be downloaded from https://gdex.ucar.edu/dataset/camels.html. To add this dataset to the library, the following folder structure should be used:

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
#####  CAMELS_US hourly resolution:
Hourly products from the CAMELS_US dataset can be downloaded from https://doi.org/10.5281/zenodo.4072701. To add this dataset to the library, an additional folder (hourly/) should be created inside the CAMELS_US structure. 
```
CAMELS_US/
  hourly/               
    nldas_hourly/
    usgs-streamflow/
```