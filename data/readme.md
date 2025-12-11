Datasets: Folder Structure and Preparation
------------------------------------------

In the following, we present the folder structure required to use the different datasets with **Hy²DL**.

### CAMELS_CH

The CAMELS_CH dataset can be downloaded from [Zenodo](https://zenodo.org/records/15025258). 
To add this dataset to the library, use the following folder structure:

```
CAMELS_CH/
  static_attributes/
  


```

### CAMELS_DE

The CAMELS_DE dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.16755906).
To add this dataset to the library, use the following folder structure:

```
CAMELS_DE/
  timeseries/ 
  CAMELS_CH_climate_attributes_obs.csv
  CAMELS_CH_geology_attributes.csv
  ...
  timeseries/
    observation_based/
    CAMELS_CH_obs_based_2004.csv
    CAMELS_CH_obs_based_2007.csv
    ...
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

### CARAVAN — community extensions

Community-contributed Caravan extensions (e.g., camelsde, grdc) can be added manually:

1. Locate the extension and its Zenodo download link on the Caravan GitHub Discussions page: https://github.com/kratzert/Caravan/discussions/10  
2. Download the dataset archive from the linked Zenodo record.  
3. Unzip the archive and copy its contents into the corresponding Caravan subfolders (for example: attributes/, timeseries/, shapefiles/). Preserve the folder layout provided by the original Caravan dataset.  
4. Ensure CSV files are placed in the CSV-supported folders so Hy2DL can read them.  
5. Repeat the same process to add your own dataset (if any) created using the Caravan code.

Tip: verify the final folder names match the structure described above so Hy2DL can detect and load the data.
