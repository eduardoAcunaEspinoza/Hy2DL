The required folder structure, to use the different datasets, is equivalent to the one used in Neural Hydrology.

In the following, we present a quick scheme of such structure.

---------------
CAMELS_DE:
To use CAMELS_DE the following data structure should be used:

CAMELS_DE/
- timeseries/
- CAMELS_DE_climatic_attributes.csv
- CAMELS_DE_humaninfluence_attributes.csv
...
- CAMELS_DE_xxx_attributes.csv
---------------

CAMELS_GB:
To use CAMELS_GB the following data structure should be used:

CAMELS_GB/
- timeseries/
- CAMELS_GB_climatic_attributes.csv
- CAMELS_GB_humaninfluence_attributes.csv
...
- CAMELS_GB_xxx_attributes.csv
---------------

CAMELS_US:
To use CAMELS_US the following data structure should be used.

CAMELS_US/
- basin_mean_forcing/
	-daymet/
	-maurer/
	-nldas/ 

- camels_attributes_v2.0/
	-camels_clim.txt
	-camels_geol.txt
	-camels_xxx.txt 

- usgs_streamflow/

Note: The daymet, maurer and nldas are different forcing products. Not all of them have to be used. Moreover, inside the daymet, maurer, ndlas and usg_streamflow folders, one can directly paste the 18 folders associated with the 18 hydrological units of camels_us. Inside each of the 18 folders, there are "n" timeseries. 
---------------

CAMELS_US hourly resolution:
To use hourly products, an additional folder (hourly/) should be created inside the CAMELS_US structure. 

CAMELS_US/
- basin_mean_forcing/
	-daymet/
	-maurer/
	-nldas/ 

- camels_attributes_v2.0/
	-camels_clim.txt
	-camels_geol.txt
	-camels_xxx.txt 

- usgs_streamflow/

- hourly/               
	-nldas_hourly/     # hourly forcing data
	-usgs-streamflow/  # hourly streamflow data
---------------	