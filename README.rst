TURB2D
========================

This is a code for the simulation of turbidity currents on 2D grids.

---------------
Installation
Step 0. Since this program uses Cython, you need to install some C compiler before installing turb2d. For Windows, you can download Microsoft Visual Studio Community (https://visualstudio.microsoft.com/free-developer-offers/), which provides a free environment for compiling C code.


Step 0. Install python3.11. The turb2d may not work under other version of python.

Step 1. Execute the following command in the downloaded turb2d directory (the directory where this file is located).

> pip install .

You may be requested to install pip before the installation of turb2d. Note that the package management system of conda may be collapsed by pip.

Step 2. If the installation of turb2d is successful, you can run run_turb2d_script.py to try out the calculations.

> python run_turb2d_script.py

Calculation results will be output to tc*.nc files in the NetCDF format, which can be read by visualization software such as Paraview.

---------------
# Revision by Ryo Nakanishi 2025.03.01
@ turb2d.py utilis.py sediment_func.py run_turb2d_script.py pyproject.toml

## Change of initial ratio distribution of active layers and specification of initial flow area by tiff file
*distribution_filename* specifies the ratio of active layer for each grain size *setting_gs*. See *GB_sand01.tif* for use of tiff file.
Note that outside the specified range of tiff file, the last grain size scale element in the Ds array is 1.0 (assuming finest size).

To specify the initial flow range and height by means of a tiff file, specify the file path in *ini_type*.
The *height_factor* can be used to change the magnification factor of the tiff values.

All tiff files, including topographic data, must have the same x and y (longitude and latitude) spacing at meter scale.
Enter the mesh size in WGS84 into *grid_degree*.

## The initial shape of the turbidity flow can be selected.
In addition to the conventional cylindrical collapse, rectangular collapse and boundary inlet can be selected.
> ini_type = "circle" or "rectangle" or "inlet"

Initial flow velocity and azimuth can be set.
Note that there may be exceptions or mistakes in the azimuth setting.

The **inlet** is set by a simple boundary condition based on latitude and longitude or the depth of the seafloor(*inlet_depth*).
Only the y-direction (latitude) is implemented. *inlet_lat* is used to select the range of initial flow conditions, and *inlet_width* is used to set the number of grids.
There are two methods to give initial conditions: one is to give a constant concentration and velocity over a certain duration (*constant_c* and *constant_v*), and the other is to give a csv file discrived time series variation such as *C0.csv* or *velocity.csv* formats.
Time series variation requres two flags (*C_time_series_flag* and *V_time_series_flag*), and gives different conditions by specifying a folder *input_dir*.
Note that this method has not been fully tested.

## Added the ability to save time-series data in csv at points set in the coordinate system.
Specify a csv file with the coordinates you wish to record in obs_csv.
For the format of the csv file, refer to the file in the obs_csv folder.

## Several sediment entrainment formulas were implemented.
Traer et al. 2012 and NRv2 (my experimental)
Calculate D50 and sorting from the active layers
Camax, p, df, ef can be configured in config.yml.

## Output maximum erosion depth as nc file
By considering the maximum erosion thickness, the final turbidite layer thickness can now be calculated.

## Visualization support option **turb2d_nc.py**
Optionally, a script (running separately from turb2d) is provided to calculate and chart the turbidite layer thickness (or erosion depth) for each grain size class.
This script can be overridden by the observed layer thickness added *obs_csv* to facilitate comparisons or to create a YY plot of calculated vs. observed thicknesses.
