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
## Revision by Ryo Nakanishi 2025.03.01

#