"""This is a script to run the model of TurbidityCurrent2D
"""
import os
import csv
import shutil
# os.environ['MKL_NUM_THREADS'] = '24'
# os.environ['OMP_NUM_THREADS'] = '24'
import numpy as np
from turb2d.utils import create_topography, cordiate_to_node, create_folder
from turb2d.utils import create_init_flow_region, create_topography_from_geotiff
from turb2d.utils import load_inlet, interpolate_area
from turb2d import TurbidityCurrent2D
import time
# from landlab import FIXED_GRADIENT_BOUNDARY, FIXED_VALUE_BOUNDARY
import pdb
from tqdm import tqdm
import random

#----------------------MAP SETTING-----------------------------------------------------------
root_path = os.getcwd()
xlim = [-57.5, -50.5]
ylim = [37.0, 45.5]
spacing = 1000 #m
grid_degree = [0.0121603, 0.0086239] #x, y

#QGIS WGS84 to UTM(NAD27 / UTM zone 21N) to WGS84
grid = create_topography_from_geotiff(os.path.join(root_path,"bathymetry/GB04_WGS_1000.tif"),
                                       xlim=xlim, ylim=ylim, spacing=spacing, filter_size=[10, 10],
                                       distribution_filename="bathymetry/GB_sand01.tif",
                                       setting_gs = [0.4, 0.4, 0.2] # if None is same ratio in all grain size
                                       )
#-------------------------------------------------------------------------------------------

#----------------------BASE SETTING-----------------------------------------------------------
path = "/mnt/d/turb2d" #"/mnt/f/Turb2d/Test"
obs_csv = 'obs_csv/obs_list.csv'
dirname = 'test01'#'YC086'#'GB222' #"test2024-04" #
last = 3*24*60*60 #8640 #100000
itsnap = 1200
random_sw = False #True #
ini_type = "bathymetry/GB_vfill_2.tif" #"rectangle" #"circle" #"inlet" #
Ds = [1.8e-4, 6.4e-5, 3.0e-5]
#-------------------------------------------------------------------------------------------

#---------------------------TIFF TYPE-------------------------------
height_factor = 0.1 #flow depth provided from tif * height_factor

#---------------------Rectangle or Circle TYPE----------------------
#BOTH
initial_cordiate = [44.5419, -55.9347] #lat, lon
initial_flow_concentration=[0.00, 0.03, 0.03] #0.01, #[0.01,0.01], #[0.001,0.001,0.001],
initial_flow_thickness = 20           #[m]
velocity = 50 #m/s

#Rectangle
azimuth = 160 #180 - 20
width = 3 *1000
height = 20 *1000

#Circle
initial_region_radius = 1000 #半径[m]

#--------------------------Inlet TYPE------------------------------
# Define inlet renge based water depth
inlet_depth = 450 #  [m] In case of using "inlet_lat", this value is None

# Define inlet renge based on baundary by lat
inlet_lat = [38.643, 38.982] #[37.5, 39.0] #
inlet_width = 10 # grid
depth_scale = -1.0 # Must 0-1 for using water depth (water depth * depth_scale)
initial_flow_height = 20 #depth_scale # float or int types for constant flow depth
C_time_series_flag = False # #False #True
V_time_series_flag = False # #False #True
input_dir = 'input_csv' # C0.csv velocity.csv
# If False
inlet_end = 10800 #last-1 #  For constant
constant_c = [1e-3, 1e-3, 1e-3]
constant_v = 0.5 # positive = Eastward

#------------------------------------------------------------------

grid.status_at_node[grid.nodes_at_top_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
grid.status_at_node[grid.nodes_at_bottom_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
grid.status_at_node[grid.nodes_at_left_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_FIXED_GRADIENT

dirpath = os.path.join(path, dirname)
dirname = create_folder(dirpath)

shutil.copy(__file__, dirpath)
shutil.copy(os.path.join(root_path,'config.yml'), dirpath)
shutil.copy(os.path.join(root_path,'turb2d/sediment_func.py'), dirpath)

if ini_type == "inlet":
    x_size = grid.x_of_node.max() / spacing
    y_size = grid.y_of_node.max() / spacing
    # print("x_of_node:", x_size)
    # print("y_of_node:", y_size)

    if inlet_depth == None:
        inlet_S = int((inlet_lat[0] - ylim[0]) / grid_degree[1])
        inlet_N = int((inlet_lat[1] - ylim[0]) / grid_degree[1])

        inlet = np.where((grid.x_of_node > 0)
                            & (grid.x_of_node < inlet_width * spacing)
                            & (grid.y_of_node > inlet_S * spacing)
                            & (grid.y_of_node < inlet_N * spacing))

    else:
        inlet = np.where(grid.at_node["topographic__elevation"] >= -inlet_depth)

    grid.status_at_node[inlet] = grid.BC_NODE_IS_FIXED_VALUE

    if initial_flow_height == depth_scale:
        grid.at_node['flow__depth'][inlet] = -1 * grid.at_node["topographic__elevation"
                                                               ][inlet] * depth_scale
        print(grid.at_node['flow__depth'][inlet])
    else:
        grid.at_node['flow__depth'][inlet] = initial_flow_height

    if V_time_series_flag == True:
        Vc = load_inlet('input_csv', "velocity", inlet_lat)
        v_value = interpolate_area(Vc, inlet, 0)
        inlet_end = len(Vc.iloc[0].values)-1
    else:
        v_value = constant_v

    c_data = {}
    for d in np.arange(len(Ds)):
        grid.add_zeros("flow__sediment_concentration_{}".format(d), at="node")
        if C_time_series_flag == True:
            identifier = f"C{d}"
            c_data[identifier] = load_inlet(input_dir, identifier, inlet_lat)
            # print(c_data[identifier])
            c_value = interpolate_area(c_data[identifier], inlet, 0)

        else:
            c_value = constant_c[d]
        # c_data[identifier].to_csv("testC2.csv")
        grid.at_node[f"flow__sediment_concentration_{d}"][inlet] = c_value
    grid.at_node['flow__horizontal_velocity_at_node'][inlet] = v_value
    grid.at_node['flow__vertical_velocity_at_node'][inlet] = 0
    print("inlet end time [s]", inlet_end)

else:
    initial_region_center = cordiate_to_node(initial_cordiate, xlim, ylim, grid_degree, spacing)
    create_init_flow_region(
        grid,
        initial_flow_concentration=initial_flow_concentration,       #0.01, #[0.01,0.01], #[0.001,0.001,0.001],
        initial_flow_thickness=initial_flow_thickness,                   #[m]
        initial_region_radius=initial_region_radius,                  #半径[m]
        initial_region_center=initial_region_center,  # 1000, 4000 # spacing*グリッド　0,0はxlimとylimの範囲から
        initial_region_shape = ini_type,
        width = width,
        height = height,
        azimuth = azimuth,
        velocity = velocity, #m/s
        flow_azimuth = 30+azimuth, #180-azimuth,
        xlim=xlim, ylim=ylim,
        height_factor=height_factor
    )

obs_list = []
with open(os.path.join(root_path, obs_csv),
           'r', encoding = "utf-8-sig") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        obs_list.append(row)

######################################### ONE CASE ##########################################
# if random_sw == False:
tc = TurbidityCurrent2D(grid,
    Ds = Ds,
    config_path = os.path.join(root_path, dirpath,'config.yml')
    )

print("START", dirname)
# start calculation
t = time.time()
tc.save_nc('{}/{}_{:04d}.nc'.format(dirpath, dirname, 0))
Ch_init = np.sum(tc.C * tc.h)
dt=1.0
num = 1
for i in tqdm(range(1, last + 1), disable=False):
    if ini_type == "inlet":
        boundary_reverted = False
        if i > inlet_end and not boundary_reverted:
            # ノードとリンクを計算ノードに戻す（1回のみ）

            grid.at_node['flow__depth'][inlet] = 0
            grid.at_node['flow__horizontal_velocity_at_node'][inlet] = 0
            # grid.at_node['flow__vertical_velocity_at_node'][inlet] = 0
            for d in np.arange(len(Ds)):
                grid.at_node[f"flow__sediment_concentration_{d}"][inlet] = 0

            grid.status_at_node[inlet] = grid.BC_NODE_IS_CORE  # 計算ノードに戻す
            boundary_reverted = True

        elif not boundary_reverted:
            grid.status_at_node[inlet] = grid.BC_NODE_IS_FIXED_VALUE
            if V_time_series_flag == True:
                grid.at_node['flow__horizontal_velocity_at_node'
                            ][inlet] = interpolate_area(Vc, inlet, i)
                # grid.at_node['flow__vertical_velocity_at_node'][inlet] = 0
            if C_time_series_flag == True:
                for d in np.arange(len(Ds)):
                    identifier = f"C{d}"
                    grid.at_node[f"flow__sediment_concentration_{d}"
                                ][inlet] = interpolate_area(c_data[identifier], inlet, i)

    tc.run_one_step(dt=dt)

    #Save point
    for j in range(len(obs_list)):
        obs_cordiate = cordiate_to_node([float(obs_list[j][1]), float(obs_list[j][2])], xlim, ylim, grid_degree ,spacing)
        tc.save_point_data('{}/{}_{}.csv'.format(dirpath, dirname, obs_list[j][0]), obs_cordiate, i)

    #Save nc file
    if i % itsnap == 0 and i != 0:
        tc.save_nc('{}/{}_{:04d}.nc'.format(dirpath, dirname, num))
        # tc.save_grid('{}/es{:04d}'.format(dirpath, num))
    if ini_type != "inlet":
        if np.sum(tc.C * tc.h) / Ch_init < 0.01:
        # if np.sum((tc.C_i[0, :] + tc.C_i[1, :]) * tc.h) / Ch_init < 0.01:
            break

    num = num + 1
tc.save_nc('{}/{}_{:04d}.nc'.format(dirpath, dirname, num), xlim, ylim)
tc.save_grid('{}/tc{:04d}'.format(dirpath, num))
print('elapsed time: {} sec.'.format(time.time() - t))

######################################### MULTI CASE ##########################################
# else:
#     counts = 1
#     while counts <= 10:
#         case_name = '{}_{:03d}'.format(dirname, counts)
#         Ds_num = 2
#         r_thickness = random.randrange(10, 500, 10)
#         if ini_type == "rectangle":
#             r_width = random.randrange(10000, 500000, 100)
#             r_height = random.randrange(1000, 50000, 100)
#         else:
#             r_width = random.randrange(10000, 500000, 100)
#             r_height = "NaN"
#         r_Ds = [round(np.power(2,-random.uniform(0, 5)), 5)/1000 for i in range(Ds_num)] #range based on phi scale
#         r_concentration = [round(random.uniform(0.001, 0.100), 3) for i in range(Ds_num)]

#         print(case_name)
#         casepath = os.path.join(dirpath, case_name)
#         if not os.path.exists(casepath):
#             os.mkdir(casepath)

#         with open(os.path.join(dirpath, dirname+'_param.csv'), 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow(([case_name, *r_concentration, r_thickness, r_width, r_height, *r_Ds]))

#         create_init_flow_region(
#             grid,
#             initial_flow_concentration=r_concentration,
#             initial_flow_thickness=r_thickness,
#             initial_region_radius=r_width,
#             initial_region_center=initial_region_center,
#             initial_region_shape = ini_type,
#             width = r_width,
#             height= r_height,
#             azimuth=azimuth,
#             xlim=xlim, ylim=ylim,
#             height_factor=height_factor
#         )

#         tc = TurbidityCurrent2D(grid,
#             Ds = r_Ds,
#             config_path = os.path.join(root_path,'config.yml'))

#         # start calculation
#         t = time.time()
#         tc.save_nc('{}/{}_{:04d}.nc'.format(casepath, case_name, 0), xlim, ylim)
#         Ch_init = np.sum(tc.C * tc.h)

#         num = 1
#         for i in tqdm(range(1, last + 1), disable=False):
#             tc.run_one_step(dt=1.0)
#             # tc.save_point_data('{}/{}_{:04d}.csv'.format(casepath, case_name), xlim, ylim)
#             if i % itsnap == 0 and i != 0:
#                 tc.save_nc('{}/{}_{:04d}.nc'.format(casepath, case_name, num), xlim, ylim)
#             if np.sum(tc.C * tc.h) / Ch_init < 0.01:
#                 break
#             num = num + 1
#         tc.save_nc('{}/{}_{:04d}.nc'.format(casepath, case_name, num), xlim, ylim)
#         tc.save_grid('{}/tc{:04d}.nc'.format(casepath, num))
#         print('elapsed time: {} sec.'.format(time.time() - t))

#         counts += 1
##################################################################################################
