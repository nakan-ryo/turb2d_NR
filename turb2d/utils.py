"""A module for TurbidityCurrent2D to produce a grid object from a geotiff
   file or from scratch.

   codeauthor: : Hajime Naruse
"""

from landlab import RasterModelGrid
import numpy as np
from scipy.ndimage import median_filter, zoom
from landlab import FieldError
import rasterio
import os
import yaml
import pandas as pd
import shutil
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def create_topography_from_geotiff(
    bathymetry_filename, xlim=None, ylim=None, spacing=None, filter_size=[1, 1],
    distribution_filename=None, setting_gs = [1.0, 0.4, 0.2]
):
    """create a landlab grid file from a geotiff file

    Parameters
    -----------------------
    geotiff_filename: String
       Name of a geotiff-format file to import.
       DEM coordinates must be in a projected coordinate system (e.g. UTM).

    xlim: list, optional
       list [xmin, xmax] to specify x coordinates of a region of interest
          in a geotiff file to import

    ylim: list, optional
       list [ymin, ymax] to specify y coordinates of a region of interest
          in a geotiff file to import

    spacing: float, optional
       grid spacing.
       Normally, the grid interval is automatically read from the geotif file,
       so there is no need to specify this parameter. However, if you do
       specify it, an interpolation process will be carried out to convert
       the DEM data to match the specified value. This process can take a long
       time.

    filter_size: list, optional
       [x, y] size of a window used in a median filter.
         This filter is applied for smoothing DEM data.

    Return
    ------------------------
    grid: RasterModelGrid
       a landlab grid object to be used in TurbidityCurrent2D

    """
    def geotiff2grid(geotiff_filename):
    # read a geotiff file into ndarray
        with rasterio.open(geotiff_filename) as src:
            grid_data = src.read(1)[::-1, :]
            transform = src.transform
            dx = transform[0]
            dy = -transform[4]
            # print("dxdy", dx, dy)

        # print(topo_data.shape)
        if (xlim is not None) and (ylim is not None):
            # GeoTIFFの左上からの相対距離を求める（緯度経度差）
            col_start = int((xlim[0] - src.bounds.left) / dx)
            col_end = int((xlim[1] - src.bounds.left) / dx)
            row_start = int((ylim[0] - src.bounds.bottom) / dy)
            row_end = int((ylim[1] - src.bounds.bottom) / dy)

            # インデックスを整数に変換してスライス
            col_start, col_end = int(np.floor(col_start)), int(np.ceil(col_end))
            row_start, row_end = int(np.floor(row_start)), int(np.ceil(row_end))

            # スライスして範囲指定
            grid_data = grid_data[row_start:row_end, col_start:col_end]

        # Smoothing by median filter
        grid_data = median_filter(grid_data, size=filter_size)
        # print("y",row_start,row_end, "x",col_start,col_end)
        return grid_data

    topo_data = geotiff2grid(bathymetry_filename)
    # # change grid size if the parameter spacing is specified
    # if spacing is not None and spacing != dx:
    #     zoom_factor = dx / spacing
    #     topo_data = zoom(topo_data, zoom_factor)
    #     dx = spacing

    # grid = RasterModelGrid(
    #     topo_data.shape, xy_spacing=[dx, dx], xy_of_lower_left=xy_of_lower_left
    # )
    grid = RasterModelGrid(topo_data.shape, xy_spacing=[spacing, spacing])
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")
    # grid.add_zeros("max__erosion", at="node")
    grid.at_node["topographic__elevation"][grid.nodes] = topo_data
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    grid.add_zeros("flow__sediment_concentration_total", at="node")

    if distribution_filename:
        dist_data = geotiff2grid(distribution_filename)
        inverted_dist_data= 1 - dist_data

        for i in range(len(setting_gs)):
            if i == len(setting_gs)-1:
                grid.at_node["bed__active_layer_fraction_{}".format(i)
                             ] = dist_data *setting_gs[i] + inverted_dist_data
            else:
                grid.at_node["bed__active_layer_fraction_{}".format(i)
                             ] = dist_data *setting_gs[i]
    elif setting_gs != None:
        for i in range(len(setting_gs)):
            grid.at_node["bed__active_layer_fraction_{}".format(i)
                            ] = topo_data * 0 + setting_gs[i]

            # grid.add_ones(
            #     "bed__active_layer_fraction_" + str(i),
            #     at="node",
            # ) * setting_gs[i]
    return grid

def create_topography(
    config_file=None,
    length=8000,
    width=2000,
    spacing=20,
    slope_outside=0.1,
    slope_inside=0.05,
    slope_basin=0.02,
    slope_basin_break=2000,
    canyon_basin_break=2200,
    canyon_center=1000,
    canyon_half_width=100,
    canyon="parabola",
    noise=0.01,
):
    """create an artificial topography where a turbidity current flow down
       A slope and a flat basin plain are set in calculation domain, and a
       parabola or v-shaped canyon is created in the slope.

       Parameters
       ------------------
        length: float, optional
           length of calculation domain [m]

        width: float, optional
           width of calculation domain [m]

        spacing: float, optional
           grid spacing [m]

        slope_outside: float, optional
           topographic inclination in the region outside the canyon

        slope_inside: float, optional
           topographic inclination in the region inside the thalweg of
           the canyon

        slope_basin: float, optional
           topographic inclination of the basin plain

        slope_basin_break: float, optional
           location of slope-basin break

        canyon_basin_break: float, optional
           location of canyon-basin break. This value must be
           larger than slope-basin break.

        canyon_center: float, optional
           location of center of the canyon

        canyon_half_width: float, optional
           half width of the canyon

        canyon: String, optional
           Style of the canyon. 'parabola' or 'V' can be chosen.

        random: float, optional
           Range of random noise to be added on generated topography

        Return
        -------------------------
        grid: RasterModelGrid
           a landlab grid object. Topographic elevation is stored as
           grid.at_node['topographic__elevation']
    """

    if os.path.exists(config_file):
        with open(config_file) as yml:
            config = yaml.safe_load(yml)
        length=config['grid']['length']
        width=config['grid']['width']
        spacing=config['grid']['spacing']
        slope_outside=config['grid']['slope_outside']
        slope_inside=config['grid']['slope_inside']
        slope_basin=config['grid']['slope_basin']
        slope_basin_break=config['grid']['slope_basin_break']
        canyon_basin_break=config['grid']['canyon_basin_break']
        canyon_center=config['grid']['canyon_center']
        canyon_half_width=config['grid']['canyon_half_width']
        canyon=config['grid']['canyon']
        noise=config['grid']['noise']
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m

    lgrids = int(length / spacing)
    wgrids = int(width / spacing)
    grid = RasterModelGrid((lgrids, wgrids), xy_spacing=[spacing, spacing])
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")
    grid.add_zeros("max__erosion", at="node")

    # making topography
    # set the slope
    grid.at_node["topographic__elevation"] = (
        grid.node_y - slope_basin_break
    ) * slope_outside

    if canyon == "parabola":
        # set canyon
        d0 = slope_inside * (canyon_basin_break - slope_basin_break)
        d = slope_inside * (grid.node_y - canyon_basin_break) - d0
        a = d0 / canyon_half_width**2
        canyon_elev = a * (grid.node_x - canyon_center) ** 2 + d
        inside = np.where(canyon_elev < grid.at_node["topographic__elevation"])
        grid.at_node["topographic__elevation"][inside] = canyon_elev[inside]

    # set basin
    basin_height = (grid.node_y - slope_basin_break) * slope_basin
    basin_region = grid.at_node["topographic__elevation"] < basin_height
    grid.at_node["topographic__elevation"][basin_region] = basin_height[basin_region]

    # add random value on topographic elevation (+- noise)
    grid.at_node["topographic__elevation"] += (
        2.0 * noise * (np.random.rand(grid.number_of_nodes) - 0.5)
    )

    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    return grid


def create_init_flow_region(
    grid,
    initial_flow_concentration=0.02,
    initial_flow_thickness=200,
    initial_region_radius=200,
    initial_region_center=[15000, 15000],
    initial_region_shape="circle",
    width = 200,
    height=50,
    azimuth=30,
    velocity=1,
    flow_azimuth=110,
    xlim=None, ylim=None, height_factor=None
):
    """ making initial flow region in a grid, assuming lock-exchange type initiation
         of a turbidity current. Plan-view morphology of a suspended cloud is a circle,

         Parameters
         ----------------------
         grid: RasterModelGrid
            a landlab grid object

         initial_flow_concentration: float, optional
            initial flow concentration

         initial_flow_thickness: float, optional
            initial flow thickness

         initial_region_radius: float, optional
            radius of initial flow region

         initial_region_center: list, optional
            [x, y] coordinates of center of initial flow region
    """
    # check number of grain size classes
    if type(initial_flow_concentration) is float or type(initial_flow_concentration) is np.float64:
        initial_flow_concentration_i = np.array([initial_flow_concentration])
    else:
        initial_flow_concentration_i = np.array(initial_flow_concentration).reshape(
            len(initial_flow_concentration), 1
        )

    # initialize flow parameters
    for i in range(len(initial_flow_concentration_i)):
        try:
            grid.add_zeros("flow__sediment_concentration_{}".format(i), at="node")
        except FieldError:
            grid.at_node["flow__sediment_concentration_{}".format(i)][:] = 0.0
        try:
            grid.add_zeros("bed__sediment_volume_per_unit_area_{}".format(i), at="node")
        except FieldError:
            grid.at_node["bed__sediment_volume_per_unit_area_{}".format(i)][:] = 0.0

    try:
        grid.add_zeros("flow__sediment_concentration_total", at="node")
    except FieldError:
        grid.at_node["flow__sediment_concentration_total"][:] = 0.0
    try:
        grid.add_zeros("flow__depth", at="node")
    except FieldError:
        grid.at_node["flow__depth"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__horizontal_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__vertical_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity", at="link")
    except FieldError:
        grid.at_link["flow__horizontal_velocity"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity", at="link")
    except FieldError:
        grid.at_link["flow__vertical_velocity"][:] = 0.0

    # set initial flow region
    print(initial_region_shape)
    if initial_region_shape == "circle":
        initial_flow_region = (
            (grid.node_x - initial_region_center[0]) ** 2
            + (grid.node_y - initial_region_center[1]) ** 2
        ) < initial_region_radius ** 2
        grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness

    elif initial_region_shape == "rectangle":
        corners = get_rotated_rectangle_corners(
            initial_region_center,
            width,
            height,
            azimuth
        )
        polygon = Polygon(corners)
        x_coords = grid.node_x
        y_coords = grid.node_y
        node_coordinates = np.column_stack((x_coords, y_coords))
        initial_flow_region = np.array(
            [polygon.contains(Point(point)) for point in node_coordinates]
        )
        grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness

    else:
        with rasterio.open(initial_region_shape) as src:
            grid_data = src.read(1)[::-1, :]
            transform = src.transform
            dx = transform[0]
            dy = -transform[4]
            # print("dxdy", dx, dy)

        # print(topo_data.shape)
        if (xlim is not None) and (ylim is not None):
            # GeoTIFFの左上からの相対距離を求める（緯度経度差）
            col_start = int((xlim[0] - src.bounds.left) / dx)
            col_end = int((xlim[1] - src.bounds.left) / dx)
            row_start = int((ylim[0] - src.bounds.bottom) / dy)
            row_end = int((ylim[1] - src.bounds.bottom) / dy)

            # インデックスを整数に変換してスライス
            col_start, col_end = int(np.floor(col_start)), int(np.ceil(col_end))
            row_start, row_end = int(np.floor(row_start)), int(np.ceil(row_end))

            # スライスして範囲指定
            grid_data = grid_data[row_start:row_end, col_start:col_end]

        initial_flow_region = (grid_data.flatten() >= 1)

        # Smoothing by median filter
        # grid_data = median_filter(grid_data, size=filter_size)

        initial_flow_thickness = np.where(grid_data < 1, 0, grid_data).flatten() * height_factor
        print(sum(initial_flow_thickness)/1000, "km^3")
        grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness[initial_flow_region]


    grid.at_node["flow__depth"][~initial_flow_region] = 0.0
    for i in range(len(initial_flow_concentration_i)):
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            initial_flow_region
        ] = initial_flow_concentration_i[i]
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            ~initial_flow_region
        ] = 0.0
    grid.at_node["flow__sediment_concentration_total"][initial_flow_region] = np.sum(
        initial_flow_concentration_i
    )

    # set initial velocity
    azimuth_rad = np.radians(flow_azimuth)
    initial_vertical_velocity = velocity * np.cos(azimuth_rad)
    initial_horizontal_velocity = velocity * np.sin(azimuth_rad)
    grid.at_node["flow__vertical_velocity_at_node"][initial_flow_region] = initial_vertical_velocity
    grid.at_node["flow__horizontal_velocity_at_node"][initial_flow_region] = initial_horizontal_velocity
    links_at_true_nodes = grid.links_at_node[initial_flow_region]
    unique_links_at_true_nodes = np.unique(links_at_true_nodes)
    grid.at_link["flow__vertical_velocity"][unique_links_at_true_nodes] = initial_vertical_velocity
    grid.at_link["flow__horizontal_velocity"][unique_links_at_true_nodes] = initial_horizontal_velocity

def rotate_point(point, center, azimuth):
    angle_radians = np.radians(azimuth)
    x, y = point
    cx, cy = center

    x_rotated = np.cos(angle_radians) * (x - cx) - np.sin(angle_radians) * (y - cy) + cx
    y_rotated = np.sin(angle_radians) * (x - cx) + np.cos(angle_radians) * (y - cy) + cy

    return x_rotated, y_rotated

def get_rotated_rectangle_corners(center, width, height, azimuth):
    half_width = width / 2
    half_height = height / 2

    corner1 = rotate_point((center[0] - half_width, center[1] - half_height), center, azimuth)
    corner2 = rotate_point((center[0] + half_width, center[1] - half_height), center, azimuth)
    corner3 = rotate_point((center[0] + half_width, center[1] + half_height), center, azimuth)
    corner4 = rotate_point((center[0] - half_width, center[1] + half_height), center, azimuth)

    return [corner1, corner2, corner3, corner4]

def cordiate_to_node(
    cordiate=[15000, 15000],
    xlim=None,
    ylim=None,
    grid_degree=[0.0121603, -0.0086239],
    spacing=463,
    ):

    center_node_y = (cordiate[0] - ylim[0]) / grid_degree[1] * spacing
    center_node_x = (cordiate[1] - xlim[0]) / grid_degree[0] * spacing
    center_node = [np.round(center_node_x,0), np.round(center_node_y,0)]

    return center_node


# grid = create_topography_from_geotiff('../bathymetry/KikaiN-25mneg.tif',
#                                        xlim=[130.1, 130.4],
#                                        ylim=[28.4, 28.7],
#                                        spacing=25,
#                                        filter_size=[5, 5])
# create_init_flow_region(
#      grid,
#      initial_flow_concentration=[0.01,0.02,0.01],#0.01, #[0.01,0.01], #[0.001,0.001,0.001],
#      initial_flow_thickness=100,            #[m]
#      initial_region_radius=100,             #半径[m]
#      initial_region_center=[15000, 15000],  # 1000, 4000 # spacing*グリッド　0,0はxlimとylimの範囲から
# )

# print(max(grid.at_node["flow__sediment_concentration_total"]+grid.at_node["flow__depth"]))
# np.savetxt('sample.txt', grid.at_node["flow__sediment_concentration_total"])

def create_folder(dirpath):
    directory, folder_name = os.path.split(dirpath)
    if os.path.exists(dirpath):
        choice = input(f"Existing'{dirpath}'. Do you overwrite? (y/n): ").strip().lower()
        if choice == 'y':
            shutil.rmtree(dirpath)
            os.makedirs(dirpath)
            print(f"Overwrited '{dirpath}'")
            f_name = folder_name
        elif choice == 'n':
            match = re.match(r"([A-Za-z]+)(\d+)$", folder_name)
            # print(match)
            prefix, number = match.groups()
            next_num = int(number)+1
            new_folder_name = f"{prefix}{next_num:03d}"
            os.makedirs(os.path.join(directory, new_folder_name))
            f_name = new_folder_name
        else:
            print("Please input 'y' or 'n'")
    else:
        os.makedirs(dirpath)
        f_name = folder_name
    return f_name

def load_inlet(input_dir, identifier, inlet_lat):
    file_path = os.path.join(input_dir, identifier + '.csv')
    df = pd.read_csv(file_path)
    y_range = (df['deg'] >= inlet_lat[0]) & (df['deg'] <= inlet_lat[1])
    C_df = df[y_range]
    time_columns = [int(float(col)) for col in df.columns if col != 'deg']
    str_columns = [col for col in df.columns if col != 'deg']
    inlet_time_steps = np.arange(min(time_columns), max(time_columns)+1, 1)
    original_time = np.array(time_columns, dtype=float)
    c_array = C_df[str_columns].values  # shape = (行数, 時間列数)
    interpolated_array = np.apply_along_axis(
        lambda row: np.interp(inlet_time_steps, original_time, row),
        1,  # axis=1 (行ごとに適用)
        c_array
    )
    result_df = pd.DataFrame(interpolated_array, index=C_df.index)
    return result_df

def interpolate_area(data, inlet, time_step):
    original_index = np.arange(len(data))
    new_index = np.linspace(0, len(data) - 1, len(inlet[0]))
    ip_data = np.interp(new_index, original_index, data[int(time_step)].values)
    return ip_data
