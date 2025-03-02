import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial import cKDTree
import shutil
import math
import re

#######################  INPUT  ##################################
# root_path ="\\10.226.62.220\public\Nakanishi\liutest\401"
root_path = r"D:/turb2d/GrandBanks" #"F:Turb2d/GrandBanks"
output = "../Article/Grandbanks/py_out" 
criteria_csv = "../Article/Grandbanks/criteria.csv"

gs_num = 3 # number of grain size class
# assume nc file name as "GB028"
event = "K"#"GB"#"YC"#
case_s, case_e = 28, 29 #151
# erosion_flag = True #False #
vmax = 0.1 # sediment thickness and erosion depth scales [m]

# For YY plots
plot_gs = '64um'
gs_order = 1 #order of target grain size class at "Ds"
ymin, ymax = 0, 2.5 
######################################################################

def find_max_numbered_file(directory, prefix="GB", suffix=".nc"):
    # ファイル名に含まれる数値を抽出するための正規表現
    pattern = re.compile(rf"{prefix}_(\d+){suffix}")
    max_number = -1
    max_file = None

    # ディレクトリ内のすべてのファイルを調べる
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_file = filename

    return max_file, max_number

def get_variable_from_file(file_path, variable_name, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
    except UnicodeDecodeError:
        raise ValueError(f"Failed to decode the file with encoding {encoding}") 
    # 変数名に対応する正規表現パターンを作成
    pattern = rf"{variable_name}\s*=\s*(.+)"
    # 正規表現を使用して変数の値を取得
    match = re.search(pattern, content)
    if match:
        value = match.group(1).strip()
        try:
            # 評価して適切な型に変換
            return eval(value)
        except:
            return value
    else:
        raise ValueError(f"Variable '{variable_name}' not found in the file.")

def grd_sampling(criteria, root_path, out_path, gs_num, vmax, fig_on=False, erosion_flag=False, new_dir=None):
    pd_list = [pd.DataFrame() for _ in range(gs_num)]
    um = [col for col in criteria.columns if col.endswith('um')]
    
    if new_dir is None:
        subfolders = [f.name for f in os.scandir(root_path) if f.is_dir()]
        subfolders.sort()
        print("計算結果のフォルダ数:", len(subfolders))
    else:
        subfolders = new_dir
        
    for subfolder in subfolders:
        # print("CASE",subfolder)
        # print(os.path.join(root_path, subfolder))
        if not os.path.isdir(os.path.join(root_path, subfolder)):
            continue
        max_file, max_num = find_max_numbered_file(os.path.join(root_path, subfolder), prefix=subfolder)
        if max_num < 1000:
            continue  
        else:
            result = pd.DataFrame() 
            nc_path = os.path.join(root_path, subfolder, max_file)
            file_pattern = os.path.join(root_path, subfolder, "run_turb2d*")
            files = glob.glob(file_pattern)
            file_path = files[0]
            xlim = get_variable_from_file(file_path, "xlim")
            ylim = get_variable_from_file(file_path, "ylim")
            # print(type(xlim))
            if isinstance(xlim, tuple):
                xlim,ylim = xlim[0],ylim[0]
            # print(xlim,ylim)
#             if result.empty:
#                 result = criteria.copy() #pd.read_csv(criteria)
                
            for i in np.arange(gs_num):
                # pd_name = "GS" + str(i)
                bed_name = "bed__sediment_volume_per_unit_area_"+ str(i)
                c_name = "flow__sediment_concentration_" + str(i)
                e_name = "max__erosion_" + str(i)
                
                if os.path.isfile(nc_path):
                    # print(f"{subfolder} の{pd_name} をサンプリング")
                    row_data = nc.Dataset(nc_path, 'r')  
                    # for variable in row_data.variables:
                    #     print(variable)
                    bed = row_data.variables[bed_name][0]
                    erosion = row_data.variables[e_name][0]##############
                    c = row_data.variables[c_name][0]
                    flow = row_data.variables["flow__depth"][0]
                    topo = row_data.variables["topographic__elevation"][0]
                    # print("lat, lon", z.shape)
                    row_data.close()

                    lat_min, lat_max = ylim
                    lon_min, lon_max = xlim
                    lat_num_grids, lon_num_grids = bed.shape
                    lat_interval = (lat_max - lat_min) / (lat_num_grids - 1)
                    lon_interval = (lon_max - lon_min) / (lon_num_grids - 1)
                    latitudes = np.linspace(lat_min, lat_max, lat_num_grids)
                    longitudes = np.linspace(lon_min, lon_max, lon_num_grids)
                    X, Y = np.meshgrid(longitudes, latitudes)
                    grid_points = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)
                    tree = cKDTree(grid_points)
                    # 各ポイントの最近傍グリッドポイントのインデックスを取得
                    within_index = (
                        (criteria['lon'] >= lon_min) & (criteria['lon'] <= lon_max) &
                        (criteria['lat'] >= lat_min) & (criteria['lat'] <= lat_max)
                    )
                    within_criteria = criteria[within_index]
                    distances, indices = tree.query(within_criteria[['lat', 'lon']].values)
                    threshold = 0.01
                    result_bed = np.where(distances <= threshold, bed.flatten()[indices], np.nan)
                    result_c = np.where(distances <= threshold, c.flatten()[indices], np.nan) 
                    result_e = np.where(distances <= threshold, erosion.flatten()[indices], np.nan) #########
                    result_flow = np.where(distances <= threshold, flow.flatten()[indices], np.nan) 
                    result[subfolder] = result_bed + (result_c * result_flow) - result_e
                    if pd_list[i].empty:
                        pd_list[i] = within_criteria.copy()
                    pd_list[i] = pd.concat([pd_list[i], result], axis=1)
                    
                    # Figure
                    if fig_on==True:
                        obs = within_criteria[um[i]].copy()
                        mask_999 = obs == 999
                        obs_normal = obs.copy()
                        obs_normal[mask_999] = np.nan  # 通常のデータポイントでは NaN にする
                        obs_999 = obs[mask_999]
                        
                        plt.figure(figsize=(10, 8))
                        if erosion_flag==True:
                            z = -erosion
                            vmin = 0.001
                            label = "Erosion"
                            cmap_c = 'jet'
                        else:
                            z = bed + (c * flow) - erosion
                            vmin = 0.001
                            label = "Thickness"
                            cmap_c = 'jet'
                        z_masked = np.ma.masked_less(z, vmin)
                        
                        # 'jet' カラーマップを取得し、マスクされた部分に白色を追加
                        cmap = plt.get_cmap(cmap_c)
                        cmap = cmap.copy()
                        cmap_obs = cmap.copy()
                        cmap.set_bad(color='white')
                        cmap_obs.set_under(color='white')
                        
                        plt.imshow(z_masked, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower', 
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                        cbar = plt.colorbar(aspect=50, pad=0.02, #orientation='horizontal', extend='both',
                                            shrink=0.9, label=label+" [m]")
                        # plt.colorbar()
                        min_val = round(topo.min(), -2)
                        levels = np.arange(min_val, 0, 100)
                        cs = plt.contour(X, Y, topo, levels=levels, colors="k", 
                                         linestyles='solid', linewidths=0.2, alpha=0.5,#cmap='terrain',
                                        )
                        levels_coarse = np.arange(min_val, 0, 1000)
                        cs_coarse = plt.contour(X, Y, topo, levels=levels_coarse, colors='k', 
                                                linestyles='solid', linewidths=0.6, alpha=0.5)
                        plt.clabel(cs_coarse, inline=True, fontsize=6)
                        
                        # obs が 999 のデータポイントを黒色でプロット
                        sc = plt.scatter(within_criteria.loc[mask_999, 'lon'], within_criteria.loc[mask_999, 'lat'], 
                                    c='w', marker='^', linewidths=0.5, edgecolors="k", alpha=1, 
                                    label='Not Recover', zorder=30)
                        # 通常のデータポイントを散布図でプロット
                        plt.scatter(within_criteria['lon'], within_criteria['lat'], c=obs_normal, 
                                    cmap=cmap_obs, marker='o', linewidths=0.7, edgecolors="k", alpha=1, 
                                    vmin=vmin, vmax=vmax, label='Coring Site', zorder=20)
            
                        plt.title(f'{subfolder}_{max_num}s - {um[i]}')
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        # plt.legend()
                        plt.savefig(os.path.join(out_path,"map",label, f"{subfolder}- {um[i]}.png"), 
                                    format="png", dpi=300, bbox_inches='tight')
                        # plt.show()

                else:
                    continue
    return pd_list

# if event == "GB":
#     output = "../Article/Grandbanks/py_out" 
#     # criteria_csv = "../Article/Grandbanks/criteria.csv"
# elif event =="YC":
#     output = "../Article/Yuchun/output"#"../Article/Grandbanks/test.csv" #サンプリングの出力ファイル名　デフォルト出力されない
#     # criteria_csv = "../Article/Yuchun/criteria.csv"
# elif event == "K":
#     output = "../Article/amami_hist/py_out"#"../Article/Grandbanks/test.csv" #サンプリングの出力ファイル名　デフォルト出力されない
#     # criteria_csv = "../Article/amami_hist/criteria.csv"
# elif event == "TA":
#     output = "../Article/Arai2013/py_out"#"../Article/Grandbanks/test.csv" #サンプリングの出力ファイル名　デフォルト出力されない
#     # criteria_csv = "../Article/Arai2013/criteria.csv"

case_list=[]
for i in np.arange(case_s, case_e+1, 1):
    case = event + str(i).zfill(3)
    case_list.append(case)
print(case_list)

result = pd.DataFrame()
criteria = pd.read_csv(criteria_csv) 
res_bt = grd_sampling(criteria, root_path, output, gs_num, vmax, fig_on=True, erosion_flag=False, new_dir=case_list)
res_ero = grd_sampling(criteria, root_path, output, gs_num, vmax, fig_on=True, erosion_flag=True, new_dir=case_list)

for i in np.arange(gs_num):
    res_bt[i].to_csv(os.path.join(output,"csv",event+case+"_GS"+str(i)+".csv"), index=False)

calc_list = res_bt[0].columns[8:].tolist()
num_plots = len(calc_list)
print(calc_list)
cols = 3  # 列数を設定
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))

# 各 calc に対してプロットを作成
for i, calc in enumerate(calc_list):
    row = i // cols
    col = i % cols
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.plot([ymin, ymax], [ymin, ymax], 'k--', alpha=0.3, label='1:1')
    tthickness = 0 
    for j in np.arange(gs_num):
        tthickness += res_bt[j][calc]
    ax.scatter(res_bt[0]['thickness'], tthickness, marker=".", label='Total')
    ax.scatter(res_bt[0][plot_gs], res_bt[1][calc], marker=".", label='vf sand')

    dfs = [res_bt[gs_order][calc], res_bt[0][plot_gs]]
    invalid_indices = set()
    for df in dfs:
        invalid_indices.update(df.index[(df == 999) | (df.isna())])
    pre = res_bt[gs_order][calc].drop(index=invalid_indices)
    obs_ = res_bt[0][plot_gs].drop(index=invalid_indices)
    rmse = ((pre -  obs_)**2)/len(pre)
    RMSE = (rmse.sum())**0.5
    print(calc,RMSE)
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(ymin, ymax)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Obs')
    ax.set_ylabel(calc)
    ax.legend()

for j in range(i + 1, rows * cols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.savefig(os.path.join(output, f"{calc_list[0]}-{calc_list[-1]}.png"), 
                                    format="png", dpi=600, bbox_inches='tight')
# plt.show()