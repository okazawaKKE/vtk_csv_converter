import os
import vtk as vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
import ast
from matplotlib import patches
import datetime
from functools import partial
import mpl_toolkits.mplot3d.proj3d as proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
from scipy.spatial.transform import Rotation

def is_subset(row, target):
    return all(item in target for item in row)

# 元の値を新しい番号に置き換える関数
def replace_values(lst, value_map):
    return [value_map[val] for val in lst]


def visualization2d(x,y,u,x_min,x_max,y_min,y_max,df,figsize):
    fig,ax = plt.subplots(figsize=figsize)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    
    distance = x
    sorted_id = np.argsort(distance)
    x_sorted = x[sorted_id]
    y_sorted = y[sorted_id]
    u_sorted = u[sorted_id] 
    
    scatter = plt.scatter(x_sorted,y_sorted,c=u_sorted,s=3,cmap="jet")
    cbar = plt.colorbar(scatter) 
    cbar.set_label("u")
    #plt.scatter(x,y,c=u,s=1)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    if axis=="z":
        plt.xlabel("x")
        plt.ylabel("y")
    elif axis=="y":
        plt.xlabel("x")
        plt.ylabel("z")
    elif axis=="x":
        plt.xlabel("y")
        plt.ylabel("z")   
    fig.tight_layout()
    onclick_list = []
    callback_with_arg = partial(onclick,ax=ax,onclick_list=onclick_list,df=df)
    fig.canvas.mpl_connect('button_press_event', callback_with_arg)
    plt.show()
    
def onclick(event,ax,onclick_list,df):
    #index_click = data_array[int(round(event.ydata,0)),int(round(event.xdata,0))]
    print("x:",event.xdata,"y:",event.ydata)#"index:",int(index_click))
    #print (event.button,event.x,event.y,event.xdata, event.ydata)
    ax.scatter(event.xdata,event.ydata,s=3,color="red")
    if len(onclick_list) % 2 == 0:
        onclick_list.append([event.xdata,event.ydata])
    else:      
        width  = event.xdata - onclick_list[-1][0]
        height = event.ydata - onclick_list[-1][1]
        r = patches.Rectangle( onclick_list[-1] , width, height, fill=False, edgecolor="red", linewidth=2, label="rectangle")
        ax.add_patch(r)
        x_min = min([event.xdata ,onclick_list[-1][0]])
        x_max = max([event.xdata ,onclick_list[-1][0]])
        y_min = min([event.ydata ,onclick_list[-1][1]])
        y_max = max([event.ydata ,onclick_list[-1][1]])
        if axis=="z":
            condition = (df["x"]>=x_min)&(df["x"]<=x_max)&(df["y"]>=y_min)&(df["y"]<=y_max)
        elif axis=="y":
            condition = (df["x"]>=x_min)&(df["x"]<=x_max)&(df["z"]>=y_min)&(df["z"]<=y_max)
        elif axis=="x":
            condition = (df["y"]>=x_min)&(df["y"]<=x_max)&(df["z"]>=y_min)&(df["z"]<=y_max)
            
        filtered_dfaa = df[condition]
        if rotation:
            print(filtered_dfaa)
            filtered_dfaa = decode_rotaion(filtered_dfaa,degree,axis)
            print(f"dataframe was succesfuly converted \neuler angle={degree}")
        
        print(len(df),len(filtered_dfaa))
        filtered_dfaa.to_csv(csv_path3,index=False)
        print(f"saved {csv_path3}")
        if rotation:
            filtered_dfaa_bound = filtered_dfaa.drop(columns=["u_mag"])
        else:
            filtered_dfaa_bound = filtered_dfaa.drop(columns=["node_id","u_mag"])
        filtered_dfaa_bound.to_csv(csv_path4,index=False)
        print(f"saved {csv_path4}")
        onclick_list.append([event.xdata,event.ydata])
        #onclick_list = []
    plt.draw()
    plt.show()
    
    #ax[0].text(event.xdata-1,event.ydata+1,"id:"+str(index_click)+"\n x:"
    #+str(round(event.ydata,3))+"\n y:"+str(round(event.xdata,3)))
    print("")





def motion(event,ln_v,ln_h,ax,x,y,z,surf):  
    x_click = event.xdata
    y_click = event.ydata
    ln_v.set_xdata(x_click)
    ln_h.set_ydata(y_click)
    x_hyp = np.linspace(int(min(x)),int(max(x)),300)
    y_hyp = np.linspace(int(min(y)),int(max(y)),300)
    z_hyp = np.linspace(int(min(z)),int(max(z)),300)
    if PlotSurf:
        closest_point = search_3d_coor(ax,x_click,y_click,x_hyp,y_hyp,z_hyp,click=False)
        if axis=="z":
            x3_val = closest_point[2]
        elif axis=="y":
            x3_val = closest_point[1]
        elif axis=="x":
            x3_val = closest_point[0]
        else:
            raise IndexError(f"axis {axis} is not correct definition")
        #print(f"x3_val={x3_val}")
        surf = make_surface(x,y,z,ax,axis=axis,surf=surf,closest_point=closest_point,initial=False)
        plt.draw()
    return surf,x3_val


def make_surface(x,y,z,ax,axis,surf=np.nan,closest_point=[0,0,0],initial=True):
    if axis=="z":
        ax1 = x
        ax2 = y
        val = closest_point[2]
    elif axis=="y":
        ax1 = x
        ax2 = z 
        val = closest_point[1]
    elif axis=="x":
        ax1 = y
        ax2 = z
        val = closest_point[0]
    else:
        raise ValueError("axisの入力が間違っています \n x or y or z")       
    
    print(f"{axis}={val}")
    ax1_hyp = np.linspace(min(ax1),max(ax1),10)
    ax2_hyp = np.linspace(min(ax2),max(ax2),10)
    X1, X2 = np.meshgrid(ax1_hyp, ax2_hyp)
    
    if initial:
        X3 = np.full((len(ax2_hyp),len(ax1_hyp)),1) #初期位置
    else:
        if surf in ax.collections:
            surf.remove()
        X3 = np.full((len(ax2_hyp),len(ax1_hyp)),val)
    
    if axis=="z":
        surf = ax.plot_surface(X1,X2,X3,color="black",alpha=0.8)
    elif axis=="y":
        surf = ax.plot_surface(X1,X3,X2,color="black",alpha=0.8)
    elif axis=="x":
        surf = ax.plot_surface(X3,X1,X2,color="black",alpha=0.8)
    else:
        raise ValueError("axisの入力が間違っています \n x or y or z")
    
    return surf


def plot_mesh(ax, tri, x, y, z):
    verts = []
    for simplex in tri.simplices:
        verts.append([(x[simplex[0]], y[simplex[0]], z[simplex[0]]),
                      (x[simplex[1]], y[simplex[1]], z[simplex[1]]),
                      (x[simplex[2]], y[simplex[2]], z[simplex[2]])])
    
    mesh = Poly3DCollection(verts, alpha=0.5, linewidths=1, edgecolors='r')
    ax.add_collection3d(mesh)


def convert_2d(df,z_val):
    dfs = df[round(df[axis],order)==z_val]
    return dfs

def search_3d_coor(ax,x_click,y_click,x,y,z,click=True):
    if x_click is not None and y_click is not None:
        # 3D座標への逆投影を計算する
        # 最も近い点を見つける
        min_distance = float('inf')
        closest_point = None
        for i in range(len(x)):
            x_proj, y_proj, _ = proj3d.proj_transform(x[i], y[i], z[i], ax.get_proj())
            distance = np.sqrt((x_proj - x_click)**2 + (y_proj - y_click)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = (x[i], y[i], z[i])
        if closest_point is not None and click:
            print(f'Clicked coordinates: x={closest_point[0]}, y={closest_point[1]}, z={closest_point[2]}')
   
    return closest_point

def onclick3d(event,ax,df):
    global x3_val
    
    pastfolders = glob.glob(save_dir+"z_*")
    if pastfolders:
        for f in pastfolders:
            os.rmdir(f)
    
    x3_val = round(x3_val,order)
    def_dirname = save_dir + f"{axis}_{x3_val}"
    os.makedirs(def_dirname,exist_ok=True)

    #index_click = data_array[int(round(event.ydata,0)),int(round(event.xdata,0))]
    if event.inaxes == ax:
        # 2D投影面上のクリック位置
        x_click = event.xdata
        y_click = event.ydata
        try:
            #2次元への変換
            if axis =="z":
                x1 = df["x"].to_list()
                x2 = df["y"].to_list()
                x3 = df["z"].to_list()
                x1_min,x1_max = min(x1),max(x1)
                x2_min,x2_max = min(x2),max(x2)
                
            elif axis =="y":
                x1 = df["x"].to_list()
                x2 = df["z"].to_list()
                x3 = df["y"].to_list()
                x1_min,x1_max = min(x1),max(x1)
                x2_min,x2_max = min(x2),max(x2)
                
            elif axis == "x":
                x1 = df["y"].to_list()
                x2 = df["z"].to_list()
                x3 = df["x"].to_list()
                x1_min,x1_max = min(x1),max(x1)
                x2_min,x2_max = min(x2),max(x2)
            else:
                raise ValueError(f"axis {axis} is not defined !")

            #print(f"maximum val {axis} = {max(x3)}")
            print(f"clicked {axis}={x3_val}")
            
            dfs = convert_2d(df,x3_val)

            if axis =="z":
                x1 = dfs["x"].to_list()
                x2 = dfs["y"].to_list()
                u = dfs["u_mag"].to_list()
            elif axis =="y":
                x1 = dfs["x"].to_list()
                x2 = dfs["z"].to_list()
                u = dfs["u_mag"].to_list()
            elif axis == "x":
                x1 = dfs["y"].to_list()
                x2 = dfs["z"].to_list()
                u = dfs["u_mag"].to_list()
            else:
                raise ValueError(f"axis {axis} is not defined !")

            visualization2d(x1,x2,u,x1_min,x1_max,x2_min,x2_max,df,figsize=(6,5))
            plt.show()
        except:
            pass
    print("")

def visualization3d(df):
    if rotation:
        #回転座標系の場合はdfを変換する
        df = encode_rotaion(df,degree=degree,axis=axis)
    
    #各コラムを読み込み
    x = df["x"].to_list()
    y = df["y"].to_list()
    z = df["z"].to_list()
    u = df["u_mag"].to_list()
    #ダウンサンプリングする
    x,y,z,u = np.array(x),np.array(y),np.array(z),np.array(u)
    ids = np.random.choice(len(x), size=downsampling_size, replace=False)
    x_ids = x[ids]
    y_ids = y[ids]
    z_ids = z[ids]
    u_ids = u[ids]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x_ids, y_ids, z_ids,c=u_ids,cmap="jet")
    # カラーバーを追加し、位置を調整
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #fig.colorbar(surf, cax=cax)

    cbar = plt.colorbar(scatter) 
    cbar.set_label("u_mag")
    ax.set_xlim(min(x),max(x))
    ax.set_ylim(min(y),max(y))
    ax.set_zlim(min(z),max(z))
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    onclick_list = []
    
    ln_v = ax.axvline(0,color="black")
    ln_h = ax.axhline(0,color="black")
    
    surf = make_surface(x,y,z,ax,axis=axis,initial=True)
    x3_val = 0
    
    def onclick_wrapper(event):
        nonlocal surf
        global x3_val
        try:
            surf,x3_val = motion(event, ln_v=ln_v,ln_h=ln_h,ax=ax,x=x,y=y,z=z,surf=surf)
        except:
            pass
            #print("out of axis")
    
    
    
    plt.ion()
    plt.show()
    fig.canvas.mpl_connect("motion_notify_event",onclick_wrapper)
    callback_with_arg1 = partial(onclick3d,df=df,ax=ax)
    fig.canvas.mpl_connect('button_press_event', callback_with_arg1)
    
    plt.ioff()
    plt.show()
    

def encode_rotaion(df,degree,axis):
    df = df.reset_index()
    array = np.array([df["x"].to_list(),
                      df["y"].to_list(),
                      df["z"].to_list()]).T
    
    rot = Rotation.from_euler(axis,degree)
    array_rot = rot.apply(array).T
    new_x = array_rot[0,:]
    new_y = array_rot[1,:]
    new_z = array_rot[2,:]
    df_u = df.loc[:,"u_x":]
    new_df_pre = pd.DataFrame({"x":new_x,"y":new_y,"z":new_z})
    new_df = pd.concat([new_df_pre,df_u],axis=1)
    
    return new_df

def decode_rotaion(df,degree,axis):
    df = df.reset_index()
    array = np.array([df["x"].to_list(),
                      df["y"].to_list(),
                      df["z"].to_list()]).T
    
    rot = Rotation.from_euler(axis,-degree)
    array_rot = rot.apply(array).T
    new_x = array_rot[0,:]
    new_y = array_rot[1,:]
    new_z = array_rot[2,:]
    df_u = df.loc[:,"u_x":]
    print(df_u)
    new_df_pre = pd.DataFrame({"x":new_x,"y":new_y,"z":new_z})
    new_df = pd.concat([new_df_pre,df_u],axis=1)
    return new_df

"""
3次元から2次元に削減
クリックで座標を取得
csvに保存

mode1 openfoamのデータからcsv作成
    read_foam = True
    read_csv  = False
mode2 3次元のcsvデータを2次元に切り出し、クリックで取得
    read_foam = False
    read_csv  = False
option 過去のmode2のcsvから
"""

current_time = datetime.datetime.now()
current_time_str = current_time.strftime("%Y-%m%d-%H%M")
print("現在の時間:", current_time_str)


#Input################################################
rootDir     = "./SBD_project/LES_smagorinsky_10s_1_25s"
read_foam   = True  #openfoamのデータからcsvを作成
read_csv    = False #openfoamデータを加工したcsvから読み込み(すでに2次元に加工したもの)
get_csv     = True  #openfoamを加工したcsvからインタラクティブに座標を取得してcsv変換
#if get_csv==True:
rotation    = False
degree       = np.pi/3

downsampling_size = 5000

order       = 2     # z方向をどこまでそろえるか　0.01
View3D      = True  #3次元のデータを可視化するか
PlotSurf    = True  #３次元プロットで面をインタラクティブに可視化
axis        = "z"   #面を可視化する時にどの断面を選択するか
########################################################


fileName = rootDir + "/system/controlDict"
folder_name = os.path.basename(rootDir)
save_dir = f"./click/{folder_name}/{current_time_str}/"
os.makedirs(save_dir,exist_ok=True)


csv_path1 = rootDir + f"/data_{folder_name}.csv"                     #座標と状態変数
csv_path2 = rootDir + f"/cell_node_{folder_name}.csv"                #セルとノードの関係
csv_path3 = save_dir + f"click_{folder_name}.csv"         #クリックした状態量
csv_path4 = save_dir + f"click_boundary_{folder_name}.csv"#クリックした状態量をsimscale用に加工


df_rootDir = f"./dataframe/{folder_name}/" 
os.makedirs(df_rootDir,exist_ok=True)

if read_csv:
    order = 3
    z     = 2.999
    path_filtered  = df_rootDir + f"filtered_df_z{z}_order{order}.csv"
    path_dfaa      = df_rootDir + f"dfaa_z{z}_order{order}.csv"



if read_foam:
    # reader
    print("****reading foam results****")
    reader = vtk.vtkOpenFOAMReader()
    reader.SetFileName(fileName)
    reader.CreateCellToPointOn()
    reader.DecomposePolyhedraOn()
    reader.EnableAllCellArrays()
    reader.Update()
    print("finished")
    
    print("getting latest results")
    tArray = vtk_to_numpy(reader.GetTimeValues())    
    reader.UpdateTimeStep(tArray[-1]) #latest time           
    reader.Update() 
    
    print("reading variable")
    for index in range(reader.GetNumberOfCellArrays()):
        nname = reader.GetCellArrayName(index)
        print(f"id={index} {nname}")
    print("finised")
    
    
    data = reader.GetOutput()
    usg = data.GetBlock(0)
    gf = vtk.vtkGeometryFilter() 
    gf.SetInputData(usg)
    gf.Update()
    vtk_data = gf.GetOutput()
       
    #cell_id_list : セル頂点の節点番号
    #cell_node_indices_list : セルを構成する他３つの節点番号
    cell_id_list = []
    cell_node_indices_list = []
    
    points = vtk_data.GetPoints()
    number_of_cells = vtk_data.GetNumberOfCells()
    number_of_nodes = vtk_data.GetNumberOfPoints()
    
    for cell_id in range(number_of_cells):
        cell = vtk_data.GetCell(cell_id)
        cell_point_ids = cell.GetPointIds()
        number_of_points_in_cell = cell_point_ids.GetNumberOfIds()

        cell_node_indices = []
        for point_id in range(number_of_points_in_cell):
            cell_node_indices.append(cell_point_ids.GetId(point_id))
            #point = vtk_data.GetPoints().GetPoint(point_id)
            #print(f"id:{point_id} point{list(point)}")

        cell_id_list.append(cell_id)
        cell_node_indices_list.append(cell_node_indices)
    
    
    
    
    cell2point = vtk.vtkCellDataToPointData()
    cell2point.SetInputData(gf.GetOutput())
    cell2point.Update()
    
    #節点番号と座標を取得する
    node_coordinates = [cell2point.GetOutput().GetPoint(i) for i in range(number_of_nodes)]
    node_number = [i for i in range(number_of_nodes)]
    x = [node[0] for node in node_coordinates]
    y = [node[1] for node in node_coordinates]
    z = [node[2] for node in node_coordinates]
    
    
    #p = vtk_to_numpy(cell2point.GetOutput().GetPointData().GetAbstractArray(7)) #p
    
    u = vtk_to_numpy(cell2point.GetOutput().GetPointData().GetArray("U")) #U
    

    u_x = u[:,0]    
    u_y = u[:,1]    
    u_z = u[:,2]    
    
    u_mag = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    print(f"セル数{len(cell_id_list)} 節点数{len(node_number)}")
    
    ###############################
    dict1=dict(node_id=node_number,
               x=x,
               y=y,
               z=z,
               u_x=u_x,
               u_y=u_y,
               u_z=u_z,
               u_mag=u_mag)
    df1 = pd.DataFrame(data=dict1)
    df1.to_csv(csv_path1,index=False)
    ################################
    dict2=dict(cell_id=cell_id_list,
               cell_node_ids=cell_node_indices_list)
    df2 = pd.DataFrame(data=dict2)
    df2.to_csv(csv_path2,index=False)
    ###############################
    
if read_csv:
    df = pd.read_csv(csv_path1)
    #print(df)
    df_xsort = df.sort_values("x")
    df_ysort = df.sort_values("y")
    df_zsort = df.sort_values("z")
    #print(df_zsort)
    #z座標が同じものを取得
    x_list = df_xsort["x"].to_list()
    y_list = df_ysort["y"].to_list()
    z_list = df_zsort["z"].to_list()
    
    x_min, x_max = np.min(x_list), np.max(x_list)
    y_min, y_max = np.min(y_list), np.max(y_list)
    
    filtered_df = pd.read_csv(path_filtered)
    filtered_df['cell_node_ids'] = [ast.literal_eval(d) for d in filtered_df['cell_node_ids']]
    cellnode_list = filtered_df['cell_node_ids'].to_list()
    
    dfaa = pd.read_csv(path_dfaa)
    x       = dfaa["x"].to_list()
    y       = dfaa["y"].to_list() 
    u_mag   = dfaa['u_mag'].to_list()
    

    visualization2d(x,y,u_mag,x_min,x_max,y_min,y_max,cellnode_list=cellnode_list)
    
if get_csv:
    df = pd.read_csv(csv_path1)
    x = df["x"].to_list()
    y = df["y"].to_list()
    z = df["z"].to_list()
    u_mag = df["u_mag"].to_list()
    #print(df)
    df_xsort = df.sort_values("x")
    df_ysort = df.sort_values("y")
    df_zsort = df.sort_values("z")
    #print(df_zsort)
    #z座標が同じものを取得
    x_list = df_xsort["x"].to_list()
    y_list = df_ysort["y"].to_list()
    z_list = df_zsort["z"].to_list()
    
    x_min, x_max = np.min(x_list), np.max(x_list)
    y_min, y_max = np.min(y_list), np.max(y_list)
    

        
    if View3D:
        visualization3d(df) #df
        if not os.listdir(save_dir):
            os.rmdir(save_dir)
    else:
        df_list =  []
        z_list_df = []
        z_list = [round(z,order) for z in z_list]
        z_list = list(dict.fromkeys(z_list))

        for z_val in z_list:
            dfs = df_zsort[round(df_zsort["z"],order)==z_val]
            if len(dfs) != 0:
                df_list.append(dfs)
                z_list_df.append(z_val)

        df_cellnode = pd.read_csv(csv_path2)
        # str -> list　
        df_cellnode['cell_node_ids'] = [ast.literal_eval(d) for d in df_cellnode['cell_node_ids']]

        i = 0
        for dfa in df_list:
            z = z_list_df[i]
            i += 1
            if z >= 3:
                print(f"z = {z}")
                #print(dfa)
                nodes = dfa["node_id"].to_list() #target nodes
                print(nodes)
                #print(df_cellnode)
                #nodesの値のみで構成される節点番号をcell node indices listから抽出したい
                filtered_df = df_cellnode[df_cellnode['cell_node_ids'].apply(lambda x: is_subset(x, nodes))]

                #filtered_dfのcell_node_idsの値を1から振りなおす
                #全ての値を取得して一意にする
                unique_values = sorted(set(val for sublist in filtered_df['cell_node_ids'] for val in sublist))
                #新しい番号を割り当てる辞書を作成
                new_value_map = {val: idx for idx, val in enumerate(unique_values)}
                #データフレーム内のリストを置き換える
                filtered_df['cell_node_ids'] = filtered_df['cell_node_ids'].apply(lambda x: replace_values(x, new_value_map))
                print(filtered_df)
                filtered_df.to_csv(df_rootDir+f"filtered_df_z{z}_order{order}.csv",index=False)

                cellnode_list = filtered_df['cell_node_ids'].to_list()


                #dfaをnode_id列を基準にして昇順に並べなおす->dfaa
                dfaa = dfa.sort_values(by="node_id")
                x       = dfaa["x"].to_list()
                y       = dfaa["y"].to_list() 
                u_mag   = dfaa['u_mag'].to_list()
                #print(dfaa)
                dfaa.to_csv(df_rootDir+f"dfaa_z{z}_order{order}.csv",index=False)
                #print(cellnode_list)
                #raise ValueError
                visualization2d(x,y,u_mag,x_min,x_max,y_min,y_max,figsize=(8,5))


