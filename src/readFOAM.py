import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import patches
import pandas as pd
import ast
from functools import partial
import mpl_toolkits.mplot3d.proj3d as proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
from scipy.spatial.transform import Rotation
import settings
from src.rotation_df import*


current_time = settings.current_time
#Input################################################
rootDir     = settings.rootDir
#if get_csv==True:
rotation    = settings.rotation
degree       = settings.degree

downsampling_size = settings.downsampling_size

order       = settings.order     # z方向をどこまでそろえるか　0.01
View3D      = settings.View3D  #3次元のデータを可視化するか
axis        = settings.axis   #面を可視化する時にどの断面を選択するか
########################################################
csv_path1 = settings.csv_path1
csv_path2 = settings.csv_path2
csv_path3 = settings.csv_path3
csv_path4 = settings.csv_path4
csv_path3_original = settings.csv_path3
csv_path4_original = settings.csv_path4
save_dir = settings.save_dir

scatter_size_3d = settings.scatter_size_3d
scatter_size_2d = settings.scatter_size_2d

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
    
    scatter = plt.scatter(x_sorted,y_sorted,c=u_sorted,s=scatter_size_2d,cmap="jet")
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
    global onclick_list
    onclick_list = []
    callback_with_arg = partial(onclick,ax=ax,onclick_list=onclick_list,df=df)
    fig.canvas.mpl_connect('button_press_event',callback_with_arg)
    print(f"onclick_list {onclick_list}")
    plt.show()
    
def onclick(event,ax,onclick_list,df):
    ax.scatter(event.xdata,event.ydata,s=10,color="red")
    if len(onclick_list) % 2 == 0:
        print(f"top left of rectangle x:,{event.xdata}, y:{event.ydata}")
        onclick_list.append([event.xdata,event.ydata])
    else:
        print(f"bottom right of rectangle x:,{event.xdata}, y:{event.ydata}\n")
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
        
        print(f"based df size :{len(df)}, output df size:{len(filtered_dfaa)}")
        filtered_dfaa.to_csv(csv_path3,index=False)
        print(f"saved csv {csv_path3}")
        if rotation:
            filtered_dfaa_bound = filtered_dfaa.drop(columns=["u_mag"])
        else:
            filtered_dfaa_bound = filtered_dfaa.drop(columns=["node_id","u_mag"])
        print(filtered_dfaa_bound)
        filtered_dfaa_bound.to_csv(csv_path4,index=False)
        print(f"saved csv {csv_path4}")
        onclick_list.append([event.xdata,event.ydata])
        #onclick_list = []
    plt.draw()
    plt.show()
    
    #ax[0].text(event.xdata-1,event.ydata+1,"id:"+str(index_click)+"\n x:"
    #+str(round(event.ydata,3))+"\n y:"+str(round(event.xdata,3)))
    #print("")





def motion(event,ln_v,ln_h,ax,df,surf):  
    x_click = event.xdata
    y_click = event.ydata
    x,y,z = df["x"].to_list(),df["y"].to_list(),df["z"].to_list()
    ln_v.set_xdata(x_click)
    ln_h.set_ydata(y_click)
    x_hyp = np.linspace(min(x),max(x),50)
    y_hyp = np.linspace(min(y),max(y),50)
    z_hyp = np.linspace(min(z),max(z),50)
    #print(min(z),max(z))
    #raise ValueError
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
    #print(z)
    dfs = df[(df[axis]<=z_val+order)&(df[axis]>=z_val-order)]
    print(f"converted 3d to 2d\nselected val : {z_val}, min val : {min(dfs[axis].to_list())}, max val:{max(dfs[axis].to_list())}")
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
    #closest_point = 3.4
    return closest_point

def onclick3d(event,ax,df):
    global x3_val
    
    pastfolders = glob.glob(save_dir+"z_*")
    if pastfolders:
        for f in pastfolders:
            os.rmdir(f)
    
    #x3_val = round(x3_val,order)
    def_dirname = save_dir + f"{axis}_{x3_val}_pm{order}"
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
            #x3_val = 3.2
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

            visualization2d(x1,x2,u,x1_min,x1_max,x2_min,x2_max,dfs,figsize=(6,5))
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
    if len(x)>=downsampling_size:
        ids = np.random.choice(len(x), size=downsampling_size, replace=False)
    else:
         ids = np.random.choice(len(x), size=len(x), replace=False)
    x_ids = x[ids]
    y_ids = y[ids]
    z_ids = z[ids]
    u_ids = u[ids]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x_ids, y_ids, z_ids,c=u_ids,cmap="jet",s=scatter_size_3d)

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
    
    def onclick_wrapper(event,df):
        nonlocal surf
        global x3_val
        try:
            #print(min(z),max(z))
            surf,x3_val = motion(event, ln_v=ln_v,ln_h=ln_h,ax=ax,df=df,surf=surf)
        except:
            pass
    
    plt.ion()
    plt.show()
    callback_with_arg1 = partial(onclick_wrapper,df=df)
    fig.canvas.mpl_connect("motion_notify_event",callback_with_arg1)
    callback_with_arg2 = partial(onclick3d,df=df,ax=ax)
    fig.canvas.mpl_connect('button_press_event', callback_with_arg2)
    plt.ioff()
    plt.show()


def visualization3d_series(list_csv,list_time,f_time):
    debug = []
    for csvpath,time in zip(list_csv[::-1],list_time[::-1]):
        print(csvpath,f"t = {time}")
        #raise ValueError
        df = pd.read_csv(csvpath)
        if rotation:
            #回転座標系の場合はdfを変換する
            df = encode_rotaion(df,degree=degree,axis=axis)
        if csvpath == list_csv[-1]:
            #各コラムを読み込み
            x = df["x"].to_list()
            y = df["y"].to_list()
            z = df["z"].to_list()
            u = df["u_mag"].to_list()

            #ダウンサンプリングする
            x,y,z,u = np.array(x),np.array(y),np.array(z),np.array(u)

            if len(x)>=downsampling_size:
                ids = np.random.choice(len(x), size=downsampling_size, replace=False)
            else:
                 ids = np.random.choice(len(x), size=len(x), replace=False)
            x_ids = x[ids]
            y_ids = y[ids]
            z_ids = z[ids]
            u_ids = u[ids]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x_ids, y_ids, z_ids,c=u_ids,cmap="jet",s=scatter_size_3d)
            cbar = plt.colorbar(scatter) 
            cbar.set_label("u_mag")
            ax.set_xlim(min(x),max(x))
            ax.set_ylim(min(y),max(y))
            ax.set_zlim(min(z),max(z))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            fig.tight_layout()
            ln_v = ax.axvline(0,color="black")
            ln_h = ax.axhline(0,color="black")
            surf = make_surface(x,y,z,ax,axis=axis,initial=True)
            x3_val = 0
            def onclick_wrapper(event,df):
                nonlocal surf
                global x3_val
                try:
                    surf,x3_val = motion(event, ln_v=ln_v,ln_h=ln_h,ax=ax,df=df,surf=surf)
                except:
                    pass
            plt.ion()
            plt.show()
            callback_with_arg1 = partial(onclick_wrapper,df=df)
            fig.canvas.mpl_connect("motion_notify_event",callback_with_arg1)
            callback_with_arg2 = partial(onclick3d,df=df,ax=ax)
            fig.canvas.mpl_connect('button_press_event', callback_with_arg2)
            plt.ioff()
            plt.show()
            df_concat = pd.read_csv(csv_path4)
            df_concat = df_concat.assign(t=time)
            debug.append(x3_val)
            #print(df_concat)
        else:
            filtered_df = convert_df_csv(df)
            filtered_df_t = filtered_df.assign(t=time)
            debug.append(x3_val)
            print(f"made df time:{time}, size:{len(filtered_df_t)}")
            df_concat = pd.concat([df_concat,filtered_df_t])
    
    if list_time[-1]<=f_time:
        time_list = np.linspace(list_time[-1]+10,f_time,20)
        for time in time_list:
            print(f"{time}s")
            df = pd.read_csv(list_csv[-1])
            filtered_df = convert_df_csv(df)
            filtered_df_t = filtered_df.assign(t=time)
            debug.append(x3_val)
            df_concat = pd.concat([df_concat,filtered_df_t])
    #print(debug)    
    return df_concat



def convert_df_csv(df):
    global x3_val
    df = convert_2d(df,x3_val)
    #print(onclick_list)
    x_min = onclick_list[0][0]
    x_max = onclick_list[1][0]
    y_min = onclick_list[1][1]
    y_max = onclick_list[0][1]
    if axis=="z":
        condition = (df["x"]>=x_min)&(df["x"]<=x_max)&(df["y"]>=y_min)&(df["y"]<=y_max)
    elif axis=="y":
        condition = (df["x"]>=x_min)&(df["x"]<=x_max)&(df["z"]>=y_min)&(df["z"]<=y_max)
    elif axis=="x":
        condition = (df["y"]>=x_min)&(df["y"]<=x_max)&(df["z"]>=y_min)&(df["z"]<=y_max)
        
    filtered_dfaa = df[condition]
    
    if rotation:
        filtered_dfaa = decode_rotaion(filtered_dfaa,degree,axis)
        filtered_dfaa_bound = filtered_dfaa.drop(columns=["u_mag"])
    else:
        filtered_dfaa_bound = filtered_dfaa.drop(columns=["node_id","u_mag"])
    
    return filtered_dfaa_bound
    


def add_timeinfo_path(csv_path,time):
    list = os.path.split(csv_path)
    folder = list[0]
    file = list[1]
    name,ext = os.path.splitext(file)
    new_name = name + f"_{time}" + ext
    new_path = os.path.join(folder,new_name) 
    return new_path    