from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd

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