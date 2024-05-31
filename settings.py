import numpy as np
import datetime,os


current_time = datetime.datetime.now()
current_time_str = current_time.strftime("%Y-%m%d-%H%M")
print("現在の時間:", current_time_str)


#Input################################################
rootDir     = "./LES_smagorinsky_10s_1_25s"
print(f"rootDir : {rootDir}")

"""
mode1 openfoamのデータからcsv作成
    read_foam = True
mode2 3次元のcsvデータを2次元に切り出し、クリックで取得
    get_csv = True
"""
read_foam           = True  #openfoamのデータからcsvを作成
get_csv             = True  #openfoamを加工したcsvからインタラクティブに座標を取得してcsv変換
read_time_series    = False
var                 = "U"
#if get_csv==True
rotation            = False
degree              = np.pi/3
downsampling_size   = 6000 #可視化をする際にどこまでサンプリング数を減らすか、元の解像度より大きい値を入力するとダウンサンプリングされない
#可視化粒子のサイズ
scatter_size_3d     = 50
scatter_size_2d     = 25


#全体計算を回す際の総計算時間（足りていない部分は外挿する）
f_time              = 201

order               = 0.5   #断面軸方向をどこまでそろえるか　ex. 0.1はプラスマイナス0.1で断面が検索される
View3D              = True  #3次元のデータを可視化するか
axis                = "z"   #面を可視化する時の軸
########################################################


fileName = rootDir + "/system/controlDict"
folder_name = os.path.basename(rootDir)
save_dir = f"./click/{folder_name}/{current_time_str}/"
os.makedirs(save_dir,exist_ok=True)
csv_path1 = rootDir + f"/data_{folder_name}.csv"       #座標と状態変数
csv_path2 = rootDir + f"/cell_node_{folder_name}.csv"  #セルとノードの関係
csv_path3 = save_dir + f"click_{folder_name}.csv"   #クリックした状態量
csv_path4 = save_dir + f"click_boundary_{folder_name}.csv"#クリックした状態量をsimscale用に加工
df_rootDir = f"./dataframe/{folder_name}/" 
os.makedirs(df_rootDir,exist_ok=True)