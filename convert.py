from src.readFOAM import *
import settings

fileName = settings.fileName
df_rootDir = settings.df_rootDir
read_time_series = settings.read_time_series
read_foam = settings.read_foam
get_csv  = settings.get_csv
var = settings.var
f_time = settings.f_time

csv_path1 = settings.csv_path1
csv_path2 = settings.csv_path2
csv_path1_original = settings.csv_path1
csv_path2_original = settings.csv_path2

View3D = settings.View3D


"""
3次元から2次元に削減
クリックで座標を取得
csvに保存

mode1 openfoamのデータからcsv作成
    read_foam = True
mode2 3次元のcsvデータを2次元に切り出し、クリックで取得
    get_csv = True
"""



if read_foam:
    # reader
    print("****reading foam results****")
    print("converting foam to vtk")
    reader = vtk.vtkOpenFOAMReader()
    reader.SetFileName(fileName)
    reader.CreateCellToPointOn()
    reader.DecomposePolyhedraOn()
    reader.EnableAllCellArrays()
    reader.Update()
    reader_original = reader
    print("****finished****")
    print("getting latest results")
    tArray = vtk_to_numpy(reader.GetTimeValues())
    if read_time_series:
        pass
    else:
        tArray = [tArray[-1]]

    list_csv_path1 = []
    list_csv_path2 = []
    for time in tArray:
        reader = reader_original
        reader.UpdateTimeStep(time) #latest time 
        print(f"reading time is {time}")       
        reader.Update() 
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

        if var == "U":
            u = vtk_to_numpy(cell2point.GetOutput().GetPointData().GetArray(var)) #U
            u_x = u[:,0]    
            u_y = u[:,1]    
            u_z = u[:,2]    
            u_mag = np.sqrt(u_x**2 + u_y**2 + u_z**2)

            ###############################
            dict1=dict(node_id=node_number,
                       x=x,
                       y=y,
                       z=z,
                       u_x=u_x,
                       u_y=u_y,
                       u_z=u_z,
                       u_mag=u_mag)
        else:
            print(f"{var}\n対応していない物理量です\nコードを書き換えてください")
            raise ValueError
        df1 = pd.DataFrame(data=dict1)
        csv_path1 = add_timeinfo_path(csv_path1_original,time)
        list_csv_path1.append(csv_path1)
        df1.to_csv(csv_path1,index=False)
        ################################
        dict2=dict(cell_id=cell_id_list,
                   cell_node_ids=cell_node_indices_list)
        df2 = pd.DataFrame(data=dict2)
        csv_path2 = add_timeinfo_path(csv_path2_original,time)
        list_csv_path2.append(csv_path2)
        df2.to_csv(csv_path2,index=False)
        ###############################
    
    
if get_csv:
    df = pd.read_csv(list_csv_path1[-1])
    x = df["x"].to_list()
    y = df["y"].to_list()
    z = df["z"].to_list()
    u_mag = df["u_mag"].to_list()
    df_xsort = df.sort_values("x")
    df_ysort = df.sort_values("y")
    df_zsort = df.sort_values("z")
    #z座標が同じものを取得
    x_list = df_xsort["x"].to_list()
    y_list = df_ysort["y"].to_list()
    z_list = df_zsort["z"].to_list()
    
    x_min, x_max = np.min(x_list), np.max(x_list)
    y_min, y_max = np.min(y_list), np.max(y_list)
    
    if View3D:
        if read_time_series:
            df_concat = visualization3d_series(list_csv_path1,tArray,f_time) #df
            df_concat.sort_values('t').to_csv(csv_path4_original,index=None)
            print(f"all time csv was converted in {csv_path4}")
        else:    
            visualization3d(df) #df
            if not os.listdir(save_dir):
                os.rmdir(save_dir)

    else:
        df_list =  []
        z_list_df = []
        z_list = [round(z,1) for z in z_list]
        z_list = list(dict.fromkeys(z_list))

        for z_val in z_list:
            dfs = df_zsort[(df_zsort[axis]<=z_val+order)&(df_zsort[axis]>=z_val-order)]
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
            print(f"z = {z}")
            nodes = dfa["node_id"].to_list() #target nodes
            #nodesの値のみで構成される節点番号をcell node indices listから抽出したい
            filtered_df = df_cellnode[df_cellnode['cell_node_ids'].apply(lambda x: is_subset(x, nodes))]
            #filtered_dfのcell_node_idsの値を1から振りなおす
            #全ての値を取得して一意にする
            unique_values = sorted(set(val for sublist in filtered_df['cell_node_ids'] for val in sublist))
            #新しい番号を割り当てる辞書を作成
            new_value_map = {val: idx for idx, val in enumerate(unique_values)}
            #データフレーム内のリストを置き換える
            filtered_df['cell_node_ids'] = filtered_df['cell_node_ids'].apply(lambda x: replace_values(x, new_value_map))
            #print(filtered_df)
            filtered_df.to_csv(df_rootDir+f"filtered_df_z{z}_order{order}.csv",index=False)
            cellnode_list = filtered_df['cell_node_ids'].to_list()
            #dfaをnode_id列を基準にして昇順に並べなおす->dfaa
            dfaa = dfa.sort_values(by="node_id")
            x       = dfaa["x"].to_list()
            y       = dfaa["y"].to_list() 
            u_mag   = dfaa['u_mag'].to_list()
            dfaa.to_csv(df_rootDir+f"dfaa_z{z}_order{order}.csv",index=False)
            visualization2d(x,y,u_mag,x_min,x_max,y_min,y_max,df=dfaa,figsize=(6,5))


