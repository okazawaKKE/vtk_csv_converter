# vtk_csv_converter
This is for converting openfoam result from simscale to csv file. We can get values from GUI interface.

main : project_folder
project_folderの直下にsimscaleの結果を保存する

・ ./src/readFOAM.py
可視化などの処理をするためのライブラリ

・./src/rotation_df.py
dataframeの回転座標系対応のためのライブラリ

・ ./convert.py
実行用メインスクリプト
カレントディレクトリのopenfoamの結果フォルダを読み込み可視化する

・ ./settings.py
convert.pyを実行する際に使う、パスや引数などの記述
最初に可視化のオプションをここで設定する


***必要な外部ライブラリ***
・matplotlib
・numpy
・vtk
・pandas
・scipy


・convert.py
read_foamがTrueの時，vtkからcsvの出力を行う
get_csvがTrueの時,csvから可視化とGUI操作を行う
（settings.py内の値を参照）

read_time_seriesがTrueの時，Visualization3d_series()を呼び出す
read_time_seriesがFalseの時，Visualization3d()を呼び出す
これら２つの関数は./src/readFOAM.pyから参照

![Visualization3d](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/7f6e5c2e-7b1e-45ae-99e3-cfdf0bb2000b)

![Visualization3d_series](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/58c51693-3a56-4ed3-8e56-44bf170bd250)


  
![image](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/e951be61-24b4-4169-b718-d22b3ea94217)


![image](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/fd582b8a-116d-4818-a4a6-f5c531cb3100)
![image](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/4fca48e7-165c-433d-aec6-60fbb1cf4855)

![image](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/8f915e8d-57c8-4709-ae1f-da74a65f8fcc)
![image](https://github.com/okazawaKKE/vtk_csv_converter/assets/171329398/7c15a8eb-8466-4310-b220-e29ca48d9d72)







