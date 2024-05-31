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
