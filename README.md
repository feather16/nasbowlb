# nasbowlb
卒業研究の実験のために書いたソースコードをリファクタリングしたものです。<br>
全てのコードを1から書きました。<br>
(自動生成されるコードである[cython_wl_kernel.cpp](cython_wl_kernel.cpp)を除く)

論文[Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel](https://arxiv.org/abs/2006.07556v1)をベースとしています。

卒業研究の際に書いたコードは[こちら]
(https://github.com/feather16/UndergraduateResearch)

# 実行環境
- CentOS 7.9-2009
- Python 3.10.7
- GCC 4.8.5

# 必要なモジュール
- torch
- Cython
- numpy 
- matplotlib
- yaml
- requests
- nats_bench
- tqdm

# ソースコードの説明
|ファイル名|内容|
|-|-|
|[nasbowl.py](nasbowl.py)|メインプログラム|
|[GPWithWLKernel.py](GPWithWLKernel.py)|実験の本体|
|[NATSBenchCell.py](NATSBenchCell.py)|[NATS-Bench](https://github.com/D-X-Y/NATS-Bench)のニューラルネットワークに対応するクラス|
|[NATSBenchWrapper.py](NATSBenchWrapper.py)|[NATS-Bench](https://github.com/D-X-Y/NATS-Bench)の探索空間に対応するクラス|
|[CachedKernel.py](CachedKernel.py)|カーネルの計算結果をキャッシュする|
|[util.py](util.py)|共通で用いる関数|
|[Timer.py](Timer.py)|実行時間を計測するタイマー|
|[cython_setup.py](cython_setup.py)|[Cythonのコード](cython_wl_kernel.pyx)と[C++実装のWLカーネル](wl_kernel_impl.cpp)をPython向けにコンパイルする|
|[cython_wl_kernel.cpp](cython_wl_kernel.cpp)|Cythonによって自動生成されたWLカーネルのC++プログラム|
|[cython_wl_kernel.pyx](cython_wl_kernel.pyx)|WLカーネルをCython経由でPythonから呼び出せるようにする|
|[CythonWLKernel.py](CythonWLKernel.py)|WLカーネルをCython経由でPythonから呼び出す|
|[wl_kernel_impl.cpp](wl_kernel_impl.cpp)|WLカーネルのC++実装|
|[wl_kernel_impl.hpp](wl_kernel_impl.hpp)|[wl_kernel_impl.cpp](wl_kernel_impl.cpp)に対応するヘッダファイル|
|[Config.py](Config.py)|実験のパラメータ|
|[create_kernel_cache.py](create_kernel_cache.py)|WLカーネルの値をキャッシュとしてファイルに保存する|
|[test.py](test.py)|WLカーネルの実行時間テスト用|
|[submit.py](submit.py)|Slurmにジョブを投入|
|[Log.py](Log.py)|プログラムの実行結果解析|
|[show_commands.py](show_commands.py)|過去に実行したプログラムの入力コマンドを一覧表示|
|[show_execution_time.py](show_execution_time.py)|過去に実行したプログラムの実行時間を一覧表示|
|[visualize.py](visualize.py)|過去に実行したプログラムの実行結果を可視化|
|[check_errors.py](check_errors.py)|プログラムの実行結果に対しエラーが無いかチェック|

# 実行例

## 画像分類性能の測定
`python3 nasbowl.py acc -T 1500`

## スピアマンの順位相関係数の測定
`python3 nasbowl.py srcc -T 1500`

## 実行時間の測定
`python3 nasbowl.py time -T 1500`