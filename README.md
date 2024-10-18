## ディレクトリ構造
```
OpenHouse2024
├── README.md
├── eval.py
├── main.py
├── residualblock.py
├── model_weight
│   └── 2024-10-15-14-09.pth
├── openhouse2024_competition
│   ├── eval_data
│   │   ├── images
│   │   └── images_info.csv
│   ├── test
│   │   ├── images
│   │   └── images_info.csv
│   └── train
│       ├── あ
│       ├── い
│       ├── お
│       ├── に
│       ├── ぬ
│       ├── ね
│       ├── は
│       ├── め
│       ├── れ
│       └── ろ
└── result
    ├── acc.png
    └── loss.png
```
## How to use
上のディレクトリ構造通りに配置すれば動作はするはず。
main.pyと同じ階層でデータ格納用ディレクトリを作成
```
mkdir openhouse2024_competition
```
訓練データ、検証データ、テストデータを配置。訓練データと同じ構造（各ラベルごとにディレクトリが存在）にするため、make_valid.pyを実行する。この際ディレクトリのパスを変更する必要があるので注意

同じ構造にできたらあとはプログラムにディレクトリのパスを記述して、`main.py`を実行して学習させる。

学習後にモデルが`model`ディレクトリに格納されるのでそれを`eval.py`で参照して実行すると推論結果が表示される。
## 各種プログラムについて
### main.py

メインのプログラム。訓練データと検証データを使用

### eval.py

推論用のプログラム。テストデータを使用

### residualblock.py

モデルに使用している残差ブロックのプログラム

### make_valid.py

データセットをtrainと同じ構造にするためのプログラム

## 結果
![acc](https://github.com/user-attachments/assets/c60ec00d-fe56-4297-b4e8-65a83481591e)

![loss](https://github.com/user-attachments/assets/dea45a72-7994-473e-8863-0a4b95ba929e)

推論時のスコア：77.7%