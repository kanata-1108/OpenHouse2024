## ディレクトリ構造
```
OpenHouse2024
├── README.md
├── eval.py
├── main.py
├── residualblock.py
├── model_weight
│   └── best_model.pth
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
    ├── result.txt
    ├── acc.png
    └── loss.png
```
## How to use
上のディレクトリ構造通りに配置すれば動作はするはず(ディレクトリ名も同じにする必要あり)。

同じ構造にできたらあとはプログラムにディレクトリのパスを記述して、`main.py`を実行して学習させる。

学習後にモデルが`model_weight`ディレクトリに格納されるのでそれを`eval.py`で参照して実行すると推論結果が表示される。
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