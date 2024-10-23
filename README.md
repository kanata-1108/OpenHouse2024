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
上のディレクトリ構造通りに配置すれば動作するはず(ディレクトリ名も同じにする必要あり)。

## 各種プログラムについて
### main.py

メインのプログラム。

### eval.py

推論用のプログラム。

### residualblock.py

モデルに使用している残差ブロックのプログラム

## 結果
![acc](https://github.com/user-attachments/assets/e6cb9e40-e477-480a-95c0-57fec469c919)

![loss](https://github.com/user-attachments/assets/74726163-e70a-457c-9b04-d56ab737e520)

推論時のスコア：81.8%