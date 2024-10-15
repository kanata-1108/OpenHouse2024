## OpenHouse2024 Competition
### How to use
main.pyと同じ階層でデータ格納用ディレクトリを作成
```
mkdir openhouse2024_competition
```
### 各種プログラムについて
**main.py**
メインのプログラム。訓練データと検証データを使用

**eval.py**
推論用のプログラム。テストデータを使用

**residualblock.py**
モデルに使用している残差ブロックのプログラム

**make_valid.py**
データセットをtrainと同じ構造にするためのプログラム
