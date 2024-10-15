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

### 結果
![](vscode-remote://attached-container%2B7b22636f6e7461696e65724e616d65223a222f707974686f6e335f6c6f63616c222c2273657474696e6773223a7b22636f6e74657874223a226465736b746f702d6c696e7578227d7d/src/openhouse2024/result/acc.png)