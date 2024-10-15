import os
import pandas as pd
import shutil

# testディレクトリのパス
# test_dir = "/src/openhouse2024/openhouse2024_competition/eval_data"
test_dir = ''

# 変更後のディレクトリの作成
# testv2に変更ごの内容を保存する
# save_dir = "/src/openhouse2024/openhouse2024_competition/eval_datav2"
save_dir = ''

# 既にディレクトリが作成されている場合は作成処理を行わない
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)

# images_info.csvの読み込み
img_data = pd.read_csv(test_dir + '/images_info.csv', header = None)

classes = sorted(img_data[1].unique())

for class_name in classes:

    # class_name毎のディレクトリのパス
    class_dir = save_dir + "/" + class_name

    # 既にディレクトリが作成されている場合は作成処理を行わない
    if os.path.exists(class_dir):
        pass
    else:
        os.mkdir(class_dir)

    # class_nameごとのimageファイルの抽出とリスト化
    img_list = img_data[img_data[1] == class_name][0].to_list()

    # 該当の画像をtestv2の各クラスディレクトリにコピー
    for img in img_list:
        shutil.copy(test_dir + "/images/" + img, class_dir)