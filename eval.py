import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from main import Net
import pandas as pd

# カスタムデータセット
class CustomImageDataset(Dataset):
    def __init__(self, dir_path, transform):
        self.img_paths = glob(os.path.join(dir_path, 'images/*.png'))
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        return img

def evaluation(net_model, loader):

    pred_labels = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = net_model(inputs)
            _, predict_label = outputs.max(axis = 1)
            pred_labels.append(predict_label)
        
        pred_labels = torch.cat(pred_labels)
        pred_labels = pred_labels.tolist()

    return pred_labels

if __name__ == "__main__":

    # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # main.pyの親ディレクトリのフルパス(パスをいちいち書き換える必要がなくなる)
    dir_fullpath = os.path.dirname(__file__)

    # データの処理
    transform_eval = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    eval_dataset = CustomImageDataset(dir_path = dir_fullpath + '/openhouse2024_competition/eval_data', transform = transform_eval)
    # ローカルだとメモリ不足になるからバッチサイズを小さくする。GPUを使う場合メモリ不足になることはないのでbatch_size = len(eval_dataset.img_paths)でOK
    eval_loader = DataLoader(eval_dataset, batch_size = 8)

    # パラメータの読み込みとモデルインスタンスの作成
    modelweight_path = dir_fullpath + '/model_weight/2024-10-15-14-09.pth'
    eval_model = Net()
    # GPUを使う場合map_location = torch.device(device)はいらない
    eval_model.load_state_dict(torch.load(modelweight_path, weights_only = True, map_location = torch.device(device)))
    eval_model = eval_model.to(device)
    eval_model.eval()

    # 推論処理
    pred = evaluation(eval_model, eval_loader)

    # 出力が数値なので平仮名に直す
    class_index = {0: 'あ', 1: 'い', 2: 'お', 3: 'に', 4: 'ぬ', 5: 'ね', 6: 'は', 7: 'め', 8: 'れ', 9: 'ろ'}
    pred = [class_index[label] for label in pred]
    pred_df = pd.DataFrame(pred, columns = ["label"])

    # 正解ラベルの抽出
    true_file = dir_fullpath + '/openhouse2024_competition/eval_data/images_info.csv'
    true_df = pd.read_csv(true_file, names = ['img', 'label'], header=None)
    true_label = true_df.iloc[:, [1]]

    # 正解率の算出
    matches = (true_label == pred_df).all(axis = 1)

    num_matches = matches.sum()

    print(f"Accuracy: {num_matches / 10}")
    