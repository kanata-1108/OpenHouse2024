import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from main import Net
import pandas as pd

# 画像データを番号順にソートする関数
def sort_key(fname):
    return int(''.join(filter(str.isdigit, fname)))

# カスタムデータセット
class CustomImageDataset(Dataset):
    def __init__(self, dir_path, transform, infocsv_path, class_to_idx):
        self.img_paths = sorted([os.path.join(dir_path, fname) for fname in os.listdir(dir_path)], key = sort_key)
        self.transform = transform
        self.infocsv_path = infocsv_path
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        true_file = self.infocsv_path
        true_df = pd.read_csv(true_file, names = ['img', 'label'], header = None)
        labels = true_df.iloc[:, 1].to_numpy()
        labels = [self.class_to_idx[label] for label in labels]

        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        label = labels[index]
        
        if self.transform:
            img = self.transform(img)

        return img, label

# 評価用関数
def evaluation(net_model, loader):
    sum_correct = 0

    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_model(inputs)
            _, predict_label = outputs.max(axis = 1)
            sum_correct += (predict_label == labels).sum().item()
        
        accuracy = sum_correct / len(loader.dataset)

    print(f"正解率: {accuracy * 100} %")

if __name__ == "__main__":

    # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # main.pyの親ディレクトリのフルパス(パスをいちいち書き換える必要がなくなる)
    dir_fullpath = os.path.dirname(__file__)

    # eval_dataのパス
    eval_dir = dir_fullpath + '/openhouse2024_competition/eval_data'

    # データの処理
    transform_eval = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    class_to_idxes = {'あ': 0, 'い': 1, 'お': 2, 'に': 3, 'ぬ': 4, 'ね': 5, 'は': 6, 'め': 7, 'れ': 8, 'ろ': 9}
    eval_dataset = CustomImageDataset(dir_path = eval_dir + '/images', transform = transform_eval, infocsv_path = eval_dir + '/images_info.csv', class_to_idx = class_to_idxes)
    eval_loader = DataLoader(eval_dataset, batch_size = len(eval_dataset.img_paths))

    # パラメータの読み込みとモデルインスタンスの作成
    modelweight_path = dir_fullpath + '/model_weight/best_model.pth'
    eval_model = Net()
    eval_model.load_state_dict(torch.load(modelweight_path, weights_only = True))
    eval_model = eval_model.to(device)
    eval_model.eval()

    # 推論&精度算出
    pred = evaluation(eval_model, eval_loader)