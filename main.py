import os
import pandas as pd
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from residualblock import ResidualBlock
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

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

# 評価用関数
def evaluation(net_model, loader, train_flag, infocsv_path, class_to_idx):

    sum_loss = 0
    sum_correct = 0

    with torch.no_grad():

        # trainデータとvalidデータでラベルの有無が存在するため、処理を分割
        if train_flag:
            for (inputs, labels) in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net_model(inputs)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                _, predict_label = outputs.max(axis = 1)
                sum_correct += (predict_label == labels).sum().item()
        else:
            # images_info.csvからラベルデータを取得して、数値データに変換している
            # 数値データに変換することによってvalidデータに対する誤差と精度を算出できるようになる
            true_file = infocsv_path
            true_df = pd.read_csv(true_file, names = ['img', 'label'], header = None)
            true_label = true_df.iloc[:, 1].to_numpy()
            true_label = [class_to_idx[label] for label in true_label]
            true_label = torch.tensor(true_label).to(device)

            for inputs in loader:
                inputs = inputs.to(device)
                outputs = net_model(inputs)
                loss = criterion(outputs, true_label)
                sum_loss += loss.item()
                _, predict_label = outputs.max(axis = 1)
                sum_correct += (predict_label == true_label).sum().item()

        mean_loss = sum_loss / len(loader.dataset)
        accuracy = sum_correct / len(loader.dataset)

    return mean_loss, accuracy

# モデル構造の定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 5, padding = 1),
            ResidualBlock(input_dim = 8, output_dim = 16),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.3)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(input_dim = 16, output_dim = 32),
            ResidualBlock(input_dim = 32, output_dim = 32),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.3)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(input_dim = 32, output_dim = 64),
            ResidualBlock(input_dim = 64, output_dim = 64),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.3)
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 15 * 15, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(negative_slope = 0.01)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":

    # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # main.pyの親ディレクトリのフルパス(パスをいちいち書き換える必要がなくなる)
    dir_fullpath = os.path.dirname(__file__)

    # データの前処理
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomInvert(p = 0.4),
        transforms.Resize(128),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p = 0.3, scale = (0.02, 0.33), ratio = (0.3, 3.3))
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # trainとvalidのパス
    train_dir = dir_fullpath + '/openhouse2024_competition/train'
    valid_dir = dir_fullpath + '/openhouse2024_competition/test'

    # データセット＆データローダー作成
    train_dataset = ImageFolder(root = train_dir, transform = transform_train)
    valid_dataset = CustomImageDataset(dir_path = valid_dir, transform = transform_valid)

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = len(valid_dataset.img_paths), shuffle = False)
    
    # 学習の設定
    epochs = 1
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.005)
    optimizer = optim.Adam(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial = epochs, lr_min = 0.0001, warmup_t = 20, warmup_lr_init = 0.00005, warmup_prefix = True)

    # 結果を格納するリスト
    train_loss_value = []
    train_acc_value = []
    valid_loss_value = []
    valid_acc_value = []

    # 学習
    for epoch in range(epochs):

        model.train()

        for (inputs, labels) in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step(epoch + 1)

        model.eval()
        train_loss, train_acc = evaluation(model, train_loader, train_flag = True, infocsv_path = None, class_to_idx = None)
        valid_loss, valid_acc = evaluation(model, valid_loader, train_flag = False, infocsv_path = valid_dir + '/images_info.csv', class_to_idx = train_dataset.class_to_idx)

        print(f"[{epoch + 1}/{epochs}] :: train loss: {train_loss:.5f}, train acc: {train_acc:.5f}, valid loss: {valid_loss:.5f}, valid acc: {valid_acc:.5f}")

        train_loss_value.append(train_loss)
        train_acc_value.append(train_acc)
        valid_loss_value.append(valid_loss)
        valid_acc_value.append(valid_acc)

    # 結果を格納するディレクトリの作成
    result_savedir = dir_fullpath + '/result'
    if os.path.exists(result_savedir):
        pass
    else:
        os.mkdir(result_savedir)
    
    # 結果の描画
    plt.plot(range(epochs), train_loss_value, c = 'orange', label = 'train loss')
    plt.plot(range(epochs), valid_loss_value, c = 'blue', label = 'valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.savefig(result_savedir + '/loss.png')
    plt.clf()

    plt.plot(range(epochs), train_acc_value, c = 'orange', label = 'train acc')
    plt.plot(range(epochs), valid_acc_value, c = 'blue', label = 'valid acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid()
    plt.legend()
    plt.title('acc')
    plt.savefig(result_savedir + '/acc.png')

    # モデルの保存
    model_savedir = dir_fullpath + '/model_weight'
    if os.path.exists(model_savedir):
        pass
    else:
        os.mkdir(model_savedir)

    model_pram = model.state_dict()
    torch.save(model.state_dict(), model_savedir + '/best_model.pth')