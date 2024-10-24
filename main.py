import os
import pandas as pd
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

    sum_loss = 0
    sum_correct = 0

    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predict_label = outputs.max(axis = 1)
            sum_correct += (predict_label == labels).sum().item()

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
            nn.Dropout(p = 0.3),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(input_dim = 16, output_dim = 32),
            ResidualBlock(input_dim = 32, output_dim = 32),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.3),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(input_dim = 32, output_dim = 64),
            ResidualBlock(input_dim = 64, output_dim = 64),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.3),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 15 * 15, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(256, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU()
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
        transforms.RandomInvert(p = 0.5),
        transforms.Resize(128),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    # class_to_idxesの内容 -> {'あ': 0, 'い': 1, 'お': 2, 'に': 3, 'ぬ': 4, 'ね': 5, 'は': 6, 'め': 7, 'れ': 8, 'ろ': 9}
    class_to_idxes = train_dataset.class_to_idx
    valid_dataset = CustomImageDataset(dir_path = valid_dir + '/images', transform = transform_valid, infocsv_path = valid_dir + '/images_info.csv', class_to_idx = class_to_idxes)

    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = 8, shuffle = False)
    
    # 学習の設定
    epochs = 1000
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.005)
    optimizer = optim.Adam(model.parameters(), weight_decay = 0.0001)
    scheduler = CosineLRScheduler(optimizer, t_initial = epochs, lr_min = 0.0001, warmup_t = 50, warmup_lr_init = 0.00005, warmup_prefix = True)

    # 結果を格納するリスト
    train_loss_value = []
    train_acc_value = []
    valid_loss_value = []
    valid_acc_value = []

    # 最小の損失値
    best_loss = float('inf')

    # 学習
    for epoch in range(epochs):

        model.train()

        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step(epoch + 1)

        model.eval()
        train_loss, train_acc = evaluation(model, train_loader)
        valid_loss, valid_acc = evaluation(model, valid_loader)

        print(f"[{epoch + 1}/{epochs}] :: train loss: {train_loss:.5f}, train acc: {train_acc:.5f}, valid loss: {valid_loss:.5f}, valid acc: {valid_acc:.5f}")

        train_loss_value.append(train_loss)
        train_acc_value.append(train_acc)
        valid_loss_value.append(valid_loss)
        valid_acc_value.append(valid_acc)

        # 学習回数が900回以上かつ、検証スコアが高いモデルを保存する
        if epoch > 900 and valid_loss < best_loss:
            model_pram = model.state_dict()
            torch.save(model.state_dict(), dir_fullpath + '/model_weight/best_model.pth')
            best_loss = valid_loss

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