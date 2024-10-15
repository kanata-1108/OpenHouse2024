import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datetime import datetime
from zoneinfo import ZoneInfo
from main import Net

def evaluation(net_model, loader):

    sum_correct = 0

    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device, torch.float32), labels.to(device)
            outputs = net_model(inputs)
            _, predict_label = outputs.max(axis = 1)
            sum_correct += (predict_label == labels).sum().item()
    
    accuracy = sum_correct / len(loader.dataset)

    return accuracy

now_date = datetime.now(ZoneInfo("Asia/Tokyo"))
print(f"-----{now_date}-----")

dir_path = '/src/openhouse2024'
os.chdir(dir_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データの処理
eval_dir = './openhouse2024_competition/eval_datav2'

transform_eval = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

eval_dataset = ImageFolder(root = eval_dir, transform = transform_eval)
# GPUを使う場合メモリ不足になることは少ないからbatch_size = len(eval_dataset.samples)でOK
eval_loader = DataLoader(eval_dataset, batch_size = 8)

# 推論処理
modelweight_path = "model_weight/2024-10-15-14-09.pth"
eval_model = Net()
# GPUを使う場合map_location = torch.device("cpu")はいらない
eval_model.load_state_dict(torch.load(modelweight_path, weights_only = True, map_location = torch.device("cpu")))
eval_model = eval_model.to(device, torch.float32)
eval_model.eval()

acc = evaluation(eval_model, eval_loader)
print(f"Accuracy: {acc * 100} %")