import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# ===================== 自定义数据集（和step1保持一致） =====================
class DatasetLoader(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # cat=0 dog=1
        if "cat" in img_name.lower():
            label = 0
        else:
            label = 1
        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== 训练参数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr = 1e-4
epoch_num = 10
seed = 42

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

data_dir = "./datasets"
full_dataset = DatasetLoader(image_dir=data_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ===================== 迁移学习：加载预训练AlexNet =====================
net = models.alexnet(pretrained=True)
# 冻结特征提取层
for param in net.features.parameters():
    param.requires_grad = False
# 修改最后的分类头，二分类
net.classifier[6] = nn.Linear(4096,2)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=lr)

# ===================== 训练循环 =====================
print("===== 迁移学习 AlexNet 开始训练 =====")
for epoch in range(epoch_num):
    net.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        trainer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        trainer.step()
        train_loss += loss.item()

    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = net(imgs)
            _, pred = torch.max(out, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = correct / total
    print(f"Epoch[{epoch+1}/{epoch_num}] | TrainLoss:{train_loss:.3f} | Val Acc:{acc:.3f}")
