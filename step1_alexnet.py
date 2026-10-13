import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ===================== 权重初始化函数 =====================
def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

# ===================== 自定义数据集 =====================
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

        # 打标签 cat=0，dog=1
        if "cat" in img_name.lower():
            label = 0
        else:
            label = 1

        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== AlexNet网络定义 =====================
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ===================== 训练参数设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr = 1e-4
epoch_num = 10
seed = 42

# 图像预处理
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

# 验证集不用增强
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ===================== 模型、损失、优化器【重点！顺序不能乱】 =====================
net = AlexNet()                 # 1.创建网络实例
net.apply(init_weight)          # 2.权重初始化
net = net.to(device)

criterion = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=lr) #3.优化器

# ===================== 训练循环 =====================
print("===== 开始训练 =====")
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

    # 验证
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
