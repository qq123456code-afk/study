from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        label = 0 if 'cat' in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label

def get_loaders(data_dir="./datasets", batch_size=16, train_transform=None, seed=42):
    """统一划分：80% 训练 / 20% 验证，固定随机种子保证各实验可比"""
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = FileNameDataset(image_dir=data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 分别给训练/验证套上不同的 transform（验证集不做增强）
    train_set = FileNameDataset(image_dir=data_dir, transform=train_transform)
    val_set   = FileNameDataset(image_dir=data_dir, transform=val_transform)
    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val = random_split(range(len(dataset)), [train_size, val_size], generator=g)

    train_loader = DataLoader(torch.utils.data.Subset(train_set, idx_train.indices),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(val_set, idx_val.indices),
                              batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    net.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader.dataset):.4f}")

    # 评估
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Y in val_loader:
            preds = net(X.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(Y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                  average='macro', zero_division=0)
    print(f"准确率: {acc:.4f} | 精确率: {p:.4f} | 召回率: {r:.4f} | F1: {f1:.4f}")
    return acc