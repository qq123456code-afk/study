from torchvision import transforms, models
import torch.nn as nn
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD

# 和step3保持一样的数据增强
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def build_model(model_name):
    if model_name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, 2)
        #冻结特征层，只训练最后的全连接
        for name, param in net.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif model_name == "mobilenet_v3":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features,2)
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    elif model_name == "shufflenet_v2":
        net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features,2)
        for name, param in net.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    return net

if __name__ == '__main__':
    train_loader, val_loader = get_loaders("./datasets", batch_size=16, train_transform=train_transform)
    model_list = ["resnet18","mobilenet_v3","shufflenet_v2"]

    for m_name in model_list:
        print(f"\n===== 正在训练模型：{m_name} =====")
        model = build_model(m_name)
        acc = train_and_eval(model, train_loader, val_loader, epochs=10, lr=1e-4, device="cpu")
        print(f">>> {m_name} 验证集准确率：{acc:.4f}")
