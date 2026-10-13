from torchvision import transforms
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD
from torchvision import models
import torch.nn as nn

# ========== 训练集数据增强组合 ==========
train_transform = transforms.Compose([
 transforms.Resize((256, 256)),
 transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), #随机裁剪缩放
 transforms.RandomHorizontalFlip(), #随机水平翻转
 transforms.ColorJitter(0.2, 0.2, 0.2), #亮度、对比度、饱和度扰动
 transforms.RandomRotation(10), #小角度旋转
 transforms.ToTensor(),
 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if __name__ == '__main__':
 # 使用带增强的train_transform，验证集transform写在common内部不用管
 train_loader, val_loader = get_loaders("./datasets", batch_size=16,
 train_transform=train_transform)

 # 加载AlexNet预训练，冻结特征层，只训练分类头（和step2保持一致）
 net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
 net.classifier[6] = nn.Linear(4096, 2)

 #冻结卷积层，只训练classifier分类头
 for name, param in net.named_parameters():
     param.requires_grad = name.startswith("classifier")

 #调用common里面封装好的训练评估函数
 acc = train_and_eval(net, train_loader, val_loader, epochs=10, lr=1e-4, device="cpu")
 print(f"\n【第三步 预训练 + 数据增强】最终验证集准确率: {acc:.4f}")

