import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD

# 训练集增强（和step4保持一致）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if __name__ == '__main__':
    train_loader, val_loader = get_loaders("./datasets", batch_size=16, train_transform=train_transform)
    #选用效果最好的mobilenet_v3
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features,2)
    for name, param in net.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    #训练
    acc = train_and_eval(net, train_loader, val_loader, epochs=10, lr=1e-4, device="cpu")
    print(f"MobileNetV3 验证集准确率：{acc:.4f}")

    #保存模型参数（推荐方式，体积小）
    torch.save(net.state_dict(), "best_model.pth")
    print("✅模型已保存至 best_model.pth")

    # ========== 单图推理测试 ==========
    # 推理用的transform，不要随机增强！
    infer_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    #加载保存好的权重
    net2 = models.mobilenet_v3_small(weights=None)
    net2.classifier[-1] = nn.Linear(net2.classifier[-1].in_features,2)
    net2.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    net2.eval()

    #测试图片，改成你自己图片路径
    img_path = "tets.jpg"
    image = Image.open(img_path).convert("RGB")
    img_tensor = infer_transform(image).unsqueeze(0)

    with torch.no_grad():
        out = net2(img_tensor)
        _, pred = torch.max(out, dim=1)
    class_names = ["猫","狗"]
    print(f"\n📸单张图片预测结果：{class_names[pred.item()]}")

