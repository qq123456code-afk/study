import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# 1. 加载训练好的模型（启动时只加载一次）
net = models.mobilenet_v3_small(weights=None)
net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
net.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
net.eval()

CLASSES = {0: "🐱 猫", 1: "🐶 狗"}

# 2. 推理预处理（固定，不做随机增强）
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. 预测函数
def predict(img: Image.Image):
    if img is None:
        return "请先上传图片"
    img = img.convert("RGB")
    tensor = infer_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = net(tensor)
        prob = torch.softmax(out, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_idx].item()
    return f"预测结果：{CLASSES[pred_idx]}（置信度 {confidence:.2%}）"

# 4. 启动网页
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传猫狗图片"),
    outputs="text",
    title="🐱🐶 猫狗图像分类系统",
    description="上传一张猫或狗的图片，模型自动识别类别",
)

if __name__ == "__main__":
    demo.launch()
