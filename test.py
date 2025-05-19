# test_unet.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from model import UNet
from dataload import FusionDataset
from thop import profile
import time

def calculate_model_complexity(model, valid_sourceImg1, valid_sourceImg2, device):
    """计算模型复杂度"""
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x1, x2):  # 修改为接收两个输入
            return self.model(x1, x2)
    
    wrapped_model = ModelWrapper(model)
    
    # 确保输入的维度一致
    if valid_sourceImg1.dim() != valid_sourceImg2.dim():
        raise ValueError("输入张量的维度不一致！")

    # 使用两个输入计算复杂度
    macs, params = profile(wrapped_model, inputs=(valid_sourceImg1, valid_sourceImg2))
    
    print("\n模型复杂度统计:")
    print(f"MACs: {macs / 1e9:.4f} G")
    print(f"FLOPs: {macs * 2 / 1e9:.4f} G")  # FLOPs ≈ 2 * MACs
    print(f"参数量: {params / 1e6:.4f} M")
    
    flops = macs * 2  # FLOPs 计算
    return flops, params

def generate_random_test_data(batch_size=1, channels=3, height=520, width=520, device="cuda"):
    """生成随机测试数据"""
    random_img1 = torch.randn(batch_size, channels, height, width).to(device)
    random_img2 = torch.randn(batch_size, channels, height, width).to(device)
    return random_img1, random_img2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练后保存的最佳模型

model = UNet(in_channels=2, out_channels=1).to(device)
checkpoint_path = "best.pt"  # <-- 替换为实际保存模型路径
checkpoint = torch.load(checkpoint_path, map_location=device)

# 自动兼容DataParallel模型保存
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# 加载测试集
data_transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = FusionDataset("dataset/test-GAN", data_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建保存融合图像目录
os.makedirs("base", exist_ok=True)

# 测试并保存融合图像
with torch.no_grad():
    for idx, (input_a, input_b) in enumerate(test_loader):
        input_a, input_b = input_a.to(device), input_b.to(device)
        start_time = time.time()
        output = model(input_a, input_b)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # flops, params = profile(model, inputs=(input_a, input_b), verbose=False)
        # print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
        # print(f"参数量: {params / 1e6:.4f} M")
        # print(f"  - Time:   {elapsed_time:.2f} S")
        # flops, params = calculate_model_complexity(model, input_a, input_b, device)

        output_np = output.detach().cpu().numpy()[0, 0]
        save_img = (output_np * 255).clip(0, 255).astype(np.uint8)
        save_path = f"./result-SOTA-GAN/fused_{idx+1:03d}.png"
        cv2.imwrite(save_path, save_img)

print("✅ 所有融合图像已保存至 results/ 目录。")
