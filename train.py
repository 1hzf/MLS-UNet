import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from dataload import FusionDataset
from ssim_loss import SSIML1RegLoss
from tqdm import tqdm
import os
import argparse
import logging
import cv2
import numpy as np

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def test_and_save_fusion(model, test_loader, device, result_dir, epoch):
    """
    使用当前模型对指定路径下的图片进行融合并保存。
    """
    model.eval()
    os.makedirs(result_dir, exist_ok=True)  # 确保结果目录存在

    with torch.no_grad():
        for image_idx, (input_a, input_b) in enumerate(test_loader):
            input_a, input_b = input_a.to(device), input_b.to(device)
            output = model(input_a, input_b)

            # 如果模型输出是元组，取第一个元素
            if isinstance(output, tuple):
                output = output[0]

            # 转换输出为 numpy 数组并保存为图像
            output_np = output.detach().cpu().numpy()[0, 0]
            save_img = (output_np * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(result_dir, f"epoch_{epoch:03d}_fused_{image_idx+1:03d}.png")
            cv2.imwrite(save_path, save_img)

    logging.info(f"✅ Fused images for epoch {epoch} have been saved to {result_dir}/")

def parse_args():
    parser = argparse.ArgumentParser(description='Train the UNet model')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint file to resume training from')
    return parser.parse_args()

def save_model(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def train_epoch(model, criterion, optimizer, train_loader, device, writer, epoch):
    model.train()
    total_loss = 0
    ssim_epoch, l1_epoch, contrast_epoch, texture_epoch, reg_epoch = 0, 0, 0, 0, 0  # 修改这里

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/300")
    for input_a, input_b in pbar:
        # 确保输入数据在正确的设备上
        input_a, input_b = input_a.to(device), input_b.to(device)
        target1, target2 = input_a, input_b

        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(input_a, input_b)
        
        # 计算损失（包含所有组件）
        loss, ssim_loss, l1_loss, contrast_loss, texture_loss, reg_loss = criterion(
            output, target1, target2, model, return_all=True
        )  # 修改这里

        # 确保损失是标量
        loss = loss.mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()

        # 累积损失值
        total_loss += loss.item()
        ssim_epoch += ssim_loss.item()
        l1_epoch += l1_loss.item()
        contrast_epoch += contrast_loss.item()
        texture_epoch += texture_loss.item()  # 添加这行
        reg_epoch += reg_loss.item()

        # 更新进度条
        pbar.set_postfix({
            'total_loss': f'{loss.item():.4f}',
            'ssim': f'{ssim_loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'contrast': f'{contrast_loss.item():.4f}',
            'texture': f'{texture_loss.item():.4f}',  # 添加这行
            'reg': f'{reg_loss.item():.4f}'
        })

    # 计算平均损失
    num_batches = len(train_loader)
    avg_total_loss = total_loss / num_batches
    avg_ssim_loss = ssim_epoch / num_batches
    avg_l1_loss = l1_epoch / num_batches
    avg_contrast_loss = contrast_epoch / num_batches
    avg_texture_loss = texture_epoch / num_batches  # 添加这行
    avg_reg_loss = reg_epoch / num_batches

    # 记录到 TensorBoard
    writer.add_scalar("Loss/train_total", avg_total_loss, epoch)
    writer.add_scalar("Loss/train_ssim", avg_ssim_loss, epoch)
    writer.add_scalar("Loss/train_l1", avg_l1_loss, epoch)
    writer.add_scalar("Loss/train_contrast", avg_contrast_loss, epoch)
    writer.add_scalar("Loss/train_texture", avg_texture_loss, epoch)  # 添加这行
    writer.add_scalar("Loss/train_reg", avg_reg_loss, epoch)

    return avg_total_loss

def validate_epoch(model, criterion, val_loader, device, writer, epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_a, input_b in val_loader:
            input_a, input_b = input_a.to(device, non_blocking=True), input_b.to(device, non_blocking=True)
            target1, target2 = input_a, input_b
            output = model(input_a, input_b)
            loss, _, _, _, _, _ = criterion(output, target1, target2, model, return_all=True)  # 修改这里
            val_loss += loss.mean().item()
    
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    return avg_val_loss
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 370
    batch_size = 2  # 降低 batch size 以减少显存占用
    learning_rate = 1e-4
    
    checkpoint_path = "checkpoints/best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    data_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FusionDataset("dataset/train", data_transform)
    val_dataset = FusionDataset("dataset/val", data_transform)
    # test_dataset = FusionDataset("dataset/test-MSP", data_transform)  # 指定测试数据集路径
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试数据集加载器

    model = UNet(in_channels=2, out_channels=1, use_swin=True).to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    criterion = SSIML1RegLoss().to(device)
    writer = SummaryWriter("runs/unet_fusion")
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    best_val_loss = float('inf')
    result_dir = "./sample_output"

    for epoch in range(epochs):
        train_loss = train_epoch(model, criterion, optimizer, train_loader, device, writer, epoch)
        val_loss = validate_epoch(model, criterion, val_loader, device, writer, epoch)
        scheduler.step(val_loss)
        
        save_model(model, optimizer, scheduler, epoch, train_loss, val_loss, f"checkpoints/model_epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_path)

        # test_and_save_fusion(model, test_loader, device, result_dir, epoch) 
    logging.info("Training completed.")

if __name__ == "__main__":
    main()