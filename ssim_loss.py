import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.utils.tensorboard import SummaryWriter

EPSILON = 1e-8
writer = SummaryWriter(log_dir="runs/ssim_loss")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq + EPSILON
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq + EPSILON
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1 + EPSILON) * (sigma1_sq + sigma2_sq + C2 + EPSILON))
    return ssim_map.mean() if size_average else ssim_map

class SSIML1RegLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, alpha=0.7, beta=0.15, gamma=0.1, texture_weight=0.2, reg_lambda=1e-4):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.texture_weight = texture_weight
        self.reg_lambda = reg_lambda
        self.channel = 1
        self.window = None
        self.step = 0

    def texture_loss(self, pred, target_a):
        # 使用 Sobel 算子提取纹理特征
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 提取梯度特征
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_a, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_a, sobel_y, padding=1)
        
        # 计算纹理损失
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        return (grad_diff_x.mean() + grad_diff_y.mean()) / 2

    def ssim_loss(self, pred, target1, target2):
        if self.window is None or self.channel != pred.size(1) or self.window.device != pred.device:
            self.window = create_window(self.window_size, pred.size(1)).to(pred.device).type(pred.dtype)
            self.channel = pred.size(1)

        ssim1 = 1 - ssim(pred, target1, self.window, self.window_size, self.channel, self.size_average)
        ssim2 = 1 - ssim(pred, target2, self.window, self.window_size, self.channel, self.size_average)
        return (ssim1 + ssim2) / 2

    def l1_loss(self, pred, target1, target2):
        l1_1 = F.l1_loss(pred, target1)
        l1_2 = F.l1_loss(pred, target2)
        return (l1_1 + l1_2) / 2

    def contrast_loss(self, pred):
        mean = pred.mean(dim=[2, 3], keepdim=True)
        std = torch.sqrt(((pred - mean) ** 2).mean(dim=[2, 3], keepdim=True) + EPSILON)
        contrast = std.mean()
        return torch.relu(1 - contrast)  # 确保 contrast_loss ≥ 0


    def reg_loss(self, model):
        return sum(torch.norm(p, 2) for p in model.parameters())

    def forward(self, pred, target1, target2, model, return_all=False):
        assert pred.shape == target1.shape == target2.shape, f"Shape mismatch: {pred.shape}, {target1.shape}, {target2.shape}"

        ssim_l = self.ssim_loss(pred, target1, target2)
        l1_l = self.l1_loss(pred, target1, target2)
        contrast_l = self.contrast_loss(pred)
        texture_l = self.texture_loss(pred, target1)  # 使用输入a作为纹理参考
        reg_l = self.reg_loss(model)

        total_loss = (self.alpha * ssim_l + 
                     self.beta * l1_l + 
                     self.gamma * contrast_l + 
                     self.texture_weight * texture_l +
                     self.reg_lambda * reg_l)

        writer.add_scalar("Loss/SSIM", ssim_l.item(), self.step)
        writer.add_scalar("Loss/L1", l1_l.item(), self.step)
        writer.add_scalar("Loss/Contrast", contrast_l.item(), self.step)
        writer.add_scalar("Loss/Texture", texture_l.item(), self.step)
        writer.add_scalar("Loss/Reg", reg_l.item(), self.step)
        writer.add_scalar("Loss/Total", total_loss.item(), self.step)
        self.step += 1

        if return_all:
            return total_loss, ssim_l, l1_l, contrast_l, texture_l, reg_l
        return total_loss
