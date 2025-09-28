import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# 双卷积模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 使用 reflect 填充代替默认的零填充
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
# CBAM模块
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        reduction_channels = max(channels // reduction, 1)
        
        # 通道注意力
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, channels, 1)
        )
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out
        spatial = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        spatial_out = torch.sigmoid(self.spatial_conv(spatial))
        
        return x * spatial_out
# 纹理增强融合模块
class EnhancedTextureFusionBlock(nn.Module):
    def __init__(self, channels):
        super(EnhancedTextureFusionBlock, self).__init__()
        # 增强特征提取能力
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAMBlock(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        # 双路特征提取
        feat1 = self.relu(self.bn1(self.conv1(x)))
        feat2 = self.relu(self.bn2(self.conv2(feat1)))
        # 注意力增强
        feat = self.cbam(feat1 + feat2)
        return self.relu(feat + identity)
# 亮度分支模块
class BrightnessBranch(nn.Module):
    def __init__(self, in_channels):
        super(BrightnessBranch, self).__init__()
        # 使用更深的网络结构来提取亮度特征
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)  # 添加池化层
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)  # 添加池化层
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)  # 添加池化层
        
        # 多尺度亮度权重生成
        self.weight_64 = nn.Conv2d(64, 64, kernel_size=1)
        self.weight_128 = nn.Conv2d(128, 128, kernel_size=1)
        self.weight_256 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层特征提取（保持原始分辨率）
        feat1 = self.relu(self.bn1(self.conv1(x)))
        weight64 = self.sigmoid(self.weight_64(feat1))
        
        # 第二层特征提取（降采样一次）
        feat2 = self.pool1(feat1)
        feat2 = self.relu(self.bn2(self.conv2(feat2)))
        weight128 = self.sigmoid(self.weight_128(feat2))
        
        # 第三层特征提取（降采样两次）
        feat3 = self.pool2(feat2)
        feat3 = self.relu(self.bn3(self.conv3(feat3)))
        weight256 = self.sigmoid(self.weight_256(feat3))
        
        return weight64, weight128, weight256
class BrightnessEnhancement(nn.Module):
    def __init__(self, channels):
        super(BrightnessEnhancement, self).__init__()
        self.brightness_attention = nn.Sequential(
            nn.Conv2d(channels + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, ref_brightness):
        ref_brightness = (ref_brightness - ref_brightness.min()) / (ref_brightness.max() - ref_brightness.min() + 1e-6)
        brightness_info = torch.cat([x, ref_brightness], dim=1)
        enhancement_map = self.brightness_attention(brightness_info)
        return x * (1.0 + enhancement_map)
# Swin Transformer Block
class OptimizedSwinBlock(nn.Module):
    def __init__(self, dim, heads=4, window_size=7):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        x = self.norm(x)
        b, h, w, c = x.shape
        # 计算需要填充的大小
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode='constant', value=0).permute(0, 2, 3, 1)
        _, h_pad, w_pad, _ = x.shape
        # 计算窗口
        windows = rearrange(x, 'b (h win_h) (w win_w) c -> (b h w) (win_h win_w) c', win_h=self.window_size, win_w=self.window_size)
        qkv = self.to_qkv(windows).chunk(3, dim=-1)
        q, k, v = qkv
        attn_out = torch.matmul(q, k.transpose(-2, -1)) / (c ** 0.5)
        attn_out = torch.softmax(attn_out, dim=-1)
        attn_out = torch.matmul(attn_out, v)
        # 还原形状
        out = rearrange(attn_out, '(b h w) (win_h win_w) c -> b (h win_h) (w win_w) c', h=h_pad // self.window_size, w=w_pad // self.window_size, win_h=self.window_size, win_w=self.window_size)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :]
        return self.mlp(out) + out
# 改进的 UNet 结构
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, use_swin=True, swin_heads=4, swin_window_size=7, base_channels=64):
        super(UNet, self).__init__()
        self.use_swin = use_swin
        # 使用 base_channels 参数来控制网络宽度
        self.texture_fusion = EnhancedTextureFusionBlock(in_channels)
        self.brightness_branch = BrightnessBranch(1)  # 仅处理亮度信息
        # 编码器路径
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.cbam1 = CBAMBlock(base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels*2)
        self.cbam2 = CBAMBlock(base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels*2, base_channels*4)
        self.cbam3 = CBAMBlock(base_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        # 瓶颈层
        self.bottleneck = DoubleConv(base_channels*4, base_channels*8)
        self.cbam_bottleneck = CBAMBlock(base_channels*8)
        if use_swin:
            self.swin_bottleneck = OptimizedSwinBlock(
                dim=base_channels*8, 
                heads=swin_heads, 
                window_size=swin_window_size
            )
        # 解码器路径
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4)  # 输入通道数是拼接后的通道数
        self.cbam_dec3 = CBAMBlock(base_channels*4)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2)  # 输入通道数是拼接后的通道数
        self.cbam_dec2 = CBAMBlock(base_channels*2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels*2, base_channels)  # 输入通道数是拼接后的通道数
        self.cbam_dec1 = CBAMBlock(base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        # 添加亮度增强模块
        self.brightness_enhancement = BrightnessEnhancement(out_channels) 
    @staticmethod
    def _interpolate_if_needed(x, size=None, scale_factor=None):
        """辅助函数:根据需要进行插值"""
        if size is None and scale_factor is None:
            return x
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, input_a, input_b):
        # 获取输入尺寸
        _, _, h, w = input_a.shape
        # 合并输入
        x = torch.cat([input_a, input_b], dim=1)
        x = self.texture_fusion(x) 
        # 获取亮度权重
        w64, w128, w256 = self.brightness_branch(input_b) 
        # 编码器路径
        e1 = self.cbam1(self.enc1(x))  # base_channels
        e1 = e1 * (1 + F.interpolate(w64, size=e1.shape[2:], mode='bilinear', align_corners=True))
        e2 = self.cbam2(self.enc2(self.pool1(e1)))  # base_channels*2
        e2 = e2 * (1 + F.interpolate(w128, size=e2.shape[2:], mode='bilinear', align_corners=True))
        e3 = self.cbam3(self.enc3(self.pool2(e2)))  # base_channels*4
        e3 = e3 * (1 + F.interpolate(w256, size=e3.shape[2:], mode='bilinear', align_corners=True))
        # 瓶颈层
        x = self.bottleneck(self.pool3(e3))  # base_channels*8
        if self.use_swin:
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.swin_bottleneck(x)
            x = rearrange(x, 'b h w c -> b c h w')
        x = self.cbam_bottleneck(x) 
        # 解码器路径 - 使用转置卷积进行上采样
        x = self.up3(x)  # base_channels*4
        if x.shape[2:] != e3.shape[2:]:
            x = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec3(torch.cat([x, e3], dim=1))  # 输入: base_channels*8 (4+4), 输出: base_channels*4
        x = self.cbam_dec3(x)
        x = self.up2(x)  # base_channels*2
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec2(torch.cat([x, e2], dim=1))  # 输入: base_channels*4 (2+2), 输出: base_channels*2
        x = self.cbam_dec2(x)
        x = self.up1(x)  # base_channels
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec1(torch.cat([x, e1], dim=1))  # 输入: base_channels*2 (1+1), 输出: base_channels
        x = self.cbam_dec1(x) 
        # 最终输出层
        out = self.final_conv(x)  # 1 channel
        # 保证输出尺寸与输入一致
        if out.shape[2:] != input_a.shape[2:]:
            out = F.interpolate(out, size=input_a.shape[2:], mode='bilinear', align_corners=True)   
        # 应用亮度增强
        input_b_brightness = input_b.mean(dim=1, keepdim=True)
        out = self.brightness_enhancement(out, input_b_brightness)
        return torch.sigmoid(out)