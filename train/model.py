"""
定义 MultiEddyNet 模型架构，包括 ASPP 模块。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ASPPConv(nn.Sequential):
    """带膨胀（空洞）卷积的基础块 (Conv-BN-ReLU)"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class ASPPPooling(nn.Sequential):
    """ASPP 中的全局平均池化分支"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) 模块。
    通过并行使用不同膨胀率的卷积来捕获多尺度信息。
    """
    def __init__(self, in_channels: int, out_channels: int, rates: list):
        super().__init__()
        # 1x1 卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # 多个膨胀卷积分支
        self.convs = nn.ModuleList([ASPPConv(in_channels, out_channels, rate) for rate in rates])
        # 全局池化分支
        self.pool = ASPPPooling(in_channels, out_channels)
        
        # 最终投影层，将所有分支的结果融合
        total_channels = out_channels * (2 + len(rates))
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        
        # 并行计算各个分支
        res1x1 = self.conv1x1(x)
        res_convs = [conv(x) for conv in self.convs]
        res_pool = F.interpolate(self.pool(x), size=input_size, mode="bilinear", align_corners=False)
        
        # 拼接所有结果
        res = torch.cat([res1x1] + res_convs + [res_pool], dim=1)
        return self.project(res)

class MultiEddyNet(nn.Module):
    """
    多通道、多尺度涡旋分割模型 (U-Net with ASPP)。
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        logging.info(f"初始化 MultiEddyNet: 输入通道={in_ch}, 输出类别={out_ch}")
        
        base_c = 64
        self.enc1 = self._conv_block(in_ch, base_c)       # 编码器 1
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = self._conv_block(base_c, base_c * 2)    # 编码器 2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = self._conv_block(base_c * 2, base_c * 4) # 编码器 3
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = self._conv_block(base_c * 4, base_c * 8) # 编码器 4
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # 瓶颈层使用 ASPP 模块
        aspp_out_channels = base_c * 16
        self.aspp = ASPPModule(in_channels=base_c * 8, out_channels=aspp_out_channels, rates=[1, 6, 12, 18])

        # 解码器路径
        self.up4 = nn.ConvTranspose2d(aspp_out_channels, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_c * 16, base_c * 8)
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_c * 2, base_c)
        
        # 输出层
        self.outc = nn.Conv2d(base_c, out_ch, kernel_size=1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """标准的双卷积块 (Conv-BN-ReLU x 2)"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _interpolate_if_needed(self, upsampled: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """如果上采样后的尺寸与跳跃连接的特征图尺寸不匹配，进行插值。"""
        if upsampled.shape[2:] != target.shape[2:]:
            return F.interpolate(upsampled, size=target.shape[2:], mode='bilinear', align_corners=False)
        return upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入维度转换 (B, H, W, C) -> (B, C, H, W)
        if x.ndim == 4 and x.shape[1] != self.enc1[0].in_channels:
            x = x.permute(0, 3, 1, 2)
            
        # 编码路径
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # ASPP 瓶颈层
        b = self.aspp(p4)

        # 解码路径（带跳跃连接）
        u4 = self.up4(b)
        u4 = self._interpolate_if_needed(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        u3 = self._interpolate_if_needed(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = self._interpolate_if_needed(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = self._interpolate_if_needed(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # 输出
        logits = self.outc(d1)
        return logits