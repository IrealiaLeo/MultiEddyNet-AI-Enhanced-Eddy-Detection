"""
存放通用辅助函数，如损失计算、指标评估、检查点管理等。
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List

def masked_cross_entropy_loss(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    计算带有掩膜的交叉熵损失，只考虑有效海洋区域。

    Args:
        outputs (torch.Tensor): 模型输出的 logits (B, C, H, W)。
        targets (torch.Tensor): 真实标签 (B, H, W)。
        mask (torch.Tensor): 有效区域掩膜 (B, H, W)，True为有效。

    Returns:
        torch.Tensor: 计算出的标量损失值。
    """
    loss_per_pixel = F.cross_entropy(outputs, targets, reduction='none')
    masked_loss = loss_per_pixel * mask.float()
    num_valid_pixels = mask.float().sum()
    
    if num_valid_pixels > 0:
        return masked_loss.sum() / num_valid_pixels
    return torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)


def calculate_masked_iou(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                         num_classes: int, smooth: float = 1e-6) -> Tuple[List[float], float]:
    """
    计算带有掩膜的 Intersection over Union (IoU) 指标。

    Args:
        outputs, targets, mask: 同上。
        num_classes (int): 类别总数。
        smooth (float): 防止除以零的平滑项。

    Returns:
        Tuple[List[float], float]: 
        - iou_per_class: 每个类别的平均 IoU 列表。
        - mean_iou_eddies: 涡旋类别（1和2）的平均 mIoU。
    """
    preds = torch.argmax(outputs, dim=1)
    iou_per_class = []

    for cls in range(num_classes):
        pred_inds = (preds == cls) & mask
        target_inds = (targets == cls) & mask

        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()

        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou.item())
    
    mean_iou_eddies = np.mean(iou_per_class[1:]) if num_classes > 1 else 0.0
    return iou_per_class, mean_iou_eddies


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    """
    从指定路径加载模型、优化器、调度器和混合精度缩放器的状态。
    """
    start_epoch = 0
    best_val_metric = -float('inf')

    if not os.path.exists(checkpoint_path):
        logging.info("未找到检查点文件，将从头开始训练。")
        return start_epoch, best_val_metric

    try:
        logging.info(f"正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重（处理 DataParallel/DDP 的 'module.' 前缀）
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        logging.info("模型权重已加载。")

        # 加载优化器、调度器、缩放器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("优化器状态已加载。")
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("调度器状态已加载。")
        if scaler and 'scaler_state_dict' in checkpoint and device.type == 'cuda':
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info("GradScaler 状态已加载。")
            
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
        logging.info(f"检查点加载完成。将从 Epoch {start_epoch} 继续。最佳验证指标: {best_val_metric:.4f}")

    except Exception as e:
        logging.error(f"加载检查点失败: {e}。将从头开始训练。", exc_info=True)
        start_epoch, best_val_metric = 0, -float('inf')
        
    return start_epoch, best_val_metric