"""
存放所有可配置的参数和全局设置。
"""
import torch
import platform
from pathlib import Path

# --- 1. 路径配置 ---
BASE_DIR = Path("./")  # 项目根目录
# HDF5_FILE = BASE_DIR / "h5_moba" / "eddy_data_daily_raw_combined.h5"
HDF5_FILE = Path(__file__).parent.parent / "data/h5_final/eddy_data_daily_raw_combined.h5"

LOG_DIR = BASE_DIR / "runs" / "multieddynet_final"
MODEL_SAVE_PATH = BASE_DIR / "checkpoints" / "best_multieddynet_final.pth"
LOSS_CSV_PATH = LOG_DIR / "multieddynet_loss_log_final.csv"

# 确保目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- 2. 数据集配置 ---
# 论文中使用了1996-2020共25年的数据
# 这里选择4年训练，1年验证作为示例
TRAIN_YEARS = [1997, 2002, 2010, 2015]
VAL_YEARS = [2005]

# 特征选择，对应 HDF5 'features' 数据集的通道索引
# [u, v, velocity_mag, ow, vorticity, ssh] -> 索引 [0, 1, 2, 3, 4, 5]
# 使用 'velocity_mag', 'ow', 'ssh'
FEATURE_INDICES_TO_USE = [2, 3, 5]

# 用于生成陆地/海洋掩膜的参考通道索引（必须是始终有值的通道）
# SSH (索引5) 是一个很好的选择
REFERENCE_CHANNEL_ORIGINAL_IDX = 5

# --- 3. 训练超参数 ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
WEIGHT_DECAY = 1e-5       # AdamW 优化器的权重衰减
MAX_GRAD_NORM = 1.0         # 梯度裁剪阈值，防止梯度爆炸

# --- 4. 模型配置 ---
IN_CHANNELS = len(FEATURE_INDICES_TO_USE)
NUM_CLASSES = 3  # 0: 背景, 1: 暖涡, 2: 冷涡

# --- 5. 环境配置 ---
# Windows 下 num_workers>0 可能不稳定，推荐设为 0
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pin_memory 仅在 CUDA 和 num_workers > 0 时有效
PIN_MEMORY = (DEVICE.type == 'cuda') and (NUM_WORKERS > 0)
# 自动混合精度 (AMP) 仅在 CUDA 上有效
USE_AMP = (DEVICE.type == 'cuda')