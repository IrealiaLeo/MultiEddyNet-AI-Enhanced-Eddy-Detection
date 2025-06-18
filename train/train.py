"""
主程序入口：初始化所有组件并启动训练流程。
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import logging

import config
from dataset import EddyDataset, get_time_indices
from model import MultiEddyNet
from trainer import EnhancedTrainer
from utils import load_checkpoint

# 配置日志记录
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
    """主函数，执行所有初始化和训练步骤。"""
    if not config.HDF5_FILE.exists():
        logging.error(f"HDF5 数据文件未找到: {config.HDF5_FILE}")
        return

    # --- 1. 准备数据 ---
    logging.info(f"使用训练年份: {config.TRAIN_YEARS}, 验证年份: {config.VAL_YEARS}")
    train_indices = get_time_indices(config.HDF5_FILE, config.TRAIN_YEARS)
    val_indices = get_time_indices(config.HDF5_FILE, config.VAL_YEARS)
    
    if not train_indices or not val_indices:
        logging.error("未能获取足够的训练或验证数据索引。程序退出。")
        return
        
    logging.info(f"训练样本数: {len(train_indices)}, 验证样本数: {len(val_indices)}")
    
    train_dataset = EddyDataset(
        config.HDF5_FILE, train_indices, config.FEATURE_INDICES_TO_USE, 
        config.REFERENCE_CHANNEL_ORIGINAL_IDX, phase='train'
    )
    val_dataset = EddyDataset(
        config.HDF5_FILE, val_indices, config.FEATURE_INDICES_TO_USE, 
        config.REFERENCE_CHANNEL_ORIGINAL_IDX, phase='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0)
    )

    # --- 2. 初始化模型和优化器 ---
    logging.info(f"使用设备: {config.DEVICE}")
    model = MultiEddyNet(in_ch=config.IN_CHANNELS, out_ch=config.NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=5, verbose=True)
    scaler = GradScaler(enabled=config.USE_AMP)

    # --- 3. 加载检查点 (如果存在) ---
    start_epoch, best_val_metric = load_checkpoint(
        model, optimizer, scheduler, scaler, config.MODEL_SAVE_PATH, config.DEVICE
    )

    # --- 4. 初始化并启动训练器 ---
    trainer = EnhancedTrainer(
        model, train_loader, val_loader, optimizer,
        config.DEVICE, scheduler, scaler, config
    )
    trainer.best_val_metric = best_val_metric
    
    try:
        trainer.train(start_epoch=start_epoch)
    except Exception as e:
        logging.error("训练过程中发生未捕获的异常:", exc_info=True)
    finally:
        logging.info("训练流程结束。")

if __name__ == '__main__':
    main()