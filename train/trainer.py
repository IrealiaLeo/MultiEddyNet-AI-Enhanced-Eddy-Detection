"""
定义核心的 Trainer 类，负责封装训练和验证逻辑。
"""
import torch
import numpy as np
import logging
import csv
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import kornia.augmentation as K

from utils import masked_cross_entropy_loss, calculate_masked_iou

class EnhancedTrainer:
    """封装了模型训练、验证、日志记录和检查点保存的完整流程。"""
    def __init__(self, model, train_loader, val_loader, optimizer, device,
                 scheduler, scaler, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        
        self.writer = SummaryWriter(log_dir=str(config.LOG_DIR))
        self.best_val_metric = -float('inf')
        self._init_loss_csv()
        self._init_augmentor()

    def _init_loss_csv(self):
        """初始化或检查损失记录 CSV 文件的文件头。"""
        write_header = not self.config.LOSS_CSV_PATH.exists()
        with open(self.config.LOSS_CSV_PATH, 'a', newline='') as f:
            if write_header:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mIoU_eddies',
                                 'val_iou_bg', 'val_iou_warm', 'val_iou_cold', 'lr'])
        logging.info(f"损失将记录到: {self.config.LOSS_CSV_PATH}")

    def _init_augmentor(self):
        """初始化 Kornia 数据增强器。"""
        self.augmentor = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=['input', 'mask'],
            same_on_batch=False,
        ).to(self.device)
        logging.info("Kornia 数据增强已启用。")

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)

        for features, labels, mask in progress_bar:
            features = features.permute(0, 3, 1, 2).to(self.device, non_blocking=self.config.PIN_MEMORY)
            labels = labels.to(self.device, non_blocking=self.config.PIN_MEMORY)
            mask = mask.to(self.device, non_blocking=self.config.PIN_MEMORY)

            # 数据增强
            labels_for_aug = labels.unsqueeze(1).float()
            features, labels_aug = self.augmentor(features, labels_for_aug)
            labels = labels_aug.squeeze(1).long()

            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(features)
                loss = masked_cross_entropy_loss(outputs, labels, mask)
            
            if torch.isnan(loss):
                logging.warning(f"检测到 NaN 损失！跳过此批次。")
                continue

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.config.MAX_GRAD_NORM:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        all_ious = []
        progress_bar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch+1}", leave=False)

        with torch.no_grad():
            for features, labels, mask in progress_bar:
                features = features.permute(0, 3, 1, 2).to(self.device, non_blocking=self.config.PIN_MEMORY)
                labels = labels.to(self.device, non_blocking=self.config.PIN_MEMORY)
                mask = mask.to(self.device, non_blocking=self.config.PIN_MEMORY)

                with autocast(enabled=self.config.USE_AMP):
                    outputs = self.model(features)
                    loss = masked_cross_entropy_loss(outputs, labels, mask)
                
                total_loss += loss.item()
                iou_class, _ = calculate_masked_iou(outputs, labels, mask, self.config.NUM_CLASSES)
                all_ious.append(iou_class)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou_class = np.mean(all_ious, axis=0)
        avg_mIoU_eddies = np.mean(avg_iou_class[1:])
        
        return avg_loss, avg_iou_class.tolist(), avg_mIoU_eddies
    
    def _save_checkpoint(self, epoch, is_best):
        """保存模型检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_metric': self.best_val_metric,
        }
        if is_best:
            torch.save(checkpoint, self.config.MODEL_SAVE_PATH)
            logging.info(f"最佳模型已保存到: {self.config.MODEL_SAVE_PATH}")

    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.config.EPOCHS):
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_iou_class, val_mIoU_eddies = self._validate_one_epoch(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_mIoU_eddies)
            
            # 日志
            logging.info(f"Epoch {epoch+1}/{self.config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                         f"Val Eddy mIoU: {val_mIoU_eddies:.4f} | LR: {current_lr:.1e}")
            
            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("mIoU_Eddies/validation", val_mIoU_eddies, epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)
            for i, iou in enumerate(val_iou_class):
                self.writer.add_scalar(f"IoU/class_{i}", iou, epoch)

            # CSV
            with open(self.config.LOSS_CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, train_loss, val_loss, val_mIoU_eddies] + val_iou_class + [current_lr])

            # 保存模型
            is_best = val_mIoU_eddies > self.best_val_metric
            if is_best:
                self.best_val_metric = val_mIoU_eddies
            self._save_checkpoint(epoch, is_best)
            
        self.writer.close()
        logging.info("训练完成。")