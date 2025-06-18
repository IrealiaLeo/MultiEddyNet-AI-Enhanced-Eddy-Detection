"""
定义用于加载涡旋数据的 PyTorch Dataset 类。
"""
import h5py
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple

def get_time_indices(hdf5_path: Path, target_years: List[int]) -> List[int]:
    """
    根据目标年份，从 HDF5 文件中获取对应的时间步索引。

    Args:
        hdf5_path (Path): HDF5 文件的路径。
        target_years (List[int]): 需要提取索引的目标年份列表。

    Returns:
        List[int]: 符合年份要求的时间步索引列表。
    """
    indices = []
    if not hdf5_path.exists():
        logging.error(f"HDF5 文件未找到: {hdf5_path}")
        return indices
        
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'time' not in f:
                logging.warning(f"HDF5 文件 '{hdf5_path}' 中找不到 'time' 数据集。")
                return indices
                
            times = f['time'][:]
            logging.info(f"从 HDF5 读取了 {len(times)} 个时间戳。")
            
            for i, t in enumerate(times):
                try:
                    # 时间格式为 yyyymmdd 整数
                    year = t // 10000
                    if year in target_years:
                        indices.append(i)
                except (ValueError, TypeError):
                    logging.warning(f"无法解析时间戳: {t} (索引 {i})。跳过。")
                    continue
    except Exception as e:
        logging.error(f"读取 HDF5 时间数据时出错: {e}")
        
    if not indices:
        logging.warning(f"在 HDF5 文件中未找到指定年份 {target_years} 的数据。")
        
    return indices

class EddyDataset(Dataset):
    """
    从 HDF5 文件加载涡旋数据的 PyTorch 数据集。
    在 worker 进程中独立打开文件句柄，以支持多进程数据加载。
    """
    def __init__(self, hdf5_path: Path, time_indices: List[int], feature_indices: List[int], 
                 reference_channel_idx: int, phase: str = 'train'):
        self.hdf5_path = hdf5_path
        self.time_indices = time_indices
        self.feature_indices = feature_indices
        self.reference_channel_idx = reference_channel_idx
        self.phase = phase
        self.num_selected_features = len(feature_indices)

        # 这些属性将在 __getitem__ 中被 worker 进程独立初始化
        self.file = None
        self.features_dataset = None
        self.labels_dataset = None
        
        # 预先获取维度信息以避免重复打开文件
        self._get_dims()

        logging.info(
            f"数据集 '{self.phase}' 创建: {len(self.time_indices)} 样本. "
            f"维度 (H={self.height}, W={self.width}, C={self.num_selected_features})"
        )

    def _get_dims(self):
        """仅用于获取维度信息，不在 worker 中重复调用"""
        with h5py.File(self.hdf5_path, 'r') as f:
            _, self.height, self.width, self.original_channels = f['features'].shape

    def _open_hdf5(self):
        """在需要时（每个 worker 进程首次调用时）打开 HDF5 文件。"""
        if self.file is None:
            try:
                self.file = h5py.File(self.hdf5_path, 'r')
                self.features_dataset = self.file['features']
                self.labels_dataset = self.file['labels']
            except Exception as e:
                logging.error(f"Worker 中打开 HDF5 文件失败: {e}")
                raise

    def __len__(self) -> int:
        return len(self.time_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载单个样本（特征、标签、掩膜）。
        """
        self._open_hdf5()  # 确保文件已在当前 worker 中打开
        
        actual_idx = self.time_indices[idx]
        try:
            # 加载标签和参考通道以生成掩膜
            labels_np = self.labels_dataset[actual_idx].astype(np.int64)
            reference_data = self.features_dataset[actual_idx, ..., self.reference_channel_idx]
            valid_mask_np = ~np.isnan(reference_data)

            # 加载选定的特征通道并填充 NaN
            features_selected = self.features_dataset[actual_idx, ..., self.feature_indices]
            features_filled = np.nan_to_num(features_selected, nan=0.0).astype(np.float32)

            # 转换为 Tensor
            features_tensor = torch.from_numpy(features_filled)
            labels_tensor = torch.from_numpy(labels_np)
            valid_mask_tensor = torch.from_numpy(valid_mask_np)
            
            return features_tensor, labels_tensor, valid_mask_tensor

        except Exception as e:
            logging.error(f"加载 HDF5 索引 {actual_idx} (列表 {idx}) 失败: {e}", exc_info=True)
            # 返回占位符以避免训练中断
            return (torch.zeros((self.height, self.width, self.num_selected_features), dtype=torch.float32),
                    torch.zeros((self.height, self.width), dtype=torch.int64),
                    torch.zeros((self.height, self.width), dtype=torch.bool))

    def __del__(self):
        """确保在对象销毁时关闭文件句柄。"""
        if self.file:
            self.file.close()