import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Any, Dict, Optional

class BaseDataSet(Dataset):
    def __init__(self, dt, meta_: Dict[str, Any]):
        self.dt = dt
        self.meta_ = meta_

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        item = self.dt[idx]
        # 如果是HuggingFace数据集，返回处理后的数据
        if hasattr(self.dt, 'features'):
            if self.meta_.get('task_type') == 'language_modeling':
                return {'input_ids': item['text'] if 'text' in item else item['input_ids']}
            elif self.meta_.get('task_type') == 'sequence_classification':
                return {'input_ids': item['text'], 'labels': item['label']}
            else:
                return item
        # 如果是torchvision数据集，直接返回
        return item

    @property
    def meta(self):
        return self.meta_

    @property
    def task_type(self):
        """获取数据集的任务类型"""
        return self.meta_.get('task_type', None)