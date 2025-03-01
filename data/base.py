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
                # 对于语言建模任务，确保返回的是字典格式的input_ids
                if isinstance(item, dict):
                    # 已经是字典格式，检查是否有'input_ids'键
                    if 'input_ids' in item:
                        return {'input_ids': item['input_ids']}
                    elif 'text' in item:
                        return {'input_ids': item['text']}
                    else:
                        # 如果没有期望的键，返回整个字典
                        return item
                else:
                    # 不是字典格式，尝试包装成字典
                    return {'input_ids': item}
            elif self.meta_.get('task_type') == 'sequence_classification':
                if isinstance(item, dict):
                    if 'text' in item and 'label' in item:
                        return {'input_ids': item['text'], 'labels': item['label']}
                    else:
                        return item
                else:
                    # 处理非字典格式的分类数据
                    return item
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