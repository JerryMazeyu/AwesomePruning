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
        # If the dataset is a HuggingFace dataset, return the processed data
        if hasattr(self.dt, 'features'):
            if self.meta_.get('task_type') == 'language_modeling':
                # For language modeling tasks, ensure the returned data is in dictionary format
                if isinstance(item, dict):
                    # The data is already in dictionary format, check if it has the 'input_ids' key
                    if 'input_ids' in item:
                        return {'input_ids': item['input_ids']}
                    elif 'text' in item:
                        return {'input_ids': item['text']}
                    else:
                        # If the expected key is not found, return the entire dictionary
                        return item
                else:
                    # If the data is not in dictionary format, try to wrap it in a dictionary
                    return {'input_ids': item}
            elif self.meta_.get('task_type') == 'sequence_classification':
                if isinstance(item, dict):
                    if 'text' in item and 'label' in item:
                        return {'input_ids': item['text'], 'labels': item['label']}
                    else:
                        return item
                else:
                    # Process non-dictionary format classification data
                    return item
            else:
                return item
        # If the dataset is a torchvision dataset, return the original data
        return item

    @property
    def meta(self):
        return self.meta_

    @property
    def task_type(self):
        """Get the task type of the dataset"""
        return self.meta_.get('task_type', None)