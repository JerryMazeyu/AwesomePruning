import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class BaseDataSet(Dataset, ABC):
    def __init__(self, dt, meta_):
        self.dt = dt
        self.meta_ = meta_

    @property
    def meta(self):
        return self.meta_