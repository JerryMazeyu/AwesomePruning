import os
import torch

class Config:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_root_path = os.path.join(self.root_path, 'data', 'raw_data')
        self.auth_token = 'your_auth_token'
        self.cache_dir = '/mnt/share_data'
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

CONF = Config()

__all__ = ['CONF']