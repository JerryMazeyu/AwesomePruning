import os
import torch

class Config:
    def __init__(self):
        self.root_path = '/home/ubuntu/mzy/AwesomePruning'
        self.data_root_path = os.path.join(self.root_path, 'raw_data')
        self.auth_token = 'your_auth_token'
        self.cache_dir = '/mnt/share_data'
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

CONF = Config()

__all__ = ['CONF']