import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from typing import Union, Optional
from utils.inspector import ModelInspector
from utils.io import soft_mkdir, log, generate_name

class LayerManipulator(ModelInspector):
    def __init__(self, model, log_path:str=None):
        if not log_path:
            log_path = generate_name()
        self.log_path = log_path
        super().__init__(model, log_path)
        