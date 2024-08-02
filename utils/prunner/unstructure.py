from torch import nn
from typing import Union, Optional
from inspector import ModelInspector

class WeightsCutter(ModelInspector):
    def __init__(self, model):
        super().__init__(model)
    
    def threshold_prune(self, module:Optional[nn.Module]=None, threshold=None):
        """Generate mask by threshold.

        Args:
            module (Optional[nn.Module], optional): _description_. Defaults to None.
        """