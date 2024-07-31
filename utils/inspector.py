import torch
from torch import nn
from typing import Union, Generator

class ModelInspector():
    def __init__(self, model: nn.Module) -> None:
        self.model = model
    
    def show(self):
        print(self.model)
    
    def get_layer(self, name: str, verbose=True) -> nn.Module:
        """Get a specific layer of model.

        Args:
            name (str): Layer name, split by dot(.)

        Returns:
            nn.Module: Target layer
        """
        name_list = name.split('.')
        tar = self.model
        for i in name_list:
            try:
                i = eval(i)
            except:
                pass
            if isinstance(i, str):
                tar = getattr(tar, i)
            elif isinstance(i, int):
                tar = tar[i]
        if verbose:
            print(f"Target layer: \n {tar}")
        return tar
    
    def get_para(self, layer:Union[str, nn.Module], type_:str='list', verbose=True) -> Union[list, Generator]:
        """Get target layer's parameters.

        Args:
            layer (Union[str, nn.Module]): Layer name or layer module.
            type_ (str, optional): Type of parameters, list or generator. Defaults to 'list'.

        Returns:
            Union[list, Generator]: Layer's parameters.
        """
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        para = layer.parameters()
        if verbose:
            for ind, p in enumerate(para):
                print(f"Index {ind}: shape {list(p.shape)}, min value {torch.min(p).item()}, max value {torch.max(p).item()}.")
        if type_ == 'list':
            return list(para)
        elif type_ == 'generator':
            return para
        else:
            raise ValueError(f"Make sure that type_ is in 'list' or 'generator'.")
    


if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/mzy/AwesomePruning")
    from models import get_model
    resnet50 = get_model('resnet50')
    mp = ModelInspector(resnet50)
    # mp.show()
    layer = mp.get_layer('layer1.0')
    mp.get_para(layer)