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
        """Get target layer's parameters, be careful that this method would not return a dict.

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
    
    def get_grad(self, layer:Union[str, nn.Module], verbose=True) -> list:
        """Get target layer's gradients.

        Args:
            layer (Union[str, nn.Module]): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            list: _description_
        """
        gradients = {}
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        for name, para in layer.named_parameters():
            if para.grad:
                gradients[name] = para.grad
                if verbose:
                    print(f"Name {name}: shape {list(para.shape)}, min value {torch.min(p).item()}, max value {torch.max(p).item()}.")
            else:
                if verbose:
                    print(f"Name {name}: no gradients.")
        return gradients

    


if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/mzy/AwesomePruning")
    from models import get_model
    resnet50 = get_model('resnet50')
    mp = ModelInspector(resnet50)
    # mp.show()
    layer = mp.get_layer('layer1.0')
    mp.get_para('layer1.0')
    # mp.get_grad('layer1.0')