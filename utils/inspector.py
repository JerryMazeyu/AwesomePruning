import torch
from torch import nn
from typing import Union, Generator
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from torchsummary import summary
import torchviz


class ModelInspector():
    def __init__(self, model: nn.Module) -> None:
        self.model = model
    
    def summary(self, mock_inp:tuple=(3,224,224)):
        """Show model summary.

        Args:
            mock_inp (tuple, optional): Mock input tensor shape. Defaults to (3,224,224).
        """
        summary(self.model, mock_inp, device=next(self.model.parameters()).device.type)
    
    def get_layer(self, name:str, verbose=True) -> nn.Module:
        """Get a specific layer of model.

        Args:
            name (str): Layer name, split by dot(.), especially, can be 'all'

        Returns:
            nn.Module: Target layer
        """
        if name == 'all':
            return self.model
        else:
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
    
    def get_para(self, layer:Union[str, nn.Module]='all', type_:str='list', verbose=True) -> Union[list, Generator]:
        """Get target layer's parameters, be careful that this method would not return a dict.

        Args:
            layer (Union[str, nn.Module]): Layer name or layer module, name splited by '.', for example, 'layer1.0.conv1'.
            type_ (str, optional): Type of parameters, list or generator. Defaults to 'list'.

        Returns:
            Union[list, Generator]: Layer's parameters.
        """
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        para = layer.parameters()
        list_para = list(para)
        if verbose:
            for ind, p in enumerate(list_para):
                print(f"Index {ind}: shape {list(p.shape)}, min value {torch.min(p).item()}, max value {torch.max(p).item()}.")
        if type_ == 'list':
            return list_para
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
    
    def plot_histogram(self, tensors:list[torch.Tensor], bin:int=30, save_path='./tensor_hist.png'):
        """
        Plot a histogram of the values of a flattened tensor.
        
        Parameters:
        tensor (list[torch.Tensor]): A list of tensors.
        bins (int): Number of bins for the histogram.
        """
        assert len(tensors) != 0, ValueError(f"Make sure that tensor list has values, now it is {tensors}.")
        num_tensors = len(tensors)
        num_cols = ceil(sqrt(num_tensors))
        num_rows = ceil(num_tensors / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, tensor in enumerate(tensors):
            flattened_tensor = tensor.flatten().detach().cpu().numpy()
            axes[i].hist(flattened_tensor, bins=bin, edgecolor='k', alpha=0.7)
            axes[i].set_title(f'Tensor {i+1}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path)

    


if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/mzy/AwesomePruning")
    from models import get_model
    resnet50 = get_model('resnet50').cuda()
    mp = ModelInspector(resnet50)
    mp.summary()
    layer = mp.get_layer('layer1.0')
    tl = mp.get_para('layer1.0')
    # mp.plot_histogram(tl)
    # mp.get_grad('layer1.0')