import torch
import numpy as np
from torch import nn
from typing import Union, Optional
import sys
sys.path.append('/home/ubuntu/mzy/AwesomePruning')
print(sys.path)
from utils.inspector import ModelInspector

class WeightsCutter(ModelInspector):
    def __init__(self, model):
        super().__init__(model)
    
    def threshold_prune_para(self, module:str='all', threshold:Optional[float]=0.0, mode:str='lt', verbose:bool=True):
        """Generate mask by prune paras by threshold.

        Args:
            module (str): Module to prune. Defaults to 'all'.
            threshold(Optional[float]): Threshold.
            mode (str): Prune mode: could be ['gt', 'lt', 'g', 'l', 'eq'. Defaults to lt.
            verbose (bool): If print verbose infomation. Defaults to True.
        """
        assert mode in ['gt', 'lt', 'g', 'l', 'eq'], ValueError(f"Make sure that mode is in ['gt', 'lt', 'g', 'l', 'eq'], now mode is {mode}.")
        paras = self.get_para(module, type_='list', verbose=False)
        if mode == 'lt':
            masks = [torch.tensor((x <= threshold).float()) for x in paras]
        elif mode == 'gt':
            masks = [torch.tensor((x >= threshold).float()) for x in paras]
        elif mode == 'g':
            masks = [torch.tensor((x > threshold).float()) for x in paras]
        elif mode == 'l':
            masks = [torch.tensor((x < threshold).float()) for x in paras]
        else:
            masks = [torch.tensor((x == threshold).float()) for x in paras]
        if verbose:
            for ind, x in enumerate(masks):
                print(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            print(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
        return masks
    
    def threshold_prune_grad(self, module:str='all', threshold:Optional[float]=0.0, mode:str='lt', verbose:bool=True):
        """Generate mask by prune gradients corresponding to parameters by threshold.

        Args:
            module (str): Module to prune. Defaults to 'all'.
            threshold(Optional[float]): Threshold.
            mode (str): Prune mode: could be ['gt', 'lt', 'g', 'l', 'eq'. Defaults to lt.
            verbose (bool): If print verbose infomation. Defaults to True.
        """
        assert mode in ['gt', 'lt', 'g', 'l', 'eq'], ValueError(f"Make sure that mode is in ['gt', 'lt', 'g', 'l', 'eq'], now mode is {mode}.")
        grads = self.get_grad(module, type_='list', verbose=False)
        if mode == 'lt':
            masks = [torch.tensor((x <= threshold).float()) for x in grads]
        elif mode == 'gt':
            masks = [torch.tensor((x >= threshold).float()) for x in grads]
        elif mode == 'g':
            masks = [torch.tensor((x > threshold).float()) for x in grads]
        elif mode == 'l':
            masks = [torch.tensor((x < threshold).float()) for x in grads]
        else:
            masks = [torch.tensor((x == threshold).float()) for x in grads]
        if verbose:
            for ind, x in enumerate(masks):
                print(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            print(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
        return masks

    def prune_by_weights(self, module:str='all', weights:Optional[list]=None, prune_rate:float=0.1, verbose=True):
        """Generate mask by weights.

        Args:
            module (str, optional): Threshhold to prune. Defaults to 'all'.
            weights (Optional, optional): Importance weights. Defaults to None.
            prune_rate (float, optional): Prune rate. Defaults to 0.1.
            verbose (bool): If print verbose infomation. Defaults to True.
        """  
        paras = self.get_para(module, type_='list', verbose=False)
        if not weights:
            if verbose:
                print("No weights given, random initialize it for demonstration.")
            weights = []
            for i in range(len(paras)):
                weights.append(torch.rand_like(paras[i]))
        else:
            for i in range(len(paras)):
                assert paras[i].shape == weights[i].shape, ValueError(f"Wrong weights shape, {i}-th layer parameter shape is {paras[i].shape}, but now it is {weights[i].shape}")
        flat_weights = torch.cat([torch.flatten(x) for x in weights])
        k = int(len(flat_weights) * prune_rate)
        threshold_value = torch.kthvalue(flat_weights, k).values.item()
        masks = []
        for x in weights:
            masks.append((x > threshold_value).float())
        if verbose:
            for ind, x in enumerate(masks):
                print(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            print(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
        return masks
        

if __name__ == '__main__':
    from models import get_model
    model = get_model('resnet18')
    wc = WeightsCutter(model)
    # wc.threshold_prune_para()
    # wc.threshold_prune_grad()
    wc.prune_by_weights()



