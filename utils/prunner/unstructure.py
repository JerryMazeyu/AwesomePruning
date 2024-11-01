import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from typing import Union, Optional
from utils.inspector import ModelInspector
from utils.io import soft_mkdir, log, generate_name

class WeightsCutter(ModelInspector):
    def __init__(self, model, log_path:str=None):
        if not log_path:
            log_path = generate_name()
        self.log_path = log_path
        super().__init__(model, log_path)
    
    def threshold_prune_para(self, module:str='all', threshold:Optional[float]=0.0, mode:str='lt', verbose:bool=True):
        """Generate mask by prune paras by threshold.

        Args:
            module (str): Module to prune. Defaults to 'all'.
            threshold(Optional[float]): Threshold.
            mode (str): Prune mode: could be ['gt', 'lt', 'g', 'l', 'eq']. Defaults to lt.
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
                log(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            log(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
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
                log(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            log(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
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
                log("No weights given, random initialize it for demonstration.")
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
                log(f"Layer {ind}: Total parameters: {x.numel()}; Remain parameters: {x.sum()}; Pruned parameters: {x.numel() - x.sum()}; pruned percent is {np.round((x.numel() - x.sum().detach().cpu()) / x.sum().detach().cpu(), 2)}.")
            stack_mask = torch.cat([torch.flatten(x) for x in masks])
            log(f"Total: Total parameters: {stack_mask.numel()}; Remain parameters: {stack_mask.sum()}; Pruned parameters: {stack_mask.numel() - stack_mask.sum()}; pruned percent is {np.round((stack_mask.numel() - stack_mask.sum().detach().cpu()) / stack_mask.sum().detach().cpu(), 2)}.")
        return masks
    
    def show_mask(self, weights:torch.Tensor, mask: torch.Tensor) -> None:
        """Visualizes a heatmap where unmasked (mask=1) values are shown in red tones, 
           and masked (mask=0) values are shown in a uniform blue color.

        Args:
            weights (torch.Tensor): Matrix / Vector / (1, X, Y) like tensor, the real weights.
            mask (torch.Tensor): Tensor of the same shape as weights, the mask(0 or 1).
        """
        assert weights.shape == mask.shape, f"The shape of mask({mask.shape}) and weights({weights.shape}) must match."
        weights_np = weights.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_np, mask=mask_np == 0, cmap='Blues', annot=False, cbar=True, linewidths=0.5)

        # Overlay solid red color for mask=0
        weights_np[mask_np == 0] = np.nan  # Set masked values to NaN to overlay red color

        plt.title("Heatmap with Masked Regions")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.savefig(os.path.join(self.log_path, "heatmap_with_masked_regions.png"))
        plt.close()

if __name__ == '__main__':
    from models import get_model
    model = get_model('resnet18')
    wc = WeightsCutter(model)
    wc.threshold_prune_para()
    # wc.threshold_prune_grad()
    # wc.prune_by_weights()
    weights = torch.randn(10, 10)
    mask = torch.randint(0, 2, (10, 10))
    wc.show_mask(weights, mask)



