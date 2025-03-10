import os
from torch.utils.data import DataLoader
import torchshow as ts
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from data import get_mm_datasets
from models import get_mllm_model
from config import CONF
from utils import ModelInspector, log
# import warnings
# warnings.filters('ignore')


def plot_heads_heatmap(tensor, path=None, filter_list=None):
    """
    Plot heatmaps for each head in a tensor.
    
    This function plots each [8,16] slice of the input tensor as a heatmap using seaborn.
    The heads are arranged in a grid. By default, a tensor with 32 heads is arranged in a 4x8 layout.
    If a filter_list is provided, only the specified heads are plotted and the grid layout is computed
    with 4 rows (i.e., number of columns = ceil(n_filtered / 4)). Extra subplots (if any) are hidden.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        A tensor of shape [num_heads, 8, 16]. The tensor can be on CUDA; it will be automatically moved to CPU.
    filter_list : list of int, optional
        List of head indices to plot. If provided, only these heads will be plotted.
        For example, if filter_list = [0, 3, 4, 9, 10], then only these heads are plotted.
    
    Returns:
    --------
    None
    """
    # If the tensor is on CUDA, move it to CPU.
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy for plotting.
    data = tensor.detach().numpy()
    
    # Determine which heads to plot.
    if filter_list is not None:
        selected_indices = filter_list
    else:
        selected_indices = list(range(data.shape[0]))
    
    n_heads = len(selected_indices)
    
    # Determine grid layout:
    # Default case for 32 heads: 4 rows x 8 columns.
    # If filter_list is provided, we fix rows=4 and compute number of columns as ceil(n_heads / 4).
    
    n_rows = 4
    n_cols = math.ceil(n_heads / 4)
    
    # Create subplots with an appropriate figure size.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    
    # If axes is a 2D array, flatten it.
    if isinstance(axes, (list, )):
        axes = [ax for row in axes for ax in row]
    else:
        axes = axes.flatten()
    
    # Plot each selected head as a heatmap.
    for idx, head_idx in enumerate(selected_indices):
        ax = axes[idx]
        # Plot heatmap for the head (shape [8,16])
        sns.heatmap(data[head_idx], ax=ax, cbar=False)
        # Set title with 1-indexed head number.
        ax.set_title(f"head_{head_idx+1}", fontsize=10)
        # Remove axis labels for clarity.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots.
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(path)

def compare_layer_heatmaps(data_dict, save_path):
    """
    Plot comparison heatmaps for each layer in a dictionary.
    
    Given a dictionary where each key is a layer name and its value is a list of two tensors,
    this function plots three heatmaps per key:
      - The first tensor ("Input 1")
      - The second tensor ("Input 2")
      - The difference (delta) between the two ("Delta")
    
    If a tensor has 3 dimensions and the first dimension equals 1, it is squeezed to become 2D.
    If the tensors are on CUDA, they are automatically moved to CPU.
    The plots for each key are arranged in one row (3 columns) and the keys are stacked vertically.
    The final figure is saved to the provided save_path.
    
    Parameters:
    -----------
    data_dict : dict
        A dictionary where each key (str) represents a layer name, and its value is a list of two tensors.
        Each tensor is expected to be 2D, or 3D with the first dimension equal to 1.
    save_path : str
        The file path where the generated figure will be saved.
    
    Returns:
    --------
    None
    """
    keys = list(data_dict.keys())
    n_layers = len(keys)
    n_cols = 3  # Input1, Input2, Delta
    
    # Create a subplot grid: one row per layer, 3 columns.
    fig, axes = plt.subplots(n_layers, n_cols, figsize=(n_cols * 3, n_layers * 3))
    
    # If only one layer, make axes 2D for consistent indexing.
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for row, key in enumerate(keys):
        tensors = data_dict[key]
        # 获取两个tensor
        tensor1, tensor2 = tensors[0], tensors[1]
        
        # 如果tensor在CUDA上，则转移到CPU，并detach转换为numpy数组
        if tensor1.is_cuda:
            tensor1 = tensor1.cpu()
        if tensor2.is_cuda:
            tensor2 = tensor2.cpu()
        
        # 若tensor为3维且第一维为1，则 squeeze 成二维
        if tensor1.dim() == 3 and tensor1.shape[0] == 1:
            tensor1 = tensor1.squeeze(0)
        if tensor2.dim() == 3 and tensor2.shape[0] == 1:
            tensor2 = tensor2.squeeze(0)
        
        # 转为numpy
        data1 = tensor1.detach().numpy()
        data2 = tensor2.detach().numpy()
        delta = data1 - data2
        
        # 对应每一行的3个子图
        ax1 = axes[row, 0]
        ax2 = axes[row, 1]
        ax3 = axes[row, 2]
        
        # 绘制 heatmap，设置cbar=False使得图形更紧凑；你可以根据需要开启colorbar
        sns.heatmap(data1, ax=ax1, cbar=False)
        sns.heatmap(data2, ax=ax2, cbar=False)
        sns.heatmap(delta, ax=ax3, cbar=False)
        
        # 设置子图标题，包含层名称及图示
        ax2.set_title(f"{key}", fontsize=10)
        
        # 为了美观，移除坐标轴刻度
        for ax in (ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path)

dataset = get_mm_datasets('mmsafty')
dataset_v2 = get_mm_datasets('mmsafty_v2')

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader_v2 = DataLoader(dataset_v2, batch_size=1, shuffle=False)



model, vis_processor = get_mllm_model('minigpt4_llama')

layers = ['llama_model.base_model.model.model.layers.13.self_attn.q_proj',
          'llama_model.base_model.model.model.layers.9.self_attn.q_proj',
          'llama_model.base_model.model.model.layers.3.self_attn.q_proj']

head_index = {
    'llama_model.base_model.model.model.layers.13.self_attn.q_proj': 4,
    'llama_model.base_model.model.model.layers.9.self_attn.q_proj': 9,
    'llama_model.base_model.model.model.layers.3.self_attn.q_proj': 8,
}

delta_res = {}
inspector = ModelInspector(model, tokenizer=None, log_path=None)
path_ = inspector.log_path
from functools import partial
log = partial(log, file=open(os.path.join(path_, 'log.txt')))

for idx in range(len(dataloader)):
    unsafe_batch = next(iter(dataloader))
    safe_batch = next(iter(dataloader_v2))
    res = {}
    log("="*50)
    unsafe_results = inspector.get_representation(layers, unsafe_batch, forward_type='generate', split_attention_heads=True, num_heads=32, verbose=False)
    safe_results = inspector.get_representation(layers, safe_batch, forward_type='generate', split_attention_heads=True, num_heads=32, verbose=False)
    log(f"{idx}: Unsafe inputs: {unsafe_batch['texts'][0]}")
    log(f"{idx}: Safe inputs: {safe_batch['texts'][0]}")
    log(f"{idx}: Unsafe outputs: {unsafe_results['meta']['outputs']}")
    log(f"{idx}: Safe outputs: {safe_results['meta']['outputs']}")
    delta_res[idx] = {}
    
    for layer in layers:
        tar_unsafe_head_rep = unsafe_results[layer].squeeze().reshape(32, 8, -1)[head_index[layer], :, :]
        tar_safe_head_rep = safe_results[layer].squeeze().reshape(32, 8, -1)[head_index[layer], :, :]
        res[layer] = [tar_unsafe_head_rep, tar_safe_head_rep]

        delta_sum = torch.norm(tar_unsafe_head_rep - tar_safe_head_rep, p=1)
        delta_sum_item = delta_sum.item()
        delta_ratio_unsafe = delta_sum / torch.norm(tar_unsafe_head_rep, p=1).item()
        delta_ratio_safe = delta_sum / torch.norm(tar_safe_head_rep, p=1).item()
        
        delta_res[idx][f"{layer}_delta_sum"] = delta_sum_item
        delta_res[idx][f"{layer}_delta_ratio(unsafe)"] = delta_ratio_unsafe
        delta_res[idx][f"{layer}_delta_ratio(safe)"] = delta_ratio_safe
        log(f"{idx}: delta_sum: {delta_sum_item}")
        log(f"{idx}: delta_ratio(unsafe): {delta_ratio_unsafe}")
        log(f"{idx}: delta_ratio(safe): {delta_ratio_safe}")
    
    compare_layer_heatmaps(res, os.path.join(path_, f"batch_{idx}.png"))
    