import os
import torch
from torch import nn
from typing import Union, Generator, Optional
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from torchsummary import summary
from utils.io import LogRedirectMixin, log, generate_name
from config import CONF
from collections import OrderedDict

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM 版本的 summary 函数
该函数用于打印 LLM（例如 LlamaForCausalLM）模型的各层输入/输出形状、参数数量等信息，
适用于输入为字典形式的模型。
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def format_params(n):
    """
    将参数数量 n 转换为更易读的格式，例如：
      1234 -> "1.23K"
      1234567 -> "1.23M"
      1234567890 -> "1.23B"
    """
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)

def llm_summary(model, seq_len, batch_size=1, device="cuda"):
    """
    Summarize the LLM model's layers and parameters.

    Args:
        model: Model instance to be summarized, should be a LLM model based on Transformers.
        seq_len: Length of input sequence (e.g., 128, 256).
        batch_size: Batch size (default to 1).
        device: Device used ("cuda" or "cpu").
    
    Returns:
        None, print the summary of each layer's output shape, number of parameters, etc.
    """
    device = device.lower()

    # 获取模型配置中的 vocab_size（如果有），否则默认 1000
    vocab_size = getattr(model.config, "vocab_size", 1000)
    
    # 构造 dummy 输入：输入的 token_ids 为随机整数，attention_mask 全为 1
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
    dummy_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # 用于保存各层信息的 OrderedDict
    summary_dict = OrderedDict()
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            if not isinstance(input, tuple) or len(input) == 0:
                return
            try:
                first_input = input[0]
            except IndexError:
                return
            # 如果 input 是以字典形式传入，则取其中 "input_ids" 的张量
            if isinstance(first_input, dict):
                inp_tensor = first_input.get("input_ids", None)
                if inp_tensor is None:
                    return
            else:
                inp_tensor = first_input

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_dict)
            m_key = f"{class_name}-{module_idx + 1}"
            summary_dict[m_key] = OrderedDict()

            # 记录输入形状
            summary_dict[m_key]["input_shape"] = list(inp_tensor.size())
            summary_dict[m_key]["input_shape"][0] = batch_size  # 设置 batch_size

            # 记录输出形状
            if isinstance(output, (list, tuple)):
                # 如果输出为 tuple，则取第一个张量
                out_tensor = output[0]
            else:
                out_tensor = output
            summary_dict[m_key]["output_shape"] = list(out_tensor.size())
            summary_dict[m_key]["output_shape"][0] = batch_size

            # 计算参数数量
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.tensor(list(module.weight.size()), dtype=torch.long)).item()
                summary_dict[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.tensor(list(module.bias.size()), dtype=torch.long)).item()
            summary_dict[m_key]["nb_params"] = params

        # 对于非容器模块以及模型本身，注册 hook
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and (module != model):
            hooks.append(module.register_forward_hook(hook))

    # 在模型上注册 hook
    model.apply(register_hook)

    # 进行一次前向传播
    model.eval()
    with torch.no_grad():
        model(**dummy_input)

    # 移除 hook
    for h in hooks:
        h.remove()

    # 打印摘要信息
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary_dict:
        nb_params = summary_dict[layer]["nb_params"]
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary_dict[layer]["output_shape"]),
            # "{0:,}".format(summary_dict[layer]["nb_params"]),
            format_params(nb_params)
        )
        total_params += summary_dict[layer]["nb_params"]
        total_output += np.prod(summary_dict[layer]["output_shape"])
        if summary_dict[layer].get("trainable", False):
            trainable_params += summary_dict[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print("Total params: {} ({})".format(total_params, format_params(total_params)))
    print("Trainable params: {} ({})".format(trainable_params, format_params(trainable_params)))
    print("Non-trainable params: {} ({})".format(total_params - trainable_params, format_params(total_params - trainable_params)))
    print("----------------------------------------------------------------")

    # 估算内存占用（单位：MB）
    total_input_size = abs(np.prod([batch_size, seq_len]) * 4.0 / (1024 ** 2))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2))
    total_size = total_input_size + total_output_size + total_params_size

    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")



class ModelInspector(LogRedirectMixin):
    def __init__(self, model:nn.Module, log_path:Optional[str]=None) -> None:
        super().__init__(log_path)
        self.model = model
        self.status = 'blank'  # blank, trained
        if log_path:
            self.log_path = log_path

    def summary(self, mock_inp:tuple=(3,224,224)):
        """Show model summary.

        Args:
            mock_inp (tuple, optional): Mock input tensor shape. Defaults to (3,224,224).
        """
        if self.model._model_type == 'CNN':
            summary(self.model, mock_inp, device=next(self.model.parameters()).device.type)
        elif self.model._model_type == 'LM':
            llm_summary(self.model, seq_len=128, batch_size=2, device=CONF.device)
        
    def calibrate(self, calibration_dataset:torch.utils.data.Dataset, batch_size:int=32):
        """Calibrate model with a dataset.

        Args:
            calibration_dataset (torch.utils.data.Dataset): Calibration dataset.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        self.model.eval()
        with torch.no_grad():
            for i, (inp, _) in enumerate(calibration_dataset):
                if i == 0:
                    self.model(inp.cuda())
                else:
                    self.model(inp.cuda())
    
    def get_layer(self, name:str, verbose=True) -> nn.Module:
        """Get a specific layer of model.

        Args:
            name (str): Layer name, split by dot(.), especially, can be 'all'.

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
                log(f"Target layer: \n {tar}")
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
                log(f"Index {ind}: shape {list(p.shape)}, min value {torch.min(p).item()}, max value {torch.max(p).item()}.")
        if type_ == 'list':
            return list_para
        elif type_ == 'generator':
            return para
        else:
            raise ValueError(f"Make sure that type_ is in 'list' or 'generator'.")
    
    def get_grad(self, layer:Union[str, nn.Module], type_:str='dict', verbose=True) -> Union[list, dict]:
        """Get target layer's gradients.
           Make sure that model has been trained.

        Args:
            layer (Union[str, nn.Module]): Layer name split by dot.
            verbose (bool, optional): If show verbose infomation. Defaults to True.

        Returns:
            Union[list, dict: Gradient list / dict.
        """
        gradients = {}
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        for name, para in layer.named_parameters():
            if para.grad:
                gradients[name] = para.grad
                if verbose:
                    log(f"Name {name}: shape {list(para.shape)}, min value {torch.min(para.grad).item()}, max value {torch.max(para.grad).item()}.")
            else:
                if verbose:
                    log(f"Name {name}: no gradients.")
        if type_ == 'dict':
            return gradients
        else:
            return [x[1] for x in gradients.item()]
    
    def plot_histogram(self, tensors:list[torch.Tensor], bin:int=30, save_path='tensor_hist.png'):
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
        plt.savefig(os.path.join(self.log_path, save_path))



if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/mzy/AwesomePruning")
    from models import get_model
    
    llama2, tokenizer = get_model('Llama-2-7b-hf', cache_dir=CONF.cache_dir)
    llama2.to(CONF.device)    
    mp = ModelInspector(llama2)
    mp.summary()
    tar = 'model.layers.0.self_attn'
    layer = mp.get_layer(tar)
    tl = mp.get_para(tar)
    mp.plot_histogram(tl)
    mp.get_grad(tar)