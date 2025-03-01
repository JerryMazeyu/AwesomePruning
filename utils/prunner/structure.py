import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from typing import Union, Optional, List, Dict, Any
from utils.inspector import ModelInspector
from utils.io import soft_mkdir, log, generate_name
import logging

logger = logging.getLogger(__name__)

class LayerManipulator(ModelInspector):
    def __init__(self, model, log_path:str=None):
        if not log_path:
            log_path = generate_name()
        self.log_path = log_path
        super().__init__(model, log_path)
        
# 添加Transformer模型的层级剪枝函数
def prune_transformer_layers(model: nn.Module, layers_to_prune: List[int]) -> nn.Module:
    """
    剪枝Transformer模型的指定层
    
    Args:
        model: Transformer模型
        layers_to_prune: 要剪枝的层索引列表
        
    Returns:
        pruned_model: 剪枝后的模型
    """
    if not hasattr(model, '_model_type') or model._model_type != 'LM':
        raise ValueError("模型必须是Transformer语言模型类型")
    
    # 根据模型架构获取层列表
    layers = None
    
    # 尝试常见的层路径
    layer_paths = [
        "layers",             # 直接访问层
        "model.layers",       # 通过model属性访问层
        "encoder.layers",     # 通过encoder访问层
        "transformer.layers"  # 通过transformer访问层
    ]
    
    for path in layer_paths:
        try:
            components = path.split('.')
            current = model
            for component in components:
                current = getattr(current, component)
            
            if isinstance(current, nn.ModuleList):
                layers = current
                layer_path = path
                break
        except AttributeError:
            continue
    
    if layers is None:
        raise ValueError("无法找到模型的层列表")
    
    # 验证层索引
    for layer_idx in layers_to_prune:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {len(layers)-1}]")
    
    # 确保层索引是有序的
    layers_to_prune = sorted(layers_to_prune)
    
    # 创建一个新的ModuleList，不包含要剪枝的层
    new_layers = nn.ModuleList()
    for i in range(len(layers)):
        if i not in layers_to_prune:
            new_layers.append(layers[i])
    
    # 更新模型的层列表
    components = layer_path.split('.')
    current = model
    for i, component in enumerate(components):
        if i == len(components) - 1:
            setattr(current, component, new_layers)
        else:
            current = getattr(current, component)
    
    logger.info(f"成功剪枝 {len(layers_to_prune)} 层，剩余 {len(new_layers)} 层")
    
    return model

# 添加注意力头剪枝函数
def prune_attention_heads(model: nn.Module, layer_idx: int, heads_to_prune: List[int]) -> nn.Module:
    """
    剪枝指定层中的注意力头
    
    Args:
        model: Transformer模型
        layer_idx: 层索引
        heads_to_prune: 要剪枝的注意力头索引列表
        
    Returns:
        pruned_model: 剪枝后的模型
    """
    if not hasattr(model, '_model_type') or model._model_type != 'LM':
        raise ValueError("模型必须是Transformer语言模型类型")
    
    # 根据模型架构获取指定层
    layer = None
    
    # 尝试常见的层路径
    layer_paths = [
        f"layers.{layer_idx}",
        f"model.layers.{layer_idx}",
        f"encoder.layers.{layer_idx}",
        f"transformer.layers.{layer_idx}"
    ]
    
    for path in layer_paths:
        try:
            components = path.split('.')
            current = model
            for component in components:
                current = getattr(current, component)
            
            layer = current
            break
        except AttributeError:
            continue
    
    if layer is None:
        raise ValueError(f"无法找到层 {layer_idx}")
    
    # 查找注意力模块
    attn_module = None
    for name, module in layer.named_children():
        if 'attn' in name.lower() or 'attention' in name.lower():
            attn_module = module
            attn_name = name
            break
    
    if attn_module is None:
        raise ValueError(f"在层 {layer_idx} 中找不到注意力模块")
    
    # 获取注意力头数量
    num_heads = None
    for attr_name in ['num_heads', 'n_head', 'num_attention_heads']:
        if hasattr(attn_module, attr_name):
            num_heads = getattr(attn_module, attr_name)
            break
    
    if hasattr(attn_module, 'config'):
        for attr_name in ['num_attention_heads', 'n_head']:
            if hasattr(attn_module.config, attr_name):
                num_heads = getattr(attn_module.config, attr_name)
                break
    
    if num_heads is None:
        raise ValueError(f"无法确定注意力头数量")
    
    # 验证头索引
    for head_idx in heads_to_prune:
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"头索引 {head_idx} 超出范围 [0, {num_heads-1}]")
    
    # 确保头索引是有序的
    heads_to_prune = sorted(heads_to_prune)
    
    # 剪枝注意力头
    # 注意：这里的实现是模型结构相关的，不同模型可能需要不同的剪枝逻辑
    # 这里提供一个通用框架，实际使用时需要根据具体模型调整
    
    # 对于大多数Transformer模型，注意力头的剪枝涉及修改以下组件：
    # 1. 查询/键/值权重矩阵
    # 2. 输出投影矩阵
    # 3. 可能的注意力偏置
    
    # 获取注意力头维度
    head_dim = None
    if hasattr(attn_module, 'head_dim'):
        head_dim = attn_module.head_dim
    else:
        # 尝试从隐藏维度和头数推断
        for attr_name in ['hidden_size', 'hidden_dim', 'n_embd', 'embed_dim']:
            if hasattr(attn_module, attr_name):
                hidden_size = getattr(attn_module, attr_name)
                head_dim = hidden_size // num_heads
                break
            elif hasattr(attn_module.config, attr_name):
                hidden_size = getattr(attn_module.config, attr_name)
                head_dim = hidden_size // num_heads
                break
    
    if head_dim is None:
        raise ValueError("无法确定注意力头维度")
    
    # 移除头的蒙版，保留未剪枝的头
    mask = torch.ones(num_heads, dtype=torch.bool)
    mask[heads_to_prune] = False
    
    # 新的注意力头数量
    new_num_heads = num_heads - len(heads_to_prune)
    
    # 尝试更新注意力模块的参数
    # 1. 更新投影权重矩阵
    for weight_name in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'query.weight', 'key.weight', 'value.weight']:
        try:
            # 获取权重参数
            weight_path = weight_name.split('.')
            current = attn_module
            for part in weight_path[:-1]:
                current = getattr(current, part)
            param_name = weight_path[-1]
            weight = getattr(current, param_name)
            
            # 重塑以分离头维度
            old_shape = weight.shape
            reshaped = weight.view(num_heads, head_dim, -1)
            
            # 移除被剪枝的头
            new_weight = reshaped[mask].view(new_num_heads * head_dim, -1)
            
            # 更新参数
            setattr(current, param_name, nn.Parameter(new_weight))
            
            logger.info(f"更新 {weight_name}: {old_shape} -> {new_weight.shape}")
            
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"无法更新 {weight_name}: {e}")
    
    # 2. 更新输出投影矩阵
    for weight_name in ['o_proj.weight', 'out_proj.weight', 'output.weight']:
        try:
            # 获取权重参数
            weight_path = weight_name.split('.')
            current = attn_module
            for part in weight_path[:-1]:
                current = getattr(current, part)
            param_name = weight_path[-1]
            weight = getattr(current, param_name)
            
            # 重塑以分离头维度
            old_shape = weight.shape
            transposed = weight.transpose(0, 1)
            reshaped = transposed.view(-1, num_heads, head_dim)
            
            # 移除被剪枝的头
            new_weight = reshaped[:, mask, :].reshape(-1, new_num_heads * head_dim).transpose(0, 1)
            
            # 更新参数
            setattr(current, param_name, nn.Parameter(new_weight))
            
            logger.info(f"更新 {weight_name}: {old_shape} -> {new_weight.shape}")
            
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"无法更新 {weight_name}: {e}")
    
    # 3. 更新注意力头数量属性
    for attr_name in ['num_heads', 'n_head', 'num_attention_heads']:
        if hasattr(attn_module, attr_name):
            setattr(attn_module, attr_name, new_num_heads)
            logger.info(f"更新 {attr_name} 为 {new_num_heads}")
            
    # 对于有config属性的模型，也更新配置
    if hasattr(attn_module, 'config'):
        for attr_name in ['num_attention_heads', 'n_head']:
            if hasattr(attn_module.config, attr_name):
                setattr(attn_module.config, attr_name, new_num_heads)
                logger.info(f"更新 config.{attr_name} 为 {new_num_heads}")
    
    logger.info(f"成功剪枝层 {layer_idx} 中的 {len(heads_to_prune)} 个注意力头，剩余 {new_num_heads} 个头")
    
    return model
        