import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from torch.utils.data import DataLoader
import time

logger = logging.getLogger(__name__)

def evaluate_model_performance(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer: Any = None,
    batch_size: int = 4,
    max_samples: int = 100,
    task_type: Optional[str] = None,
    device: str = 'cuda'
) -> float:
    """
    评估模型性能
    
    Args:
        model: 待评估的模型
        dataset: 评估数据集
        tokenizer: 分词器（用于语言模型）
        batch_size: 批次大小
        max_samples: 最大评估样本数
        task_type: 任务类型（如果为None，将尝试从模型推断）
        device: 计算设备
        
    Returns:
        score: 性能评分（0-1之间的值，越大越好）
    """
    # 保存当前模型状态
    training = model.training
    
    # 切换到评估模式
    model.eval()
    
    # 推断任务类型
    if task_type is None:
        if hasattr(model, '_model_type'):
            if model._model_type == 'CNN':
                task_type = 'classification'
            elif model._model_type == 'LM':
                task_type = 'language_modeling'
        else:
            # 尝试从模型结构推断
            if hasattr(model, 'config'):
                if hasattr(model.config, 'problem_type'):
                    task_type = model.config.problem_type
                elif hasattr(model.config, 'architectures'):
                    if any('ForCausalLM' in arch for arch in model.config.architectures):
                        task_type = 'language_modeling'
                    elif any('ForSequenceClassification' in arch for arch in model.config.architectures):
                        task_type = 'sequence_classification'
    
    if task_type is None:
        raise ValueError("无法推断任务类型，请明确指定task_type参数")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 限制评估的样本数
    n_samples = min(max_samples, len(dataset))
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 开始评估
    start_time = time.time()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            
            try:
                # 根据任务类型处理数据
                if task_type == 'classification':
                    # 图像分类
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                
                elif task_type in ['sequence_classification', 'sentiment_analysis']:
                    # 文本分类
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch.get('labels', None)
                        
                        # 前向传播
                        outputs = model(**batch)
                        
                        # 计算损失和准确率
                        if labels is not None:
                            loss = outputs.loss
                            logits = outputs.logits
                            _, predicted = torch.max(logits, 1)
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)
                        else:
                            loss = torch.tensor(0.0).to(device)
                    else:
                        # 处理非字典格式的数据
                        inputs, labels = batch
                        if isinstance(inputs, list) and isinstance(inputs[0], str):
                            # 文本输入需要编码
                            encoded = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
                            encoded = {k: v.to(device) for k, v in encoded.items()}
                            labels = labels.to(device)
                            
                            # 前向传播
                            outputs = model(**encoded)
                        else:
                            # 已编码的输入
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = model(inputs)
                        
                        # 计算损失和准确率
                        loss = F.cross_entropy(outputs, labels)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                
                elif task_type == 'language_modeling':
                    # 语言建模
                    if isinstance(batch, dict):
                        # HuggingFace格式
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        # 前向传播
                        outputs = model(**batch)
                        
                        # 获取损失
                        if hasattr(outputs, 'loss'):
                            loss = outputs.loss
                        else:
                            # 手动计算损失
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = batch['input_ids'][..., 1:].contiguous()
                            loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                        # (inputs, targets)格式
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # 前向传播
                        outputs = model(inputs)
                        
                        # 计算损失
                        loss = F.cross_entropy(
                            outputs.view(-1, outputs.size(-1)),
                            targets.view(-1)
                        )
                    else:
                        # 处理其他格式
                        inputs = batch
                        if isinstance(inputs, list):
                            # 尝试将文本编码
                            if tokenizer is not None and isinstance(inputs[0], str):
                                encoded = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
                                encoded = {k: v.to(device) for k, v in encoded.items()}
                                outputs = model(**encoded)
                            else:
                                # 无法处理的格式
                                logger.warning(f"无法处理的数据格式: {type(batch)}")
                                continue
                        else:
                            # 假设已编码的输入
                            inputs = inputs.to(device)
                            outputs = model(inputs)
                        
                        # 假设模型输出包含loss
                        loss = getattr(outputs, 'loss', torch.tensor(0.0).to(device))
                
                else:
                    logger.warning(f"不支持的任务类型: {task_type}")
                    return 0.0
                
                # 累加损失
                total_loss += loss.item()
                
            except Exception as e:
                logger.error(f"评估过程中出错: {e}")
                continue
    
    # 计算评估指标
    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    
    # 根据任务类型计算得分
    if task_type in ['classification', 'sequence_classification', 'sentiment_analysis']:
        # 分类任务：使用准确率作为得分
        score = correct / total if total > 0 else 0.0
    elif task_type == 'language_modeling':
        # 语言建模：使用困惑度的倒数作为得分
        # 困惑度 = exp(平均损失)
        # 我们取其倒数并归一化到[0,1]
        perplexity = np.exp(avg_loss)
        score = max(0.0, min(1.0, 1.0 / perplexity))
    else:
        # 默认使用损失的归一化倒数
        score = max(0.0, min(1.0, 1.0 / (1.0 + avg_loss)))
    
    # 恢复模型状态
    model.train(training)
    
    # 记录评估结果
    logger.info(f"评估完成: 任务={task_type}, 样本数={n_samples}, 时间={time.time()-start_time:.2f}秒")
    if task_type in ['classification', 'sequence_classification', 'sentiment_analysis']:
        logger.info(f"准确率: {score:.4f}")
    elif task_type == 'language_modeling':
        logger.info(f"困惑度: {perplexity:.4f}, 得分: {score:.4f}")
    else:
        logger.info(f"损失: {avg_loss:.4f}, 得分: {score:.4f}")
    
    return score 