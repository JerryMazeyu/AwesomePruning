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

    if task_type == 'classification':
        raise NotImplementedError("Classification task is not implemented yet.")
 
    elif task_type == 'sequence_classification':
        raise NotImplementedError("Sequence classification task is not implemented yet.")
    
    elif task_type == 'language_modeling':
        print("Language modeling task is running...")
        total_loss = 0.0
        total_tokens = 0
    
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break

                try:
                    # 根据任务类型处理数据
                    if isinstance(batch, dict):
                        if 'input_ids' in batch and isinstance(batch['input_ids'], list) and isinstance(batch['input_ids'][0], str):
                            # 需要先使用tokenizer处理文本
                            if tokenizer is None:
                                print(f"输入是文本但未提供tokenizer")
                                continue

                            # 将文本转换为token ids
                            encoded = tokenizer(
                                batch['input_ids'],  # 这里是文本列表
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=512  # 可根据需要调整
                            )
                            batch_on_device = {k: v.to(device) for k, v in encoded.items()}
                        else:
                            # 已经是tokenized的张量数据，只需移至正确设备
                            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                              for k, v in batch.items()}
                        # 对于没有labels的数据集（常见情况），创建移位labels用于计算下一个token的预测
                        if 'input_ids' in batch_on_device:
                            # 将input_ids右移一位作为标签
                            input_ids = batch_on_device['input_ids']
                            
                            # 如果有attention_mask，使用它来计算有效token数
                            if 'attention_mask' in batch_on_device:
                                attention_mask = batch_on_device['attention_mask']
                                # 有效token计数（忽略padding）
                                batch_tokens = attention_mask.sum().item()
                            else:
                                # 如果没有mask，假设所有token都有效
                                batch_tokens = input_ids.numel()
                            
                            outputs = model(**batch_on_device)
                            # 如果模型已经计算了loss
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                batch_loss = outputs.loss * batch_tokens  # 转换为总损失
                            else:
                                # 否则手动计算loss
                                # 输入序列移位：预测目标是下一个token
                                shift_logits = outputs.logits[..., :-1, :].contiguous()
                                shift_labels = input_ids[..., 1:].contiguous()
                                
                                # 使用交叉熵损失
                                loss_fct = nn.CrossEntropyLoss(reduction='sum')  # 使用sum便于计算总token的损失
                                batch_loss = loss_fct(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1)
                                )

                                # 计算有效token数量（排除padding）
                                if 'attention_mask' in batch_on_device:
                                    # 右移attention_mask来匹配shift_labels
                                    shift_mask = attention_mask[..., 1:].contiguous()
                                    batch_tokens = shift_mask.sum().item()
                                else:
                                    batch_tokens = shift_labels.numel()
                            
                            # 累加总损失和token数
                            total_loss += batch_loss.item()
                            total_tokens += batch_tokens
                    else:
                        raise ValueError(f"Unsupported input format: {type(batch)}")
                except Exception as e:
                    print(f"Error in language modeling task: {e}")
                    continue
            
            avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
            perplexity = np.exp(total_loss / total_tokens)
            score = max(0.0, min(1.0, 1.0 / perplexity))
            print(f"Perplexity: {perplexity:.4f}, Score: {score:.4f}")
        
        model.train(training)
        return {'loss': avg_loss, 'perplexity': perplexity, 'score': score}
