# rl/reward/data_generator.py
import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from concurrent.futures import ProcessPoolExecutor
import pickle

from utils.graphicalor import TransformerGraphicalor
from utils.prunner.structure import prune_transformer_layers, prune_attention_heads
from utils.metrics import evaluate_model_performance
from utils.io import LogRedirectMixin

logger = logging.getLogger(__name__)

class ModelPerformanceDataGenerator(LogRedirectMixin):
    """
    生成"模型-性能"数据对，用于训练GNN性能预测器
    
    通过对基础模型应用不同的裁剪策略，并评估裁剪后模型的性能，
    生成用于训练GNN性能预测模型的数据集。
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        tokenizer: Any,
        dataset: Any,
        task_type: str,
        output_dir: str,
        num_samples: int = 200,
        batch_size: int = 4,
        device: str = 'cuda',
        log_path: Optional[str] = None,
        seed: int = 42
    ):
        """
        初始化数据生成器
        
        Args:
            base_model: 基础模型
            tokenizer: 分词器
            dataset: 评估数据集
            task_type: 任务类型（如"language_modeling", "sequence_classification"等）
            output_dir: 输出目录
            num_samples: 要生成的样本数量
            batch_size: 评估时的批次大小
            device: 设备
            log_path: 日志路径
            seed: 随机种子
        """
        super().__init__(log_path)
        
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.task_type = task_type
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化图结构提取器
        self.graphicalor = TransformerGraphicalor(base_model, tokenizer, log_path=log_path)
        
        # 校准模型获取统计信息
        try:
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size)))
            self.graphicalor.calibrate(batch)
        except Exception as e:
            self.log(f"模型校准过程出错: {e}", level="WARNING")
        
        # 构建初始层次图
        self.layer_graph, self.head_graphs = self.graphicalor.build_hierarchical_graph(
            similarity_metric="cosine",
            similarity_threshold=0.5,
            verbose=False
        )
        
        # 获取模型结构信息
        self.num_layers = len(self.layer_graph.x)
        self.num_heads_per_layer = [g.x.shape[0] for g in self.head_graphs]
        
        # 测量基础模型性能
        self.base_performance = self._evaluate_model(base_model)
        self.log(f"基础模型性能: {self.base_performance:.4f}")
        
        # 数据样本列表
        self.samples = []
    
    def generate_dataset(self):
        """
        生成数据集
        
        通过随机裁剪策略，生成多个模型变体及其性能，构建训练数据集
        """
        self.log(f"开始生成数据集，目标样本数: {self.num_samples}")
        start_time = time.time()
        
        # 追踪进度
        pbar = tqdm(total=self.num_samples, desc="生成样本")
        
        # 记录已有的裁剪配置，避免重复
        seen_configs = set()
        
        while len(self.samples) < self.num_samples:
            # 创建当前模型的副本
            model_copy = self._clone_model()
            
            # 生成随机裁剪策略
            pruning_config = self._generate_random_pruning_config()
            
            # 转换为字符串以便比较
            config_str = self._config_to_str(pruning_config)
            
            # 如果已经见过这个配置，则跳过
            if config_str in seen_configs:
                continue
            
            seen_configs.add(config_str)
            
            # 应用裁剪策略
            try:
                model_copy = self._apply_pruning(model_copy, pruning_config)
            except Exception as e:
                self.log(f"应用裁剪失败: {e}", level="WARNING")
                continue
            
            # 构建裁剪后模型的图表示
            try:
                graphicalor = TransformerGraphicalor(model_copy, self.tokenizer)
                # 校准模型获取统计信息
                batch = next(iter(torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)))
                graphicalor.calibrate(batch)
                
                # 构建层次图
                layer_graph, head_graphs = graphicalor.build_hierarchical_graph(
                    similarity_metric="cosine",
                    similarity_threshold=0.5,
                    verbose=False
                )
            except Exception as e:
                self.log(f"构建图表示失败: {e}", level="WARNING")
                continue
            
            # 评估模型性能
            try:
                performance = self._evaluate_model(model_copy)
            except Exception as e:
                self.log(f"评估性能失败: {e}", level="WARNING")
                continue
            
            # 将裁剪配置、图表示和性能添加到样本列表
            sample = {
                'pruning_config': pruning_config,
                'layer_graph': layer_graph,
                'head_graphs': head_graphs,
                'performance': performance,
                'performance_ratio': performance / self.base_performance
            }
            
            self.samples.append(sample)
            pbar.update(1)
            
            # 每10个样本保存一次
            if len(self.samples) % 10 == 0:
                self._save_samples()
                
            # 记录进度
            if len(self.samples) % 20 == 0:
                self.log(f"已生成 {len(self.samples)} 个样本，当前性能: {performance:.4f}, "
                        f"性能比例: {(performance / self.base_performance):.4f}")
        
        pbar.close()
        
        # 最终保存
        self._save_samples()
        
        self.log(f"数据集生成完毕，共 {len(self.samples)} 个样本，"
                f"耗时: {(time.time() - start_time) / 60:.2f} 分钟")
        
        # 返回样本路径
        return os.path.join(self.output_dir, "model_performance_dataset.pkl")
    
    def _clone_model(self):
        """复制模型"""
        return type(self.base_model)(**self.base_model.config.__dict__)
    
    def _generate_random_pruning_config(self):
        """
        生成随机裁剪配置
        
        包括三种裁剪策略:
        1. 结构化裁剪 - 层
        2. 结构化裁剪 - 注意力头
        3. 非结构化裁剪 - 权重稀疏
        
        Returns:
            pruning_config: 裁剪配置
        """
        # 随机决定是否使用每种裁剪类型
        use_layer_pruning = random.random() < 0.7
        use_head_pruning = random.random() < 0.8
        use_weight_pruning = random.random() < 0.6
        
        pruning_config = {
            'layers_to_prune': [],
            'heads_to_prune': {},
            'weight_sparsity': 0.0
        }
        
        # 层裁剪
        if use_layer_pruning and self.num_layers > 2:
            # 随机选择要裁剪的层数量 (最多裁剪一半的层)
            num_layers_to_prune = random.randint(1, max(1, self.num_layers // 2))
            # 随机选择要裁剪的层
            pruning_config['layers_to_prune'] = sorted(random.sample(range(self.num_layers), num_layers_to_prune))
        
        # 注意力头裁剪
        if use_head_pruning:
            for layer_idx in range(self.num_layers):
                # 跳过已经被裁剪的层
                if layer_idx in pruning_config['layers_to_prune']:
                    continue
                
                num_heads = self.num_heads_per_layer[layer_idx]
                if num_heads > 1:
                    # 随机决定是否裁剪这一层的头
                    if random.random() < 0.7:
                        # 随机选择要裁剪的头数量 (最多裁剪一半的头)
                        num_heads_to_prune = random.randint(1, max(1, num_heads // 2))
                        # 随机选择要裁剪的头
                        pruning_config['heads_to_prune'][layer_idx] = sorted(
                            random.sample(range(num_heads), num_heads_to_prune)
                        )
        
        # 权重稀疏裁剪
        if use_weight_pruning:
            # 随机选择权重稀疏度 (0.05-0.5)
            pruning_config['weight_sparsity'] = random.uniform(0.05, 0.5)
        
        return pruning_config
    
    def _config_to_str(self, config):
        """将配置转换为字符串，用于比较"""
        return json.dumps(config, sort_keys=True)
    
    def _apply_pruning(self, model, pruning_config):
        """
        应用裁剪配置到模型
        
        Args:
            model: 要裁剪的模型
            pruning_config: 裁剪配置
            
        Returns:
            pruned_model: 裁剪后的模型
        """
        # 1. 应用层裁剪
        if pruning_config['layers_to_prune']:
            model = prune_transformer_layers(model, pruning_config['layers_to_prune'])
        
        # 2. 应用头裁剪
        for layer_idx, heads in pruning_config['heads_to_prune'].items():
            if heads:
                model = prune_attention_heads(model, layer_idx, heads)
        
        # 3. 应用权重稀疏裁剪
        if pruning_config['weight_sparsity'] > 0:
            model = self._apply_weight_pruning(model, pruning_config['weight_sparsity'])
        
        return model
    
    def _apply_weight_pruning(self, model, sparsity):
        """
        应用非结构化权重裁剪
        
        Args:
            model: 要裁剪的模型
            sparsity: 目标稀疏度
            
        Returns:
            pruned_model: 裁剪后的模型
        """
        # 实现简单的幅度裁剪
        for name, param in model.named_parameters():
            if 'weight' in name:
                # 计算阈值
                threshold = torch.quantile(param.data.abs().flatten(), sparsity)
                # 创建掩码
                mask = param.data.abs() > threshold
                # 应用掩码
                param.data = param.data * mask
        
        return model
    
    def _evaluate_model(self, model):
        """
        评估模型性能
        
        Args:
            model: 要评估的模型
            
        Returns:
            performance: 性能分数 (0-1之间)
        """
        try:
            performance = evaluate_model_performance(
                model, 
                self.dataset, 
                self.tokenizer, 
                batch_size=self.batch_size, 
                max_samples=100,  # 使用部分数据评估以加速
                task_type=self.task_type, 
                device=self.device
            )
            return performance
        except Exception as e:
            self.log(f"模型评估失败: {e}", level="ERROR")
            # 失败时返回0性能
            return 0.0
    
    def _save_samples(self):
        """保存样本到文件"""
        # 使用pickle保存图结构和性能数据
        save_path = os.path.join(self.output_dir, "model_performance_dataset.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self.samples, f)
        
        # 同时保存一个可读的配置和性能记录
        summary_path = os.path.join(self.output_dir, "dataset_summary.json")
        summary = [
            {
                'sample_id': i,
                'pruning_config': sample['pruning_config'],
                'performance': float(sample['performance']),
                'performance_ratio': float(sample['performance_ratio'])
            }
            for i, sample in enumerate(self.samples)
        ]
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"已保存 {len(self.samples)} 个样本到 {save_path}")