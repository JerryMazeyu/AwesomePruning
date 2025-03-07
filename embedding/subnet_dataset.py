import os
import torch
import numpy as np
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import time
import random
from tqdm import tqdm

from utils.prunner.unstructure import WeightsCutter
from utils.metrics import evaluate_model_performance
from utils.io import log, soft_mkdir, LogRedirectMixin
from utils.graphicalor import TransformerGraphicalor
from models.model_zoo import get_model
from config.config import CONF

class ParameterDataset(Dataset, LogRedirectMixin):
    """子网络数据集
    
    生成随机掩码对网络进行裁剪，构建子网络的图表示，并评估该子网络的性能，
    形成(子网络图表示, 性能评分)的样本对。
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 tokenizer: Any, 
                 val_dataset: Dataset, 
                 metric_fn: Callable = evaluate_model_performance,
                 num_samples: int = 100,
                 prune_rates: List[float] = None,
                 task_type: Optional[str] = None,
                 device: str = 'cuda',
                 log_path: Optional[str] = None,
                 save_dir: Optional[str] = None):
        """初始化参数数据集
        
        Args:
            model: 待剪枝的模型
            tokenizer: 对应的分词器（对语言模型需要）
            val_dataset: 用于评估模型性能的验证集
            metric_fn: 性能评估函数，默认使用evaluate_model_performance
            num_samples: 要生成的样本数量
            prune_rates: 剪枝率列表，如果不指定，默认在[0.1, 0.9]范围内随机选择
            task_type: 任务类型，可选，如不指定则尝试从模型推断
            device: 计算设备
            log_path: 日志路径
            save_dir: 数据集保存路径
        """
        super().__init__(log_path)
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.metric_fn = metric_fn
        self.num_samples = num_samples
        self.task_type = task_type
        self.device = device
        
        # 默认剪枝率范围
        if prune_rates is None:
            self.prune_rates = [round(0.1 * i, 1) for i in range(1, 10)]
        else:
            self.prune_rates = prune_rates
            
        # 设置保存路径
        if save_dir is None:
            self.save_dir = os.path.join(CONF.data_root_path, 'subnet_datasets')
        else:
            self.save_dir = save_dir
        soft_mkdir(self.save_dir)
        
        # 存储生成的样本
        self.samples = []
        self.labels = []
        
        # 初始化工具
        self.weights_cutter = WeightsCutter(model, tokenizer, log_path=self.log_path)
        
        # 如果是Transformer模型，创建图形表示器
        if hasattr(model, '_model_type') and model._model_type == 'LM':
            self.graphicalor = TransformerGraphicalor(
                model, tokenizer, log_path=self.log_path
            )
        else:
            self.graphicalor = None
            log("当前只支持Transformer模型的图表示构建", level="WARNING")
    
    def generate_samples(self, verbose: bool = True):
        """生成样本集
        
        通过随机剪枝生成多个子网络，并评估其性能，构建数据集。
        
        Args:
            verbose: 是否打印详细信息
        
        Returns:
            samples_count: 生成的样本数量
        """
        if verbose:
            log(f"开始生成子网络数据集，计划生成 {self.num_samples} 个样本...")
        
        # 记录原始模型性能作为参考
        orig_model_performance = self._evaluate_model(self.model)["perplexity"]
        if verbose:
            log(f"原始模型性能评分: {orig_model_performance:.4f}")
        
        # 使用tqdm显示进度
        pbar = tqdm(total=self.num_samples) if verbose else None
        
        # 生成指定数量的样本
        count = 0
        while count < self.num_samples:
            try:
                # 随机选择剪枝率
                prune_rate = random.choice(self.prune_rates)
                
                # 生成剪枝掩码
                masks = self.weights_cutter.prune_by_weights(
                    module='all', 
                    weights=None,  # 使用随机权重
                    prune_rate=prune_rate, 
                    verbose=False
                )
                
                # 应用掩码到模型
                model_copy = self._apply_masks_to_model(masks)
                
                # 构建子网络的图表示
                if self.graphicalor is not None:
                    if hasattr(model_copy, '_model_type') and model_copy._model_type == 'LM':
                        # 为Transformer模型构建图表示
                        graph_representation = self._build_graph_representation(model_copy)
                    else:
                        # 对于非Transformer模型，简单使用掩码扁平化作为特征
                        graph_representation = self._build_simple_representation(masks)
                else:
                    # 简单使用掩码扁平化作为特征
                    graph_representation = self._build_simple_representation(masks)
                
                # 评估剪枝后模型的性能
                performance = self._evaluate_model(model_copy)
                
                # 计算性能比率（相对于原始模型）
                performance_ratio = performance / orig_model_performance if orig_model_performance > 0 else 0
                
                # 保存样本
                sample = {
                    'graph_representation': graph_representation,
                    'masks': masks,
                    'prune_rate': prune_rate,
                    'performance': performance,
                    'performance_ratio': performance_ratio
                }
                
                self.samples.append(sample)
                self.labels.append(performance)
                
                # 更新计数和进度
                count += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_description(f"样本 {count}/{self.num_samples}, 剪枝率={prune_rate}, 性能={performance:.4f}")
                
            except Exception as e:
                log(f"生成样本时出错: {e}", level="ERROR")
                continue
        
        if pbar:
            pbar.close()
        
        if verbose:
            log(f"样本生成完成，共 {len(self.samples)} 个样本")
            self._save_samples()
        
        return len(self.samples)
    
    def _apply_masks_to_model(self, masks: List[torch.Tensor]) -> torch.nn.Module:
        """应用掩码到模型的副本
        
        Args:
            masks: 掩码列表，对应模型中的每个参数
            
        Returns:
            model_copy: 应用掩码后的模型副本
        """
        # 创建模型的深拷贝
        model_copy = type(self.model)(**vars(self.model.config) if hasattr(self.model, 'config') else {})
        model_copy.load_state_dict(self.model.state_dict())
        model_copy.to(self.device)
        
        # 应用掩码
        with torch.no_grad():
            parameterList = []
            
            # 获取所有参数
            for param in model_copy.parameters():
                if param.requires_grad:
                    parameterList.append(param)
            
            # 确保掩码和参数数量一致
            assert len(parameterList) == len(masks), f"参数数量 ({len(parameterList)}) 与掩码数量 ({len(masks)}) 不匹配"
            
            # 应用掩码
            for i, (param, mask) in enumerate(zip(parameterList, masks)):
                param.data = param.data * mask
        
        return model_copy
    
    def _build_graph_representation(self, model: torch.nn.Module) -> Dict:
        """构建子网络的图表示
        
        Args:
            model: 子网络模型
            
        Returns:
            graph_representation: 子网络的图表示
        """
        # 使用TransformerGraphicalor构建层次化图表示
        # 首先构建层级图
        layer_graph = self.graphicalor.build_layer_wise_graph(verbose=False)
        
        # 然后构建头级图
        head_graphs = self.graphicalor.build_head_wise_graphs(verbose=False)
        
        # 返回图表示
        return {
            'layer_graph': layer_graph,
            'head_graphs': head_graphs
        }
    
    def _build_simple_representation(self, masks: List[torch.Tensor]) -> torch.Tensor:
        """为非Transformer模型构建简单表示
        
        Args:
            masks: 掩码列表
            
        Returns:
            representation: 简单的特征表示
        """
        # 计算每个掩码的统计信息
        mask_stats = []
        for mask in masks:
            # 计算掩码的基本统计信息
            total = mask.numel()
            remain = mask.sum().item()
            pruned = total - remain
            pruned_ratio = pruned / total if total > 0 else 0
            
            # 如果掩码是多维的，计算每个维度的剪枝比例
            if len(mask.shape) > 1:
                dim_stats = []
                for dim in range(len(mask.shape)):
                    # 沿特定维度计算剪枝比例
                    dims_to_sum = tuple(i for i in range(len(mask.shape)) if i != dim)
                    if dims_to_sum:
                        dim_sum = mask.sum(dims_to_sum)
                        dim_total = torch.prod(torch.tensor([mask.shape[i] for i in dims_to_sum]))
                        dim_ratio = dim_sum / dim_total
                        # 计算该维度上的统计特征
                        dim_stats.extend([
                            dim_ratio.mean().item(),
                            dim_ratio.std().item(),
                            dim_ratio.min().item(),
                            dim_ratio.max().item()
                        ])
                
                mask_stats.extend([pruned_ratio] + dim_stats)
            else:
                mask_stats.append(pruned_ratio)
        
        # 返回拼接后的特征向量
        return torch.tensor(mask_stats, dtype=torch.float32)
    
    def _evaluate_model(self, model: torch.nn.Module) -> float:
        """评估模型性能
        
        Args:
            model: 要评估的模型
            
        Returns:
            performance: 性能分数 (0-1之间)
        """
        try:
            performance = self.metric_fn(
                model, 
                self.val_dataset, 
                self.tokenizer, 
                batch_size=4,  # 使用小批量以节省内存
                max_samples=100,  # 使用部分数据评估以加速
                task_type=self.task_type, 
                device=self.device
            )
            return performance
        except Exception as e:
            log(f"模型评估失败: {e}", level="ERROR")
            # 失败时返回0性能
            return 0.0
    
    def _save_samples(self):
        """保存样本到文件"""
        # 使用pickle保存图结构和性能数据
        save_path = os.path.join(self.save_dir, "subnet_performance_dataset.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self.samples, f)
        
        # 同时保存一个可读的配置和性能记录
        summary_path = os.path.join(self.save_dir, "dataset_summary.json")
        summary = [
            {
                'sample_id': i,
                'prune_rate': sample['prune_rate'],
                'performance': float(sample['performance']),
                'performance_ratio': float(sample['performance_ratio'])
            }
            for i, sample in enumerate(self.samples)
        ]
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log(f"已保存 {len(self.samples)} 个样本到 {save_path}")
    
    def __len__(self):
        """返回数据集样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.samples[idx]['graph_representation'], self.labels[idx]


if __name__ == "__main__":
    """子网络数据集测试用例"""
    import argparse
    from data import get_dataset
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="子网络数据集测试")
    parser.add_argument("--model", type=str, default="Qwen2.5-3B", help="模型名称")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="验证数据集名称")
    parser.add_argument("--task_type", type=str, default="language_modeling", 
                        help="任务类型：如language_modeling, sequence_classification等")
    parser.add_argument("--num_samples", type=int, default=5, help="要生成的样本数量")
    parser.add_argument("--device", type=str, default="cuda:1", help="运行设备")
    args = parser.parse_args()
    
    log(f"正在测试子网络数据集生成，使用模型: {args.model}, 数据集: {args.dataset}")
    
    try:
        # 获取模型和分词器
        log("正在加载模型...")
        model, tokenizer = get_model(args.model, cache_dir=CONF.cache_dir, add_padding_token=True)
        
        # 获取验证数据集
        log("正在加载验证数据集...")
        val_dataset = get_dataset(args.dataset)
        
        # 创建输出目录
        output_dir = os.path.join(CONF.data_root_path, f'subnet_data_{args.model}_{args.dataset}')
        soft_mkdir(output_dir)
        
        # 创建子网络数据集
        log("初始化子网络数据集...")
        subnet_dataset = ParameterDataset(
            model=model,
            tokenizer=tokenizer,
            val_dataset=val_dataset,
            num_samples=args.num_samples,
            task_type=args.task_type,
            device=args.device,
            save_dir=output_dir
        )
        
        # 生成样本
        log(f"开始生成 {args.num_samples} 个样本...")
        sample_count = subnet_dataset.generate_samples(verbose=True)
        log(f"样本生成完成，共 {sample_count} 个样本")
        
        # 创建数据加载器并测试数据集访问
        dataloader = DataLoader(
            subnet_dataset,
            batch_size=2,
            shuffle=True
        )
        
        # 打印样本信息
        log("数据集样本详情:")
        for i, (graph_repr, performance) in enumerate(dataloader):
            log(f"批次 {i+1}:")
            
            # 打印性能分数
            if isinstance(performance, torch.Tensor):
                performance_values = performance.tolist()
                log(f"  性能分数: {performance_values}")
            else:
                log(f"  性能分数: {performance}")
            
            # 如果是Transformer模型的图表示，打印图结构信息
            if isinstance(graph_repr, dict) and 'layer_graph' in graph_repr:
                layer_graph = graph_repr['layer_graph']
                head_graphs = graph_repr.get('head_graphs', [])
                
                log(f"  层级图: 包含 {layer_graph.x.shape[0]} 个节点, {layer_graph.edge_index.shape[1]//2} 条边")
                log(f"  头级图: {len(head_graphs)} 个")
            elif isinstance(graph_repr, torch.Tensor):
                log(f"  简单表示: 特征维度 {graph_repr.shape}")
            
            # 只显示第一个批次的信息
            break
        
        # 分析并可视化结果
        summary_file = os.path.join(output_dir, "dataset_summary.json")
        if os.path.exists(summary_file):
            log("数据集统计分析:")
            
            # 读取统计数据
            df = pd.read_json(summary_file)
            
            # 打印统计摘要
            log("\n统计摘要:")
            summary_stats = df[['prune_rate', 'performance', 'performance_ratio']].describe()
            log(f"{summary_stats}")
            
            # 绘制剪枝率与性能的关系图
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='prune_rate', y='performance', data=df)
            plt.title('剪枝率与模型性能的关系')
            plt.xlabel('剪枝率')
            plt.ylabel('性能评分')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'prune_rate_vs_performance.png'))
            
            log(f"图表已保存到 {output_dir}")
        
        log("子网络数据集测试完成!")
        
    except Exception as e:
        log(f"测试过程中出错: {e}", level="ERROR")
        import traceback
        log(traceback.format_exc(), level="ERROR") 