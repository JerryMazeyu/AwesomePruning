# rl/reward/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import logging
import time
from typing import List, Dict, Tuple, Any, Optional
import random

from rl.reward_models import GNNPerformancePredictor
from utils.io import LogRedirectMixin

logger = logging.getLogger(__name__)

class GraphPerformanceDataset(Dataset):
    """
    图性能数据集
    
    用于加载和处理图表示和对应的性能数据
    """
    
    def __init__(self, data_path, device='cuda'):
        """
        初始化数据集
        
        Args:
            data_path: 数据路径
            device: 设备
        """
        self.device = device
        
        # 加载数据
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        logger.info(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 获取图表示和性能
        layer_graph = sample['layer_graph']
        head_graphs = sample['head_graphs']
        performance = sample['performance']
        
        # 将性能转换为tensor
        performance = torch.tensor(performance, dtype=torch.float32)
        
        return {
            'layer_graph': layer_graph,
            'head_graphs': head_graphs,
            'performance': performance
        }

class GNNPerformancePredictorTrainer(LogRedirectMixin):
    """
    GNN性能预测器训练器
    """
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'cuda',
        log_path: Optional[str] = None,
        seed: int = 42
    ):
        """
        初始化训练器
        
        Args:
            data_path: 数据路径
            output_dir: 输出目录
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: GAT层数
            heads: 注意力头数
            dropout: Dropout率
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            early_stopping_patience: 早停耐心
            device: 设备
            log_path: 日志路径
            seed: 随机种子
        """
        super().__init__(log_path)
        
        self.data_path = data_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.seed = seed
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建模型
        self.model = GNNPerformancePredictor(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        ).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 创建调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 创建损失函数
        self.criterion = nn.MSELoss()
        
        # 加载数据集
        self.prepare_dataset()
    
    def prepare_dataset(self):
        """准备数据集"""
        dataset = GraphPerformanceDataset(self.data_path, self.device)
        
        # 分割数据集
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        self.log(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def train(self):
        """训练模型"""
        self.log("开始训练...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self._train_epoch()
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self._validate()
            val_losses.append(val_loss)
            
            # 调整学习率
            self.scheduler.step(val_loss)
            
            # 记录信息
            self.log(f"Epoch [{epoch+1}/{self.epochs}] - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"Time: {time.time() - start_time:.2f}s")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存最佳模型
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
                self.log(f"模型已保存 (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.log(f"早停 - 连续 {self.early_stopping_patience} 个epoch没有改善")
                    break
        
        # 绘制损失曲线
        self._plot_training_curves(train_losses, val_losses)
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model.pth")))
        
        return self.model
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            
            layer_graph = batch['layer_graph']
            head_graphs = batch['head_graphs']
            true_performance = batch['performance'].to(self.device)
            
            # 前向传播
            pred_performance = self.model(layer_graph, head_graphs)
            
            # 计算损失
            loss = self.criterion(pred_performance, true_performance)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                layer_graph = batch['layer_graph']
                head_graphs = batch['head_graphs']
                true_performance = batch['performance'].to(self.device)
                
                # 前向传播
                pred_performance = self.model(layer_graph, head_graphs)
                
                # 计算损失
                loss = self.criterion(pred_performance, true_performance)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _plot_training_curves(self, train_losses, val_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'loss_curves.png'))
        plt.close()