import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GNNPerformancePredictor(nn.Module):
    """
    基于层次化图神经网络的模型性能预测器
    
    采用分级处理策略处理Transformer模型的层次图结构：
    1. 首先处理每层的头级图，获取头级特征表示
    2. 将头级特征聚合后与层级特征融合
    3. 在增强的层级图上应用GAT进行最终处理
    4. 预测模型性能（0-1之间的值）
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super(GNNPerformancePredictor, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征编码层 - 共享特征转换
        self.feature_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 头级处理: 处理每个头级图的GAT层
        self.head_gnn = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 2, heads=heads, dropout=dropout),
            GATConv(hidden_dim // 2 * heads, hidden_dim // 2, heads=heads, dropout=dropout)
        ])
        
        # 头级特征聚合后的处理层
        self.head_aggregation = nn.Sequential(
            nn.Linear(hidden_dim // 2 * heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 层级特征增强层 - 融合头级信息
        self.layer_enhancement = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 层级处理: 在增强后的层级图上应用GAT
        self.layer_gnn = nn.ModuleList()
        self.layer_gnn.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.layer_gnn.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        
        # 全局图级特征处理
        self.global_aggregation = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层 - 预测性能
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 确保输出在0-1范围内
        )
        
    def process_hierarchical_graph(self, layer_graph, head_graphs):
        """
        处理层次化图结构，集成层级和头级信息
        
        Args:
            layer_graph: 层级图
            head_graphs: 头级图列表，每个元素对应一层的头级图
            
        Returns:
            predicted_performance: 预测的模型性能
        """
        device = layer_graph.x.device
        
        # 1. 处理层级图的初始特征
        layer_x, layer_edge_index = layer_graph.x, layer_graph.edge_index
        layer_x = self.feature_encoder(layer_x)  # [num_layers, hidden_dim]
        
        # 2. 处理每层的头级图
        aggregated_head_features = []
        
        for layer_idx, head_graph in enumerate(head_graphs):
            if layer_idx >= layer_x.size(0):  # 安全检查
                continue
                
            if head_graph is None or head_graph.x.size(0) == 0:
                # 如果某层没有头级图或头数为0，用零向量替代
                aggregated_head_features.append(torch.zeros(hidden_dim, device=device))
                continue
            
            # 获取头级图的特征和边
            head_x, head_edge_index = head_graph.x, head_graph.edge_index
            
            # 处理头级特征
            head_x = self.feature_encoder(head_x)  # [num_heads, hidden_dim]
            
            # 通过头级GNN处理
            for gnn in self.head_gnn:
                head_x = gnn(head_x, head_edge_index)
                head_x = F.relu(head_x)
            
            # 聚合该层所有头的特征 (平均池化)
            # 使用全局池化而不是简单平均，以处理可能的变长头数
            batch = torch.zeros(head_x.size(0), dtype=torch.long, device=device)
            pooled_head_x = global_mean_pool(head_x, batch)  # [1, hidden_dim * heads]
            
            # 转换维度
            pooled_head_x = self.head_aggregation(pooled_head_x.squeeze(0))  # [hidden_dim]
            
            # 添加到聚合列表
            aggregated_head_features.append(pooled_head_x)
        
        # 3. 将头级特征与层级特征融合
        enhanced_layer_x = layer_x.clone()  # 初始化为原始层特征
        
        for idx, head_feat in enumerate(aggregated_head_features):
            if idx < layer_x.size(0):  # 确保索引有效
                # 拼接层特征和对应的头级聚合特征
                combined = torch.cat([layer_x[idx], head_feat], dim=0)  # [hidden_dim * 2]
                # 通过融合层处理
                enhanced_layer_x[idx] = self.layer_enhancement(combined)  # [hidden_dim]
        
        # 4. 在增强后的层级图上应用GNN
        x = enhanced_layer_x
        
        for gnn in self.layer_gnn:
            x = gnn(x, layer_edge_index)
            x = F.relu(x)
        
        # 5. 全局池化获取图级表示
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        graph_embedding = global_mean_pool(x, batch)  # [1, hidden_dim * heads]
        
        # 全局特征处理
        graph_embedding = self.global_aggregation(graph_embedding)  # [1, hidden_dim * 2]
        
        # 6. 最终预测性能
        predicted_performance = self.output_layer(graph_embedding).squeeze(-1)
        
        return predicted_performance
        
    def forward(self, layer_graph, head_graphs):
        """
        前向传播，调用层次图处理方法
        
        Args:
            layer_graph: 层级图
            head_graphs: 头级图列表
        
        Returns:
            predicted_performance: 预测的模型性能（0-1之间的值）
        """
        return self.process_hierarchical_graph(layer_graph, head_graphs)
