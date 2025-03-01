import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Dict, List, Tuple, Optional, Union

class PruningAgent(nn.Module):
    """
    一个基于图神经网络的Agent，用于决定如何剪枝Transformer模型
    
    该Agent接收模型的层次化图表示作为输入，输出两个层次的决策：
    1. 层级决策：哪些层需要被剪枝
    2. 头级决策：对于被选中的层，哪些注意力头需要被剪枝
    """
    
    def __init__(
        self, 
        node_feature_dim: int, 
        hidden_dim: int = 128, 
        gnn_type: str = 'gcn',
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(PruningAgent, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        
        # 根据指定的GNN类型选择不同的图卷积实现
        for i in range(gnn_layers):
            if i == 0:
                input_dim = hidden_dim
            else:
                input_dim = hidden_dim
                
            if gnn_type.lower() == 'gcn':
                self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
            elif gnn_type.lower() == 'gat':
                self.gnn_layers.append(GATConv(input_dim, hidden_dim))
            else:
                raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # 层级决策网络 - 判断每一层是否需要剪枝
        self.layer_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 输出每一层的剪枝概率
            nn.Sigmoid()
        )
        
        # 头级决策网络 - 判断层内的每个注意力头是否需要剪枝
        self.head_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 输出每个注意力头的剪枝概率
            nn.Sigmoid()
        )
        
        # 价值网络 - 估计当前状态的价值
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, layer_graph, head_graphs):
        """
        前向传播，处理模型的层次化图表示并输出剪枝决策
        
        Args:
            layer_graph: 层级图，表示模型中各层之间的关系
            head_graphs: 头级图列表，每个元素表示一层中注意力头之间的关系
            
        Returns:
            layer_probs: 每层的剪枝概率
            head_probs: 每层中每个注意力头的剪枝概率 (字典，键为层索引)
            state_value: 当前状态的价值估计
        """
        # 处理层级图
        x, edge_index = layer_graph.x, layer_graph.edge_index
        
        # 特征编码
        x = self.feature_encoder(x)
        
        # 通过GNN层
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
        
        # 获取层级决策
        layer_probs = self.layer_policy(x).squeeze(-1)
        
        # 获取状态价值
        state_value = self.value_net(x.mean(dim=0))
        
        # 处理头级图，为每个被选中剪枝的层生成头级决策
        head_probs = {}
        
        for i, head_graph in enumerate(head_graphs):
            if i < len(layer_probs) and layer_probs[i] > 0.5:  # 如果该层被选中进行剪枝
                head_x, head_edge_index = head_graph.x, head_graph.edge_index
                
                # 特征编码
                head_x = self.feature_encoder(head_x)
                
                # 通过GNN层
                for gnn_layer in self.gnn_layers:
                    head_x = gnn_layer(head_x, head_edge_index)
                    head_x = F.relu(head_x)
                
                # 获取头级决策
                head_probs[i] = self.head_policy(head_x).squeeze(-1)
        
        return layer_probs, head_probs, state_value
    
    def act(self, layer_graph, head_graphs, deterministic=False):
        """
        根据当前策略执行动作采样
        
        Args:
            layer_graph: 层级图
            head_graphs: 头级图列表
            deterministic: 是否确定性采样 (True则取argmax，False则按概率采样)
            
        Returns:
            layer_actions: 每层的剪枝动作 (0或1)
            head_actions: 每个注意力头的剪枝动作 (字典，键为层索引)
            layer_probs: 每层的剪枝概率
            head_probs: 每个注意力头的剪枝概率
            state_value: 状态价值估计
        """
        # 获取策略输出
        layer_probs, head_probs, state_value = self.forward(layer_graph, head_graphs)
        
        # 层级动作采样
        if deterministic:
            layer_actions = (layer_probs > 0.5).float()
        else:
            layer_actions = torch.bernoulli(layer_probs)
        
        # 头级动作采样
        head_actions = {}
        for layer_idx, probs in head_probs.items():
            if deterministic:
                head_actions[layer_idx] = (probs > 0.5).float()
            else:
                head_actions[layer_idx] = torch.bernoulli(probs)
        
        return layer_actions, head_actions, layer_probs, head_probs, state_value

    def evaluate_actions(self, layer_graph, head_graphs, layer_actions, head_actions):
        """
        评估给定动作的对数概率和熵
        
        Args:
            layer_graph: 层级图
            head_graphs: 头级图列表
            layer_actions: 每层的剪枝动作
            head_actions: 每个注意力头的剪枝动作
            
        Returns:
            action_log_probs: 动作的对数概率
            entropy: 策略的熵
            state_value: 状态价值估计
        """
        layer_probs, head_probs, state_value = self.forward(layer_graph, head_graphs)
        
        # 计算层级动作的对数概率
        layer_log_probs = torch.log(layer_probs + 1e-10) * layer_actions + \
                          torch.log(1 - layer_probs + 1e-10) * (1 - layer_actions)
        
        # 计算头级动作的对数概率
        head_log_probs = []
        for layer_idx, actions in head_actions.items():
            if layer_idx in head_probs:
                probs = head_probs[layer_idx]
                log_probs = torch.log(probs + 1e-10) * actions + \
                            torch.log(1 - probs + 1e-10) * (1 - actions)
                head_log_probs.append(log_probs.sum())
        
        # 如果有头级动作，则合并层级和头级的对数概率
        if head_log_probs:
            action_log_probs = layer_log_probs.sum() + sum(head_log_probs)
        else:
            action_log_probs = layer_log_probs.sum()
        
        # 计算策略的熵
        layer_entropy = -(layer_probs * torch.log(layer_probs + 1e-10) + 
                           (1 - layer_probs) * torch.log(1 - layer_probs + 1e-10)).sum()
        
        head_entropy = 0
        for probs in head_probs.values():
            head_entropy += -(probs * torch.log(probs + 1e-10) + 
                             (1 - probs) * torch.log(1 - probs + 1e-10)).sum()
        
        entropy = layer_entropy + head_entropy
        
        return action_log_probs, entropy, state_value
