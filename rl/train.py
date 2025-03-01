import os
import argparse
import torch
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import copy
from typing import Dict, List, Tuple, Optional, Union, Any

from rl.agent import PruningAgent
from rl.environment import PruningEnvironment
from rl.policy import PPOTrainer
from utils.graphicalor import TransformerGraphicalor
from models import get_model
from data import get_dataset
from utils.metrics import evaluate_model_performance  # 假设有这个函数
from utils.io import LogRedirectMixin

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RL-based Transformer pruning')
    
    # 模型和数据参数
    parser.add_argument('--model', type=str, default='Qwen2.5-3B', help='Model name')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset name')
    parser.add_argument('--task_type', type=str, default='language_modeling', help='Task type: language_modeling, sequence_classification')
    
    # 剪枝参数
    parser.add_argument('--target_sparsity', type=float, default=0.3, help='Target sparsity ratio')
    parser.add_argument('--max_pruning_ratio', type=float, default=0.5, help='Maximum pruning ratio')
    parser.add_argument('--sparsity_weight', type=float, default=0.5, help='Weight for sparsity reward')
    parser.add_argument('--performance_weight', type=float, default=0.5, help='Weight for performance reward')
    
    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--steps_per_iteration', type=int, default=128, help='Number of steps per iteration')
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency of evaluation during training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for PPO updates')
    
    # 图神经网络参数
    parser.add_argument('--gnn_type', type=str, default='gcn', help='GNN type: gcn or gat')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of GNN')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    return args

def create_output_dir(args):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.model}_{args.dataset}_{args.target_sparsity}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    return output_dir

def evaluate_model(model, dataset, tokenizer):
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        dataset: 评估数据集
        tokenizer: 分词器
        
    Returns:
        score: 性能评分 (0-1)
    """
    # TODO: 实现真实的模型评估逻辑
    return evaluate_model_performance(model, dataset, tokenizer)

def plot_training_stats(stats, output_dir):
    """绘制训练过程的统计信息"""
    # 创建多子图
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制奖励曲线
    axs[0, 0].plot(stats['iterations'], stats['rewards'])
    axs[0, 0].set_title('Average Reward')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)
    
    # 绘制损失曲线
    axs[0, 1].plot(stats['iterations'], stats['policy_loss'], label='Policy Loss')
    axs[0, 1].plot(stats['iterations'], stats['value_loss'], label='Value Loss')
    axs[0, 1].set_title('Losses')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 绘制稀疏度曲线
    axs[1, 0].plot(stats['iterations'], stats['sparsity'])
    axs[1, 0].set_title('Sparsity')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Sparsity Ratio')
    axs[1, 0].grid(True)
    
    # 绘制性能比率曲线
    axs[1, 1].plot(stats['iterations'], stats['performance_ratio'])
    axs[1, 1].set_title('Performance Ratio')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Performance / Original')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_stats.png'))
    plt.close()

class PruningEnvironment(gym.Env, LogRedirectMixin):
    """
    符合Gymnasium接口的Transformer模型剪枝环境
    
    该环境支持通过强化学习对Transformer模型进行结构化剪枝，
    基于模型的层次图表示进行决策，并根据剪枝后模型的性能和稀疏度提供奖励。
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        val_dataset: torch.utils.data.Dataset,
        metric_fn: callable,
        device: str = 'cuda',
        sparsity_weight: float = 0.5,
        performance_weight: float = 0.5,
        target_sparsity: float = 0.3,
        max_pruning_ratio: float = 0.5,
        seed: int = 42,
        log_path: Optional[str] = None,
        render_mode: Optional[str] = None
    ):
        """
        初始化剪枝环境
        
        Args:
            model: 预训练的Transformer模型
            tokenizer: 模型对应的tokenizer
            val_dataset: 用于评估剪枝后模型性能的验证数据集
            metric_fn: 评估模型性能的度量函数
            device: 运行设备
            sparsity_weight: 奖励计算中稀疏度的权重
            performance_weight: 奖励计算中性能的权重
            target_sparsity: 目标稀疏度（剪枝比例）
            max_pruning_ratio: 最大允许的剪枝比例
            seed: 随机种子
            log_path: 日志路径
            render_mode: 渲染模式
        """
        # 初始化LogRedirectMixin
        super(LogRedirectMixin, self).__init__(log_path)
        gym.Env.__init__(self)
        
        self.model = model
        self.original_model = copy.deepcopy(model)
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.metric_fn = metric_fn
        self.device = device
        self.render_mode = render_mode
        
        # 奖励计算相关参数
        self.sparsity_weight = sparsity_weight
        self.performance_weight = performance_weight
        self.target_sparsity = target_sparsity
        self.max_pruning_ratio = max_pruning_ratio
        
        # 设置随机种子
        self.seed(seed)
        
        # 初始化图结构提取器
        self.graphicalor = TransformerGraphicalor(model, tokenizer, log_path=log_path)
        
        # 记录原始模型的性能，作为性能下降的基准
        self.original_performance = self._evaluate_model(self.original_model)
        self.log(f"原始模型性能: {self.original_performance}")
        
        # 初始化环境状态和空间
        self._setup_spaces()
        
        # 重置环境以初始化状态
        self.reset(seed=seed)
    
    def _setup_spaces(self):
        """设置动作空间和观察空间"""
        # 校准模型获取统计信息和梯度，构建层次化图表示
        try:
            batch = next(iter(torch.utils.data.DataLoader(self.val_dataset, batch_size=4)))
            self.graphicalor.calibrate(batch)
        except Exception as e:
            self.log(f"模型校准过程出错: {e}", level="WARNING")
        
        # 构建层次化图表示
        self.layer_graph, self.head_graphs = self.graphicalor.build_hierarchical_graph(
            similarity_metric="cosine",
            similarity_threshold=0.5,
            verbose=False
        )
        
        # 获取层数和每层的头数
        self.num_layers = len(self.layer_graph.x)
        self.num_heads_per_layer = [g.x.shape[0] for g in self.head_graphs]
        self.max_heads = max(self.num_heads_per_layer) if self.num_heads_per_layer else 0
        
        # 定义动作空间：离散动作 - 每层是否剪枝(0/1)，各层中每个头是否剪枝(0/1)
        # 使用MultiDiscrete，每个元素可以取值[0,1]
        total_actions = self.num_layers + sum(self.num_heads_per_layer)
        self.action_space = spaces.MultiDiscrete([2] * total_actions)
        
        # 定义观察空间：
        # 1. 层特征矩阵(flatten)
        # 2. 层间连接信息
        # 3. 头特征矩阵(flatten)
        # 4. 头间连接信息
        # 5. 当前稀疏度和性能指标
        
        # 为简化实现，我们将图表示转换为固定大小的向量
        # 实际应用中，可能需要使用图神经网络直接处理图结构
        feature_dim = 512  # 特征向量维度
        
        self.observation_space = spaces.Dict({
            # 层级图表示
            "layer_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(min(feature_dim, self.num_layers * self.layer_graph.x.shape[1]),),
                dtype=np.float32
            ),
            # 头级图表示 (所有层的头特征合并)
            "head_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(min(feature_dim, sum(h * g.x.shape[1] for h, g in zip(self.num_heads_per_layer, self.head_graphs))),),
                dtype=np.float32
            ),
            # 状态信息
            "state_info": spaces.Box(
                low=np.array([0, 0, 0]),
                high=np.array([1, 1, 1]),
                shape=(3,),
                dtype=np.float32
            ),
            # 已剪枝的层和头信息
            "pruned_mask": spaces.Box(
                low=0, high=1,
                shape=(self.num_layers + sum(self.num_heads_per_layer),),
                dtype=np.int8
            )
        })
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境状态
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 重置后的环境状态（层次化图表示）
            info: 附加信息
        """
        # 设置随机种子
        if seed is not None:
            self.seed(seed)
        
        # 重置模型为原始模型
        self.model = copy.deepcopy(self.original_model)
        
        # 重新构建层次化图表示
        self.graphicalor = TransformerGraphicalor(self.model, self.tokenizer, log_path=self.log_path)
        
        # 校准模型获取统计信息和梯度
        try:
            batch = next(iter(torch.utils.data.DataLoader(self.val_dataset, batch_size=4)))
            self.graphicalor.calibrate(batch)
        except Exception as e:
            self.log(f"模型校准过程出错: {e}", level="WARNING")
        
        # 构建层次化图表示
        self.layer_graph, self.head_graphs = self.graphicalor.build_hierarchical_graph(
            similarity_metric="cosine",
            similarity_threshold=0.5,
            verbose=False
        )
        
        # 当前环境信息
        self.current_sparsity = 0.0
        self.current_performance = self.original_performance
        self.pruned_layers = set()
        self.pruned_heads = {}  # 字典，键为层索引，值为被剪枝的头索引的集合
        
        # 返回观察值
        observation = self._get_observation()
        info = {}
        
        # 如果指定了渲染模式，则渲染
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action):
        """
        执行一步环境交互
        
        Args:
            action: 离散动作，表示对各层和各头的剪枝决策
            
        Returns:
            observation: 执行动作后的新状态
            reward: 获得的奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 附加信息
        """
        # 记录操作前的模型状态
        pre_performance = self.current_performance
        pre_sparsity = self.current_sparsity
        
        # 解析动作：前num_layers个元素表示对层的决策，剩余元素表示对各层中各头的决策
        layer_actions = action[:self.num_layers]
        head_actions_flat = action[self.num_layers:]
        
        # 将头决策组织成字典格式，键为层索引，值为该层各头的决策
        head_actions = {}
        start_idx = 0
        for i, num_heads in enumerate(self.num_heads_per_layer):
            if layer_actions[i] == 1:  # 只处理被选中剪枝的层
                head_actions[i] = head_actions_flat[start_idx:start_idx+num_heads]
            start_idx += num_heads
        
        # 执行剪枝操作
        self._prune_model(layer_actions, head_actions)
        
        # 评估剪枝后模型性能
        self.current_performance = self._evaluate_model(self.model)
        
        # 计算当前稀疏度
        self.current_sparsity = self._calculate_sparsity()
        
        # 计算奖励
        reward = self._calculate_reward(pre_performance, pre_sparsity)
        
        # 判断是否结束
        terminated = (self.current_sparsity >= self.target_sparsity) or (self.current_performance < 0.7 * self.original_performance)
        truncated = False
        
        # 获取新的观察值
        observation = self._get_observation()
        
        # 附加信息
        info = {
            'sparsity': self.current_sparsity,
            'performance': self.current_performance,
            'performance_ratio': self.current_performance / self.original_performance,
            'pruned_layers': self.pruned_layers,
            'pruned_heads': self.pruned_heads
        }
        
        # 如果指定了渲染模式，则渲染
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        获取当前环境状态（层次化图表示）
        
        Returns:
            observation: 当前环境状态
        """
        # 1. 提取层级图特征
        layer_features = self.layer_graph.x.flatten().detach().cpu().numpy()
        max_layer_features = self.observation_space["layer_features"].shape[0]
        if len(layer_features) > max_layer_features:
            layer_features = layer_features[:max_layer_features]
        else:
            # 填充
            padding = np.zeros(max_layer_features - len(layer_features), dtype=np.float32)
            layer_features = np.concatenate([layer_features, padding])
        
        # 2. 提取头级图特征
        head_features_list = []
        for g in self.head_graphs:
            head_features_list.append(g.x.flatten().detach().cpu().numpy())
        
        head_features = np.concatenate(head_features_list) if head_features_list else np.array([])
        max_head_features = self.observation_space["head_features"].shape[0]
        if len(head_features) > max_head_features:
            head_features = head_features[:max_head_features]
        else:
            # 填充
            padding = np.zeros(max_head_features - len(head_features), dtype=np.float32)
            head_features = np.concatenate([head_features, padding])
        
        # 3. 状态信息
        state_info = np.array([
            self.current_sparsity,
            self.current_performance / self.original_performance,
            len(self.pruned_layers) / self.num_layers if self.num_layers > 0 else 0
        ], dtype=np.float32)
        
        # 4. 已剪枝的层和头掩码
        pruned_mask = np.zeros(self.num_layers + sum(self.num_heads_per_layer), dtype=np.int8)
        
        # 设置已剪枝层的掩码
        for layer_idx in self.pruned_layers:
            if 0 <= layer_idx < self.num_layers:
                pruned_mask[layer_idx] = 1
        
        # 设置已剪枝头的掩码
        head_start_idx = self.num_layers
        for layer_idx in range(self.num_layers):
            heads_in_layer = self.num_heads_per_layer[layer_idx]
            if layer_idx in self.pruned_heads:
                for head_idx in self.pruned_heads[layer_idx]:
                    if 0 <= head_idx < heads_in_layer:
                        pruned_mask[head_start_idx + head_idx] = 1
            head_start_idx += heads_in_layer
        
        # 组合成字典观察空间
        observation = {
            "layer_features": layer_features.astype(np.float32),
            "head_features": head_features.astype(np.float32),
            "state_info": state_info,
            "pruned_mask": pruned_mask
        }
        
        return observation
    
    def _prune_model(self, layer_actions, head_actions):
        """
        根据给定的动作对模型执行剪枝操作
        
        Args:
            layer_actions: 每层的剪枝决策
            head_actions: 每个注意力头的剪枝决策 (字典，键为层索引)
        """
        from utils.prunner.structure import prune_transformer_layers, prune_attention_heads
        
        # 获取要剪枝的层索引
        layers_to_prune = []
        for i, action in enumerate(layer_actions):
            if action == 1:
                layers_to_prune.append(i)
                self.pruned_layers.add(i)
        
        # 执行层级剪枝
        if layers_to_prune:
            try:
                prune_transformer_layers(self.model, layers_to_prune)
                self.log(f"已剪枝以下层: {layers_to_prune}")
            except Exception as e:
                self.log(f"层级剪枝失败: {e}", level="ERROR")
        
        # 获取要剪枝的注意力头
        for layer_idx, head_action in head_actions.items():
            heads_to_prune = []
            for i, action in enumerate(head_action):
                if action == 1:
                    heads_to_prune.append(i)
                    
                    # 更新已剪枝头记录
                    if layer_idx not in self.pruned_heads:
                        self.pruned_heads[layer_idx] = set()
                    self.pruned_heads[layer_idx].add(i)
            
            # 执行头级剪枝
            if heads_to_prune:
                try:
                    prune_attention_heads(self.model, layer_idx, heads_to_prune)
                    self.log(f"已剪枝层 {layer_idx} 中的以下注意力头: {heads_to_prune}")
                except Exception as e:
                    self.log(f"头级剪枝失败: {e}", level="ERROR")
    
    def _evaluate_model(self, model):
        """
        评估模型性能
        
        Args:
            model: 要评估的模型
            
        Returns:
            performance: 模型性能得分
        """
        try:
            # 使用提供的评估函数
            performance = self.metric_fn(model, self.val_dataset, self.tokenizer)
            return performance
        except Exception as e:
            self.log(f"模型评估失败: {e}", level="ERROR")
            # 如果评估失败，返回一个很低的性能分数
            return 0.0
    
    def _calculate_sparsity(self):
        """
        计算当前模型的稀疏度（剪枝比例）
        
        Returns:
            sparsity: 当前稀疏度
        """
        # 计算已剪枝层的比例
        total_layers = self.num_layers
        pruned_layer_ratio = len(self.pruned_layers) / total_layers if total_layers > 0 else 0
        
        # 计算已剪枝头的比例
        total_heads = sum(self.num_heads_per_layer)
        pruned_heads = sum(len(heads) for heads in self.pruned_heads.values())
        
        pruned_head_ratio = pruned_heads / total_heads if total_heads > 0 else 0
        
        # 综合计算稀疏度
        sparsity = (pruned_layer_ratio + pruned_head_ratio) / 2
        return sparsity
    
    def _calculate_reward(self, pre_performance, pre_sparsity):
        """
        计算奖励
        
        Args:
            pre_performance: 剪枝前的性能
            pre_sparsity: 剪枝前的稀疏度
            
        Returns:
            reward: 计算得到的奖励
        """
        # 性能变化（归一化）
        performance_change = (self.current_performance - pre_performance) / self.original_performance
        
        # 稀疏度变化
        sparsity_change = self.current_sparsity - pre_sparsity
        
        # 性能奖励（性能下降越少越好）
        performance_reward = self.performance_weight * performance_change
        
        # 稀疏度奖励（剪枝比例越接近目标越好）
        if self.current_sparsity <= self.target_sparsity:
            # 当前稀疏度低于目标，鼓励增加稀疏度
            sparsity_reward = self.sparsity_weight * sparsity_change
        else:
            # 当前稀疏度高于目标，惩罚过度剪枝
            sparsity_reward = -self.sparsity_weight * (self.current_sparsity - self.target_sparsity)
        
        # 总奖励
        reward = performance_reward + sparsity_reward
        
        # 对严重性能下降进行惩罚
        if self.current_performance < 0.7 * self.original_performance:
            reward -= 2.0
        
        return reward
    
    def render(self):
        """
        渲染环境状态
        """
        if self.render_mode == "human":
            print("\n" + "="*50)
            print(f"当前稀疏度: {self.current_sparsity:.4f}")
            print(f"当前性能: {self.current_performance:.4f}")
            print(f"性能比例: {(self.current_performance / self.original_performance):.4f}")
            print(f"已剪枝层数: {len(self.pruned_layers)}/{self.num_layers}")
            pruned_heads_count = sum(len(heads) for heads in self.pruned_heads.values())
            print(f"已剪枝头数: {pruned_heads_count}/{sum(self.num_heads_per_layer)}")
            print("="*50)
    
    def close(self):
        """
        关闭环境
        """
        # 清理资源
        pass
    
    def seed(self, seed=None):
        """
        设置随机种子
        
        Args:
            seed: 随机种子
        
        Returns:
            [seed]: 随机种子列表
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return [seed]

def main():
    """主函数"""
    # 解析参数
    args = get_args()
    
    # 创建输出目录
    output_dir = create_output_dir(args)
    logger.info(f"Output directory: {output_dir}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 加载模型和数据集
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = get_model(args.model)
    model = model.to(args.device)
    
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    
    # 创建环境
    logger.info("Creating pruning environment")
    env = PruningEnvironment(
        model=model,
        tokenizer=tokenizer,
        val_dataset=dataset,
        metric_fn=evaluate_model,
        device=args.device,
        sparsity_weight=args.sparsity_weight,
        performance_weight=args.performance_weight,
        target_sparsity=args.target_sparsity,
        max_pruning_ratio=args.max_pruning_ratio,
        seed=args.seed,
    )
    
    # 初始化图结构
    logger.info("Initializing graph structure")
    state = env.reset()
    layer_graph = state['layer_graph']
    head_graphs = state['head_graphs']
    
    # 确定输入特征维度
    node_feature_dim = layer_graph.x.shape[1]
    
    # 创建Agent
    logger.info(f"Creating Agent with {args.gnn_type} GNN")
    agent = PruningAgent(
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        gnn_type=args.gnn_type,
        gnn_layers=args.gnn_layers,
    )
    
    # 创建PPO训练器
    logger.info("Creating PPO trainer")
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # 开始训练
    logger.info(f"Starting training for {args.num_iterations} iterations")
    training_stats = trainer.train(
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_freq=args.eval_freq,
    )
    
    # 绘制训练统计信息
    logger.info("Plotting training statistics")
    plot_training_stats(training_stats, output_dir)
    
    # 保存训练好的模型
    logger.info("Saving models")
    torch.save(agent.state_dict(), os.path.join(output_dir, 'agent.pth'))
    
    # 最终评估
    logger.info("Final evaluation")
    final_stats = trainer.evaluate(num_episodes=10)
    
    # 记录最终结果
    with open(os.path.join(output_dir, 'final_results.yaml'), 'w') as f:
        yaml.dump(final_stats, f)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main() 