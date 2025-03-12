import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from collections import deque
import optuna
from stable_baselines3.common.callbacks import EvalCallback

from rl.agent import PruningAgent
from rl.environment import PruningEnvironment

logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    实现基于PPO（近端策略优化）算法的训练器，用于优化神经网络剪枝策略
    """
    
    def __init__(
        self,
        agent: PruningAgent,
        env: PruningEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 5,
        batch_size: int = 16,
        device: str = 'cuda',
    ):
        """
        初始化PPO训练器
        
        Args:
            agent: 剪枝策略网络
            env: 剪枝环境
            lr: 学习率
            gamma: 折扣因子
            clip_param: PPO裁剪参数
            value_coef: 价值函数损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪的最大范数
            ppo_epochs: PPO更新时的迭代次数
            batch_size: 批量大小
            device: 计算设备
        """
        self.agent = agent
        self.env = env
        self.device = device
        
        # PPO超参数
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 优化器
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        # 将模型移至指定设备
        self.agent = self.agent.to(device)
        
        # 经验记忆
        self.memory = []
    
    def collect_trajectories(self, num_steps: int) -> List[Dict]:
        """
        收集训练数据，执行环境交互
        
        Args:
            num_steps: 要收集的环境步数
            
        Returns:
            trajectories: 收集到的轨迹数据
        """
        trajectories = []
        state = self.env.reset()
        
        for _ in range(num_steps):
            # 将状态数据移至设备
            layer_graph = state['layer_graph'].to(self.device)
            head_graphs = [graph.to(self.device) for graph in state['head_graphs']]
            
            # 获取动作
            with torch.no_grad():
                layer_actions, head_actions, layer_probs, head_probs, state_value = \
                    self.agent.act(layer_graph, head_graphs)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(layer_actions, head_actions)
            
            # 保存轨迹数据
            trajectories.append({
                'state': state,
                'layer_actions': layer_actions,
                'head_actions': head_actions,
                'layer_probs': layer_probs,
                'head_probs': head_probs,
                'value': state_value,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info
            })
            
            # 更新状态
            state = next_state
            
            if done:
                state = self.env.reset()
        
        return trajectories
    
    def compute_returns(self, trajectories: List[Dict], normalize: bool = True) -> torch.Tensor:
        """
        计算每个时间步的折扣回报
        
        Args:
            trajectories: 轨迹数据
            normalize: 是否对回报进行归一化
            
        Returns:
            returns: 计算得到的回报
        """
        # 初始化回报张量
        returns = torch.zeros(len(trajectories), device=self.device)
        
        # 计算轨迹的总回报
        for t in reversed(range(len(trajectories))):
            if t == len(trajectories) - 1:
                if trajectories[t]['done']:
                    next_value = 0
                else:
                    with torch.no_grad():
                        next_state = trajectories[t]['next_state']
                        layer_graph = next_state['layer_graph'].to(self.device)
                        head_graphs = [graph.to(self.device) for graph in next_state['head_graphs']]
                        _, _, next_value = self.agent.forward(layer_graph, head_graphs)
                returns[t] = trajectories[t]['reward'] + self.gamma * next_value
            else:
                if trajectories[t]['done']:
                    returns[t] = trajectories[t]['reward']
                else:
                    returns[t] = trajectories[t]['reward'] + self.gamma * returns[t + 1]
        
        # 归一化回报
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, trajectories: List[Dict], returns: torch.Tensor) -> Dict:
        """
        使用PPO算法更新策略网络
        
        Args:
            trajectories: 轨迹数据
            returns: 每个时间步的回报
            
        Returns:
            stats: 训练统计信息
        """
        # 训练统计信息
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'total_loss': 0,
            'clip_fraction': 0,
            'approx_kl': 0,
        }
        
        # 在多个epoch中更新策略
        for _ in range(self.ppo_epochs):
            # 随机打乱数据顺序
            indices = np.random.permutation(len(trajectories))
            
            # 批量处理
            for start_idx in range(0, len(trajectories), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_states = [trajectories[i]['state'] for i in batch_indices]
                batch_layer_actions = torch.stack([trajectories[i]['layer_actions'] for i in batch_indices])
                batch_head_actions = [trajectories[i]['head_actions'] for i in batch_indices]
                batch_layer_probs = torch.stack([trajectories[i]['layer_probs'] for i in batch_indices])
                batch_head_probs = [trajectories[i]['head_probs'] for i in batch_indices]
                batch_values = torch.stack([trajectories[i]['value'] for i in batch_indices])
                batch_returns = returns[batch_indices]
                
                # 计算优势函数估计
                advantages = batch_returns - batch_values.squeeze(-1)
                
                # 评估当前动作
                batch_layer_graphs = [state['layer_graph'].to(self.device) for state in batch_states]
                batch_head_graphs = [[graph.to(self.device) for graph in state['head_graphs']] for state in batch_states]
                
                # 计算当前策略的动作对数概率、熵和价值
                batch_log_probs = []
                batch_entropies = []
                batch_values_new = []
                
                for i in range(len(batch_indices)):
                    action_log_prob, entropy, value = self.agent.evaluate_actions(
                        batch_layer_graphs[i],
                        batch_head_graphs[i],
                        batch_layer_actions[i],
                        batch_head_actions[i]
                    )
                    batch_log_probs.append(action_log_prob.unsqueeze(0))
                    batch_entropies.append(entropy.unsqueeze(0))
                    batch_values_new.append(value)
                
                batch_log_probs = torch.cat(batch_log_probs)
                batch_entropies = torch.cat(batch_entropies)
                batch_values_new = torch.cat(batch_values_new)
                
                # 计算旧策略的动作对数概率
                batch_old_log_probs = []
                for i in range(len(batch_indices)):
                    layer_log_probs = torch.log(batch_layer_probs[i] + 1e-10) * batch_layer_actions[i] + \
                                    torch.log(1 - batch_layer_probs[i] + 1e-10) * (1 - batch_layer_actions[i])
                    
                    head_log_probs = []
                    for layer_idx, actions in batch_head_actions[i].items():
                        if layer_idx in batch_head_probs[i]:
                            probs = batch_head_probs[i][layer_idx]
                            log_probs = torch.log(probs + 1e-10) * actions + \
                                        torch.log(1 - probs + 1e-10) * (1 - actions)
                            head_log_probs.append(log_probs.sum())
                    
                    if head_log_probs:
                        old_log_prob = layer_log_probs.sum() + sum(head_log_probs)
                    else:
                        old_log_prob = layer_log_probs.sum()
                    batch_old_log_probs.append(old_log_prob.unsqueeze(0))
                
                batch_old_log_probs = torch.cat(batch_old_log_probs)
                
                # 计算比率和裁剪的比率
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                
                # 计算策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(batch_values_new.squeeze(-1), batch_returns)
                
                # 计算熵损失（用于鼓励探索）
                entropy_loss = -batch_entropies.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 更新统计信息
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy_loss'] += entropy_loss.item()
                stats['total_loss'] += loss.item()
                stats['clip_fraction'] += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                stats['approx_kl'] += ((batch_old_log_probs - batch_log_probs).mean()).item()
                
                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 计算平均值
        for key in stats.keys():
            stats[key] /= self.ppo_epochs * ((len(trajectories) + self.batch_size - 1) // self.batch_size)
        
        return stats
    
    def train(
        self, 
        num_iterations: int, 
        steps_per_iteration: int = 128,
        eval_freq: int = 10
    ) -> Dict:
        """
        训练剪枝策略
        
        Args:
            num_iterations: 训练迭代次数
            steps_per_iteration: 每次迭代收集的步数
            eval_freq: 评估频率（每隔多少次迭代评估一次）
            
        Returns:
            training_stats: 训练统计信息
        """
        # 训练统计信息
        training_stats = {
            'iterations': [],
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'sparsity': [],
            'performance': [],
            'performance_ratio': [],
        }
        
        # 奖励历史记录，用于计算平均奖励
        reward_history = deque(maxlen=100)
        
        # 训练循环
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # 收集轨迹
            trajectories = self.collect_trajectories(steps_per_iteration)
            
            # 计算回报
            returns = self.compute_returns(trajectories)
            
            # 更新策略
            update_stats = self.update(trajectories, returns)
            
            # 计算平均奖励
            avg_reward = sum(traj['reward'] for traj in trajectories) / len(trajectories)
            reward_history.extend([traj['reward'] for traj in trajectories])
            avg_reward_history = sum(reward_history) / len(reward_history)
            
            # 更新训练统计信息
            training_stats['iterations'].append(iteration)
            training_stats['rewards'].append(avg_reward)
            training_stats['policy_loss'].append(update_stats['policy_loss'])
            training_stats['value_loss'].append(update_stats['value_loss'])
            training_stats['entropy_loss'].append(update_stats['entropy_loss'])
            training_stats['total_loss'].append(update_stats['total_loss'])
            
            # 计算平均稀疏度和性能指标
            avg_sparsity = sum(traj['info']['sparsity'] for traj in trajectories) / len(trajectories)
            avg_performance = sum(traj['info']['performance'] for traj in trajectories) / len(trajectories)
            avg_performance_ratio = sum(traj['info']['performance_ratio'] for traj in trajectories) / len(trajectories)
            
            training_stats['sparsity'].append(avg_sparsity)
            training_stats['performance'].append(avg_performance)
            training_stats['performance_ratio'].append(avg_performance_ratio)
            
            # 输出训练进度
            logger.info(f"Iteration {iteration}/{num_iterations}")
            logger.info(f"Avg reward: {avg_reward:.4f}, Avg reward (history): {avg_reward_history:.4f}")
            logger.info(f"Avg sparsity: {avg_sparsity:.4f}, Avg performance ratio: {avg_performance_ratio:.4f}")
            logger.info(f"Policy loss: {update_stats['policy_loss']:.4f}, Value loss: {update_stats['value_loss']:.4f}")
            logger.info(f"Time: {time.time() - start_time:.2f}s")
            
            # 评估当前策略
            if (iteration + 1) % eval_freq == 0:
                self.evaluate()
        
        return training_stats
    
    def evaluate(self, num_episodes: int = 5) -> Dict:
        """
        评估当前策略
        
        Args:
            num_episodes: 评估的轮数
            
        Returns:
            eval_stats: 评估统计信息
        """
        # 保存Agent的当前模式
        training_mode = self.agent.training
        # 设置为评估模式
        self.agent.eval()
        
        # 评估统计信息
        eval_stats = {
            'rewards': [],
            'sparsity': [],
            'performance': [],
            'performance_ratio': [],
            'episode_length': [],
            'pruned_layers': [],
            'pruned_heads': [],
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # 将状态数据移至设备
                layer_graph = state['layer_graph'].to(self.device)
                head_graphs = [graph.to(self.device) for graph in state['head_graphs']]
                
                # 使用确定性策略选择动作
                with torch.no_grad():
                    layer_actions, head_actions, _, _, _ = self.agent.act(
                        layer_graph, head_graphs, deterministic=True
                    )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(layer_actions, head_actions)
                
                # 更新统计信息
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # 收集本轮评估的统计信息
            eval_stats['rewards'].append(episode_reward)
            eval_stats['sparsity'].append(info['sparsity'])
            eval_stats['performance'].append(info['performance'])
            eval_stats['performance_ratio'].append(info['performance_ratio'])
            eval_stats['episode_length'].append(episode_length)
            eval_stats['pruned_layers'].append(len(info['pruned_layers']))
            eval_stats['pruned_heads'].append(sum(len(heads) for heads in info['pruned_heads'].values()))
        
        # 计算平均值
        avg_reward = sum(eval_stats['rewards']) / num_episodes
        avg_sparsity = sum(eval_stats['sparsity']) / num_episodes
        avg_performance_ratio = sum(eval_stats['performance_ratio']) / num_episodes
        avg_pruned_layers = sum(eval_stats['pruned_layers']) / num_episodes
        avg_pruned_heads = sum(eval_stats['pruned_heads']) / num_episodes
        
        logger.info(f"Evaluation results ({num_episodes} episodes):")
        logger.info(f"Avg reward: {avg_reward:.4f}")
        logger.info(f"Avg sparsity: {avg_sparsity:.4f}")
        logger.info(f"Avg performance ratio: {avg_performance_ratio:.4f}")
        logger.info(f"Avg pruned layers: {avg_pruned_layers:.2f}")
        logger.info(f"Avg pruned heads: {avg_pruned_heads:.2f}")
        
        # 恢复Agent的原始模式
        self.agent.train(training_mode)
        
        return eval_stats

def objective(trial):
    # 采样超参数
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # 创建模型
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0
    )
    
    # 训练和评估
    eval_callback = EvalCallback(eval_env, n_eval_episodes=5)
    model.learn(total_timesteps=10000, callback=eval_callback)
    mean_reward = eval_callback.last_mean_reward
    
    return mean_reward

# 运行超参数优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
