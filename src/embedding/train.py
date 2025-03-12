import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path

from embedding.vaemodels import create_parameter_vae, ParameterVAE
from models.model_zoo import get_model
from config.config import CONF
from utils.io import log, LogRedirectMixin
from utils.graphicalor import TransformerGraphicalor

class ParameterDataset(Dataset):
    """Transformer参数数据集
    
    从Transformer模型中提取注意力头参数作为数据集
    """
    def __init__(self, model_params, max_input_dim=10000):
        """初始化参数数据集
        
        Args:
            model_params (list): 模型参数列表，每项是一个参数张量
            max_input_dim (int): 最大输入维度
        """
        self.params = []
        
        # 预处理每个参数张量
        for param in model_params:
            if isinstance(param, torch.Tensor):
                # 展平参数
                flat_param = param.detach().flatten()
                
                # 处理参数大小
                if flat_param.shape[0] > max_input_dim:
                    # 截断过长的参数
                    self.params.append(flat_param[:max_input_dim])
                else:
                    # 保留原始参数
                    self.params.append(flat_param)
    
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        return self.params[idx]

class VAETrainer(LogRedirectMixin):
    """VAE模型训练器"""
    
    def __init__(
        self, 
        transformer_model, 
        tokenizer=None,
        latent_dim=32, 
        hidden_dim=128, 
        max_input_dim=10000,
        batch_size=16, 
        learning_rate=1e-3,
        epochs=50,
        beta=1.0,
        device=CONF.device,
        log_path=None
    ):
        """初始化VAE训练器
        
        Args:
            transformer_model: Transformer模型
            tokenizer: 分词器
            latent_dim (int): 潜在空间维度
            hidden_dim (int): 隐藏层维度
            max_input_dim (int): 最大输入维度
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            epochs (int): 训练轮数
            beta (float): KL散度权重
            device (str): 训练设备
            log_path (str): 日志路径
        """
        super().__init__(log_path)
        self.device = device
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_input_dim = max_input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta
        
        # 使用TransformerGraphicalor提取模型结构
        self.graphicalor = TransformerGraphicalor(
            model=transformer_model,
            tokenizer=tokenizer,
            log_path=log_path
        )
        
        # 创建VAE模型
        self.vae = create_parameter_vae(
            max_input_dim=max_input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        # 训练记录
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []
    
    def extract_attention_parameters(self):
        """从Transformer模型中提取注意力头参数"""
        print("提取模型注意力头参数...")
        
        # 获取模型层级结构
        layer_paths = self.graphicalor._extract_layers(verbose=False)
        
        # 提取所有注意力头参数
        attention_params = []
        total_heads = 0
        
        for layer_idx, layer_path in enumerate(self.graphicalor.layer_paths):
            # 提取当前层的所有注意力头
            attention_heads = self.graphicalor._extract_attention_heads(layer_path, layer_idx, verbose=False)
            
            if not attention_heads:
                print(f"层 {layer_path} 中没有找到注意力头", level="WARNING")
                continue
                
            # 添加每个注意力头的参数
            for head_idx, head_data in enumerate(attention_heads):
                for name, param in head_data.items():
                    if not name.endswith('_grad') and isinstance(param, torch.Tensor):
                        attention_params.append(param)
                        total_heads += 1
        
        print(f"提取完成，共找到 {total_heads} 个注意力头参数")
        return attention_params
    
    def prepare_data(self):
        """准备训练数据"""
        # 提取模型参数
        params = self.extract_attention_parameters()
        
        # 创建数据集
        dataset = ParameterDataset(params, max_input_dim=self.max_input_dim)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        return dataloader
    
    def train(self, save_dir='./embedding/checkpoints'):
        """训练VAE模型"""
        # 准备目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        dataloader = self.prepare_data()
        
        if len(dataloader) == 0:
            print("没有足够的数据进行训练", level="ERROR")
            return
            
        print(f"开始训练VAE，数据批次数: {len(dataloader)}")
        
        # 训练循环
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            self.vae.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, data in enumerate(pbar):
                # 将数据移到设备
                data = data.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(data)
                
                # 计算损失
                loss, recon_loss, kl_loss = self.vae.loss_function(recon_batch, data, mu, logvar)
                # 加权KL散度
                total_loss = recon_loss + self.beta * kl_loss
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
                
                # 更新损失统计
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss.item() / len(data),
                    'recon': recon_loss.item() / len(data),
                    'kl': kl_loss.item() / len(data)
                })
            
            # 计算平均损失
            avg_loss = epoch_loss / len(dataloader.dataset)
            avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
            avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
            
            self.train_losses.append(avg_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)
            
            # 打印训练统计
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, "
                    f"Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
            
            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                model_path = save_dir / f"vae_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'latent_dim': self.latent_dim,
                    'hidden_dim': self.hidden_dim,
                    'max_input_dim': self.max_input_dim
                }, model_path)
                print(f"模型保存至 {model_path}")
        
        # 保存最终模型
        final_model_path = save_dir / "vae_final.pt"
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses[-1],
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'max_input_dim': self.max_input_dim
        }, final_model_path)
        print(f"最终模型保存至 {final_model_path}")
        
        # 绘制损失曲线
        self.plot_training_curves(save_dir / "training_curves.png")
        
        return self.vae
    
    def plot_training_curves(self, save_path):
        """绘制训练损失曲线
        
        Args:
            save_path: 保存路径
        """
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # 总损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, 'b', label='总损失')
        plt.title('总损失')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 重构损失
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.recon_losses, 'r', label='重构损失')
        plt.title('重构损失')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # KL散度
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.kl_losses, 'g', label='KL散度')
        plt.title('KL散度')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"训练曲线保存至 {save_path}")

def load_vae_model(checkpoint_path, device='cuda'):
    """加载训练好的VAE模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        
    Returns:
        加载的VAE模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    vae = create_parameter_vae(
        max_input_dim=checkpoint['max_input_dim'],
        latent_dim=checkpoint['latent_dim'],
        hidden_dim=checkpoint['hidden_dim']
    ).to(device)
    
    # 加载权重
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()  # 设置为评估模式
    
    return vae

def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练参数VAE模型')
    parser.add_argument('--latent-dim', type=int, default=32, help='潜在空间维度')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--max-input-dim', type=int, default=10000, help='最大输入维度')
    parser.add_argument('--batch-size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--beta', type=float, default=1.0, help='KL散度权重')
    parser.add_argument('--save-dir', type=str, default='./embedding/checkpoints', help='模型保存目录')
    parser.add_argument('--log-path', type=str, default="test1", help='日志文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=CONF.device, help='训练设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建日志目录
    # log_dir = os.path.dirname(args.log_path)
    # if log_dir:
    #     os.makedirs(log_dir, exist_ok=True)
    
    # 加载Transformer模型
    print("加载Transformer模型...")
    model, tokenizer = get_model('Qwen2.5-3B', cache_dir=CONF.cache_dir, add_padding_token=True)
    print("模型加载完成")
    
    # 创建训练器
    trainer = VAETrainer(
        transformer_model=model,
        tokenizer=tokenizer,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_input_dim=args.max_input_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        beta=args.beta,
        device=args.device,
        log_path=args.log_path
    )
    
    # 训练模型
    vae = trainer.train(save_dir=args.save_dir)
    
    print("训练完成!")
    return vae

if __name__ == "__main__":
    main()
