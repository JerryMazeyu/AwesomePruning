import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterVAE(nn.Module):
    """参数变分自编码器，用于将模型参数压缩到低维潜在空间
    
    这个VAE模型接收参数张量，将其编码到低维潜在空间，并可以解码回原始空间。
    主要用于提取参数的紧凑表示，用于图神经网络中的节点特征。
    """
    
    def __init__(self, max_input_dim=10000, latent_dim=32, hidden_dim=128):
        """初始化ParameterVAE模型
        
        Args:
            max_input_dim (int): 最大输入维度，对于超过此维度的输入将进行裁剪
            latent_dim (int): 潜在空间维度，即编码后的向量长度
            hidden_dim (int): 隐藏层维度
        """
        super(ParameterVAE, self).__init__()
        
        self.max_input_dim = max_input_dim
        self.latent_dim = latent_dim
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(max_input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # 均值和对数方差预测
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, max_input_dim)
        )
    
    def encode(self, x):
        """编码输入到潜在空间
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            tuple: (均值, 对数方差)
        """
        # 处理输入维度
        x = self._preprocess_input(x)
        
        # 通过编码器
        h = self.encoder(x)
        
        # 获取均值和对数方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def _preprocess_input(self, x):
        """预处理输入张量到固定维度
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        batch_size = x.size(0) if x.dim() > 1 else 1
        
        # 确保输入是二维的 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 处理输入维度
        if x.size(1) > self.max_input_dim:
            # 截断过长的输入
            x = x[:, :self.max_input_dim]
        elif x.size(1) < self.max_input_dim:
            # 填充过短的输入
            padding = torch.zeros(batch_size, self.max_input_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
            
        return x
    
    def reparameterize(self, mu, logvar):
        """使用重参数化技巧进行采样
        
        Args:
            mu (torch.Tensor): 均值
            logvar (torch.Tensor): 对数方差
            
        Returns:
            torch.Tensor: 采样结果
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """从潜在空间解码到原始空间
        
        Args:
            z (torch.Tensor): 潜在向量
            
        Returns:
            torch.Tensor: 重构的输入
        """
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            tuple: (重构输入, 均值, 对数方差)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # 在推理时，只返回均值作为嵌入表示
        if not self.training:
            return mu
            
        return recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """计算VAE损失函数
        
        Args:
            recon_x (torch.Tensor): 重构输入
            x (torch.Tensor): 原始输入
            mu (torch.Tensor): 均值
            logvar (torch.Tensor): 对数方差
            
        Returns:
            torch.Tensor: 总损失
        """
        # 预处理原始输入以匹配重构输入
        x = self._preprocess_input(x)
        
        # 重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss

# 工具函数，创建并返回初始化后的VAE模型
def create_parameter_vae(max_input_dim=10000, latent_dim=32, hidden_dim=128):
    """创建参数VAE模型
    
    Args:
        max_input_dim (int): 最大输入维度
        latent_dim (int): 潜在空间维度
        hidden_dim (int): 隐藏层维度
        
    Returns:
        ParameterVAE: 初始化后的VAE模型
    """
    model = ParameterVAE(max_input_dim=max_input_dim, 
                         latent_dim=latent_dim, 
                         hidden_dim=hidden_dim)
    return model
