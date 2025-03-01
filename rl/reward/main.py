# rl/reward/main.py
import os
import argparse
import torch
import logging
import yaml
import time
from datetime import datetime
from typing import Dict, Any

from rl.reward.data_generator import ModelPerformanceDataGenerator
from rl.reward.trainer import GNNPerformancePredictorTrainer
from models import get_model
from data import get_dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GNN性能预测器训练')
    
    # 数据生成参数
    parser.add_argument('--model', type=str, default='Qwen2.5-3B', help='模型名称')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='数据集名称')
    parser.add_argument('--task_type', type=str, default='language_modeling', help='任务类型')
    parser.add_argument('--num_samples', type=int, default=200, help='生成样本数量')
    parser.add_argument('--batch_size', type=int, default=4, help='评估批次大小')
    
    # 训练参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GAT层数')
    parser.add_argument('--heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--train_batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--output_dir', type=str, default='outputs/reward_model', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip_data_generation', action='store_true', help='跳过数据生成步骤')
    parser.add_argument('--data_path', type=str, default=None, help='已有数据路径(如果跳过数据生成)')
    
    args = parser.parse_args()
    return args

def create_output_dir(args):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.model}_{args.dataset}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    return output_dir

def main():
    """主函数"""
    args = get_args()
    output_dir = create_output_dir(args)
    logger.info(f"输出目录: {output_dir}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    data_path = args.data_path
    
    # 生成数据集
    if not args.skip_data_generation:
        logger.info("开始生成数据集...")
        
        # 加载模型和数据集
        logger.info(f"加载模型: {args.model}")
        model, tokenizer = get_model(args.model)
        model = model.to(args.device)
        
        logger.info(f"加载数据集: {args.dataset}")
        dataset = get_dataset(args.dataset)
        
        # 创建数据生成器
        data_generator = ModelPerformanceDataGenerator(
            base_model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task_type=args.task_type,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed
        )
        
        # 生成数据集
        data_path = data_generator.generate_dataset()
        logger.info(f"数据集生成完毕，保存在: {data_path}")
    else:
        if not args.data_path:
            raise ValueError("如果跳过数据生成，必须提供已有数据路径")
        logger.info(f"跳过数据生成，使用已有数据: {args.data_path}")
    
    # 训练GNN性能预测器
    logger.info("开始训练GNN性能预测器...")
    
    # 确定节点特征维度
    # 这个需要根据模型结构设置，我们可以通过加载一个样本来获取
    with open(data_path, 'rb') as f:
        samples = torch.load(f)
        node_feature_dim = samples[0]['layer_graph'].x.shape[1]
    
    # 创建训练器
    trainer = GNNPerformancePredictorTrainer(
        data_path=data_path,
        output_dir=output_dir,
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.train_batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device,
        seed=args.seed
    )
    
    # 训练模型
    model = trainer.train()
    logger.info(f"模型训练完毕，保存在: {os.path.join(output_dir, 'best_model.pth')}")
    
    # 保存完整模型
    torch.save(model, os.path.join(output_dir, "full_model.pt"))

if __name__ == "__main__":
    main()