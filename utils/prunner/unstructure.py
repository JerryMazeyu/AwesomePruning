import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from typing import Union, Optional, Callable
from utils.inspector import ModelInspector
from utils.io import soft_mkdir, log, generate_name

class WeightsCutter(ModelInspector):
    def __init__(self, model, tokenizer:Optional[Callable]=None, log_path:str=None):
        super().__init__(model, tokenizer, log_path)
        if log_path:
            log_path = log_path
    
    def _create_mask_memory_efficient(self, weight, threshold, comparison='ge', keep_on_cpu=True):
        """Create mask in a memory efficient way by processing weight in chunks.
        
        Args:
            weight (torch.Tensor): Weight tensor
            threshold (float): Threshold value
            comparison (str): Type of comparison: 'lt', 'le', 'gt', 'ge', 'eq'. Defaults to 'ge'.
            keep_on_cpu (bool): Whether to keep the result on CPU. Defaults to True.
        
        Returns:
            torch.Tensor: Mask tensor with same shape as weight
        """
        # Get original shape and total size
        original_shape = weight.shape
        total_size = weight.numel()
        original_device = weight.device
        
        # For small tensors, process directly to avoid overhead
        if total_size < 10000000:  # ~40MB for float32
            with torch.no_grad():
                if comparison == 'lt':
                    return (weight < threshold).float()
                elif comparison == 'le':
                    return (weight <= threshold).float()
                elif comparison == 'gt':
                    return (weight > threshold).float()
                elif comparison == 'ge':
                    return (weight >= threshold).float()
                else:  # 'eq'
                    return (weight == threshold).float()
        
        # Process large tensors in chunks
        chunk_size = min(5000000, total_size // 10)  # ~20MB chunks or divide into at least 10 parts
        
        # Create empty result tensor on CPU first (lower memory footprint)
        result = torch.zeros(total_size, dtype=torch.float32, device='cpu')
        
        # Process in chunks
        with torch.no_grad():
            for start_idx in range(0, total_size, chunk_size):
                end_idx = min(start_idx + chunk_size, total_size)
                
                # Extract chunk, flatten and process
                flat_weight = weight.view(-1)[start_idx:end_idx]
                
                # Compute mask for this chunk based on comparison type
                if comparison == 'lt':
                    chunk_result = (flat_weight < threshold).float()
                elif comparison == 'le':
                    chunk_result = (flat_weight <= threshold).float()
                elif comparison == 'gt':
                    chunk_result = (flat_weight > threshold).float()
                elif comparison == 'ge':
                    chunk_result = (flat_weight >= threshold).float()
                else:  # 'eq'
                    chunk_result = (flat_weight == threshold).float()
                    
                # Store result and free memory
                result[start_idx:end_idx] = chunk_result.cpu()
                del flat_weight, chunk_result
                torch.cuda.empty_cache()  # Explicitly free CUDA memory
        
        # Reshape result but keep it on CPU if requested
        result = result.reshape(original_shape)
        
        # Move back to original device only if not keeping on CPU and it's different from CPU
        if not keep_on_cpu and original_device.type != 'cpu':
            try:
                # Try to move back to original device
                return result.to(original_device)
            except RuntimeError:
                # If OOM occurs, warn user but still return CPU tensor
                log(f"Warning: Unable to move mask to {original_device} due to memory limits. Keeping on CPU.")
                return result
        
        return result

    def threshold_prune_para(self, module:str='all', threshold:Optional[float]=0.0, mode:str='lt', verbose:bool=True, memory_efficient:bool=False, keep_on_cpu:bool=True):
        """Generate mask by parameter values.

        Args:
            module (str, optional): Module name for pruning. Defaults to 'all'.
            threshold (Optional[float], optional): Threshhold to prune. Defaults to 0.0.
            mode (str, optional): 'gt': parameter > threshold, 'lt': parameter < threshold, 'g': parameter >= threshold, 'l': parameter <= threshold, 'eq': parameter == threshold. Defaults to 'lt'.
            verbose (bool): If print verbose infomation. Defaults to True.
            memory_efficient (bool): Whether to use memory efficient mask creation. Defaults to False.
            keep_on_cpu (bool): For memory_efficient=True, whether to keep masks on CPU. Defaults to True.
        """
        assert mode in ['gt', 'lt', 'g', 'l', 'eq'], ValueError(f"Make sure that mode is in ['gt', 'lt', 'g', 'l', 'eq'], now mode is {mode}.")
        paras = self.get_para(module, type_='list', verbose=False)
        
        masks = []
        for x in paras:
            if mode == 'lt':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='lt', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x <= threshold).float())
            elif mode == 'gt':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='gt', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x >= threshold).float())
            elif mode == 'g':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='g', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x > threshold).float())
            elif mode == 'l':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='l', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x < threshold).float())
            else:  # 'eq'
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='eq', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x == threshold).float())
            masks.append(mask)
        
        if verbose:
            for ind, x in enumerate(masks):
                # For safe statistics calculation (handles both CPU and GPU tensors)
                numel = x.numel()
                sum_val = float(x.sum().cpu().item())
                remain = int(sum_val)
                pruned = int(numel - sum_val)
                pruned_percent = np.round((pruned / numel) * 100, 4)
                
                log(f"Layer {ind}: Total parameters: {numel}; Remain parameters: {remain}; Pruned parameters: {pruned}; pruned percent is {pruned_percent}%.")
            
            # Calculate total statistics without creating a huge tensor
            total_params = sum(x.numel() for x in masks)
            total_remain = sum(float(x.sum().cpu().item()) for x in masks)
            total_pruned = total_params - total_remain
            pruned_percent = np.round((total_pruned / total_params) * 100, 4)
            
            log(f"Total: Total parameters: {total_params}; Remain parameters: {int(total_remain)}; Pruned parameters: {int(total_pruned)}; pruned percent is {pruned_percent}%.")
        return masks

    def threshold_prune_grad(self, module:str='all', threshold:Optional[float]=0.0, mode:str='lt', verbose:bool=True, memory_efficient:bool=False, keep_on_cpu:bool=True):
        """Generate mask by gradient.

        Args:
            module (str, optional): Module name for pruning. Defaults to 'all'.
            threshold (Optional[float], optional): Threshhold to prune. Defaults to 0.0.
            mode (str, optional): 'gt': gradient > threshold, 'lt': gradient < threshold, 'g': gradient >= threshold, 'l': gradient <= threshold, 'eq': gradient == threshold. Defaults to 'lt'.
            verbose (bool): If print verbose infomation. Defaults to True.
            memory_efficient (bool): Whether to use memory efficient mask creation. Defaults to False.
            keep_on_cpu (bool): For memory_efficient=True, whether to keep masks on CPU. Defaults to True.
        """
        # Check model status, only perform gradient pruning when status is 'trained'
        if not hasattr(self, 'status') or self.status != 'trained':
            raise ValueError("Model status must be 'trained' to perform gradient pruning. Please train the model or compute gradients first.")
            
        assert mode in ['gt', 'lt', 'g', 'l', 'eq'], ValueError(f"Make sure that mode is in ['gt', 'lt', 'g', 'l', 'eq'], now mode is {mode}.")
        grads = self.get_grad(module, type_='list', verbose=False)
        
        masks = []
        for x in grads:
            if mode == 'lt':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='lt', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x <= threshold).float())
            elif mode == 'gt':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='gt', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x >= threshold).float())
            elif mode == 'g':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='g', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x > threshold).float())
            elif mode == 'l':
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='l', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x < threshold).float())
            else:  # 'eq'
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='eq', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x == threshold).float())
            masks.append(mask)
        
        if verbose:
            for ind, x in enumerate(masks):
                # For safe statistics calculation (handles both CPU and GPU tensors)
                numel = x.numel()
                sum_val = float(x.sum().cpu().item())
                remain = int(sum_val)
                pruned = int(numel - sum_val)
                pruned_percent = np.round((pruned / numel) * 100, 4)
                
                log(f"Layer {ind}: Total parameters: {numel}; Remain parameters: {remain}; Pruned parameters: {pruned}; pruned percent is {pruned_percent}%.")
            
            # Calculate total statistics without creating a huge tensor
            total_params = sum(x.numel() for x in masks)
            total_remain = sum(float(x.sum().cpu().item()) for x in masks)
            total_pruned = total_params - total_remain
            pruned_percent = np.round((total_pruned / total_params) * 100, 4)
            
            log(f"Total: Total parameters: {total_params}; Remain parameters: {int(total_remain)}; Pruned parameters: {int(total_pruned)}; pruned percent is {pruned_percent}%.")
        return masks

    def _find_threshold_memory_efficient(self, flat_weights, k, num_samples=10000, verbose=False):
        """Find threshold using memory efficient method.
        
        Args:
            flat_weights (torch.Tensor): Flattened weight tensor
            k (int): Number of elements to keep
            num_samples (int): Number of samples for estimation
            verbose (bool): Whether to print detailed information
        
        Returns:
            float: Estimated threshold
        """
        if verbose:
            log(f"Using memory efficient threshold calculation method, samples: {num_samples}")
        
        total_size = flat_weights.numel()
        # If data size is small, calculate directly
        if total_size <= num_samples:
            if verbose:
                log(f"Data size is small ({total_size} <= {num_samples}), calculating directly")
            return self._find_threshold_direct(flat_weights, k)
        
        # Calculate target keep ratio (rather than prune ratio)
        keep_ratio = 1.0 - (k / total_size)
        target_keep = total_size - k
        if verbose:
            log(f"Finding threshold, target keep ratio: {keep_ratio:.4f}, target elements to keep: {target_keep}")
        
        # Use binary search to find appropriate threshold
        # First estimate min and max values of the data
        # To avoid OOM, use chunked processing
        chunk_size = min(100000, total_size // 10)  # Ensure at least 10 chunks
        
        # Initialize min and max values
        data_min = float('inf')
        data_max = float('-inf')
        
        # Find min and max values using chunked processing
        with torch.no_grad():
            for i in range(0, total_size, chunk_size):
                end = min(i + chunk_size, total_size)
                chunk = flat_weights[i:end]
                chunk_min = chunk.min().item()
                chunk_max = chunk.max().item()
                data_min = min(data_min, chunk_min)
                data_max = max(data_max, chunk_max)
        
        if verbose:
            log(f"Data range: [{data_min:.6f}, {data_max:.6f}]")
        
        # Binary search for appropriate threshold
        low = data_min
        high = data_max
        best_threshold = None
        best_error = float('inf')
        
        max_iterations = 20  # Maximum 20 iterations
        for iteration in range(max_iterations):
            threshold = (low + high) / 2
            
            # Count elements above threshold using chunked processing
            count = 0
            with torch.no_grad():
                for i in range(0, total_size, chunk_size):
                    end = min(i + chunk_size, total_size)
                    chunk = flat_weights[i:end]
                    count += (chunk >= threshold).sum().item()
            
            # Calculate current error
            error = abs(count - target_keep)
            
            # Update best threshold
            if error < best_error:
                best_error = error
                best_threshold = threshold
            
            # Early stop if error is small enough
            if error < max(100, total_size * 0.001):  # Error less than 0.1% or 100 elements
                if verbose:
                    log(f"Iteration {iteration+1}/{max_iterations}: Threshold = {threshold:.6f}, "
                          f"Elements kept = {count}/{total_size}, Target = {target_keep}, "
                          f"Error = {error} (small enough, early stopping)")
                break
            
            if verbose:
                log(f"Iteration {iteration+1}/{max_iterations}: Threshold = {threshold:.6f}, "
                      f"Elements kept = {count}/{total_size}, Target = {target_keep}, "
                      f"Error = {error}")
            
            # Adjust search range
            if count > target_keep:
                low = threshold  # Current threshold too low, keeping too many elements
            else:
                high = threshold  # Current threshold too high, keeping too few elements
        
        if verbose:
            log(f"Final threshold: {best_threshold:.6f}, Minimum error: {best_error}")
        
        return best_threshold

    def _find_threshold_direct(self, flat_weights, k):
        """Find threshold by torch.topk (original method).
        
        Args:
            flat_weights (torch.Tensor): Flattened weight tensor
            k (int): Number of elements to keep
        
        Returns:
            float: Calculated threshold
        """
        # Use torch.topk to find the k-th largest element as threshold
        topk, _ = torch.topk(flat_weights, k)
        threshold = topk[-1].item()
        return threshold

    def prune_para_by_weights(self, module:str='all', weights:Optional[list]=None, prune_rate:float=0.1, verbose=True, memory_efficient:bool=True, keep_on_cpu:bool=True):
        """Generate mask by weights.

        Args:
            module (str, optional): Module name for pruning. Defaults to 'all'.
            weights (Optional, optional): Importance weights. Defaults to None.
            prune_rate (float, optional): Prune rate. Defaults to 0.1.
            verbose (bool): If print verbose infomation. Defaults to True.
            memory_efficient (bool): Whether to use memory efficient threshold calculation method. Defaults to True.
            keep_on_cpu (bool): For memory_efficient=True, whether to keep masks on CPU. Defaults to True.
        """  
        paras = self.get_para(module, type_='list', verbose=False)
        if not weights:
            if verbose:
                log("No weights given, random initialize it for demonstration.")
            weights = []
            for i in range(len(paras)):
                weights.append(torch.rand_like(paras[i]))
        else:
            for i in range(len(paras)):
                assert paras[i].shape == weights[i].shape, ValueError(f"Wrong weights shape, {i}-th layer parameter shape is {paras[i].shape}, but now it is {weights[i].shape}")
        flat_weights = torch.cat([torch.flatten(x) for x in weights])
        k = int(len(flat_weights) * prune_rate)
        if k == 0:
            masks = [torch.ones_like(x, device='cpu' if keep_on_cpu else x.device) for x in paras]
        else:
            if memory_efficient:
                threshold = self._find_threshold_memory_efficient(flat_weights, k, verbose=verbose)
            else:
                threshold = self._find_threshold_direct(flat_weights, k)
            
            # Create masks with memory-efficient method if requested
            masks = []
            for i, x in enumerate(weights):
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='ge', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x >= threshold).float())
                masks.append(mask)
            
        if verbose:
            for ind, x in enumerate(masks):
                # For safe statistics calculation (handles both CPU and GPU tensors)
                numel = x.numel()
                sum_val = float(x.sum().cpu().item())
                remain = int(sum_val)
                pruned = int(numel - sum_val)
                pruned_percent = np.round((pruned / numel) * 100, 4)
                
                log(f"Layer {ind}: Total parameters: {numel}; Remain parameters: {remain}; Pruned parameters: {pruned}; pruned percent is {pruned_percent}%.")
            
            # Calculate total statistics without creating a huge tensor
            total_params = sum(x.numel() for x in masks)
            total_remain = sum(float(x.sum().cpu().item()) for x in masks)
            total_pruned = total_params - total_remain
            pruned_percent = np.round((total_pruned / total_params) * 100, 4)
            
            log(f"Total: Total parameters: {total_params}; Remain parameters: {int(total_remain)}; Pruned parameters: {int(total_pruned)}; pruned percent is {pruned_percent}%.")
        return masks
    
    def prune_grad_by_weights(self, module:str='all', weights:Optional[list]=None, prune_rate:float=0.1, verbose=True, memory_efficient:bool=True, keep_on_cpu:bool=True):
        """Generate mask by gradient weights.

        Args:
            module (str, optional): Module name for pruning. Defaults to 'all'.
            weights (Optional, optional): Importance weights. Defaults to None.
            prune_rate (float, optional): Prune rate. Defaults to 0.1.
            verbose (bool): If print verbose infomation. Defaults to True.
            memory_efficient (bool): Whether to use memory efficient threshold calculation method. Defaults to True.
            keep_on_cpu (bool): For memory_efficient=True, whether to keep masks on CPU. Defaults to True.
        """  
        # Check model status, only perform gradient pruning when status is 'trained'
        if not hasattr(self, 'status') or self.status != 'trained':
            raise ValueError("Model status must be 'trained' to perform gradient pruning. Please train the model or compute gradients first.")
            
        grads = self.get_grad(module, type_='list', verbose=False)
        if not weights:
            if verbose:
                log("No weights given, random initialize it for demonstration.")
            weights = []
            for i in range(len(grads)):
                weights.append(torch.rand_like(grads[i]))
        else:
            for i in range(len(grads)):
                assert grads[i].shape == weights[i].shape, ValueError(f"Wrong weights shape, {i}-th layer gradient shape is {grads[i].shape}, but now it is {weights[i].shape}")
        
        flat_weights = torch.cat([torch.flatten(x) for x in weights])
        k = int(len(flat_weights) * prune_rate)
        
        if k == 0:
            masks = [torch.ones_like(x, device='cpu' if keep_on_cpu else x.device) for x in grads]
        else:
            if memory_efficient:
                threshold = self._find_threshold_memory_efficient(flat_weights, k, verbose=verbose)
            else:
                threshold = self._find_threshold_direct(flat_weights, k)
            
            # Create masks with memory-efficient method if requested
            masks = []
            for i, x in enumerate(weights):
                if memory_efficient:
                    mask = self._create_mask_memory_efficient(x, threshold, comparison='ge', keep_on_cpu=keep_on_cpu)
                else:
                    mask = torch.tensor((x >= threshold).float())
                masks.append(mask)
        
        if verbose:
            for ind, x in enumerate(masks):
                # For safe statistics calculation (handles both CPU and GPU tensors)
                numel = x.numel()
                sum_val = float(x.sum().cpu().item())
                remain = int(sum_val)
                pruned = int(numel - sum_val)
                pruned_percent = np.round((pruned / numel) * 100, 4)
                
                log(f"Layer {ind}: Total gradients: {numel}; Remain gradients: {remain}; Pruned gradients: {pruned}; pruned percent is {pruned_percent}%.")
            
            # Calculate total statistics without creating a huge tensor
            total_params = sum(x.numel() for x in masks)
            total_remain = sum(float(x.sum().cpu().item()) for x in masks)
            total_pruned = total_params - total_remain
            pruned_percent = np.round((total_pruned / total_params) * 100, 4)
            
            log(f"Total: Total gradients: {total_params}; Remain gradients: {int(total_remain)}; Pruned gradients: {int(total_pruned)}; pruned percent is {pruned_percent}%.")
        
        return masks
    
    def show_mask(self, weights:torch.Tensor, mask: torch.Tensor) -> None:
        """Visualizes a heatmap where unmasked (mask=1) values are shown in red tones, 
           and masked (mask=0) values are shown in a uniform blue color.

        Args:
            weights (torch.Tensor): Matrix / Vector / (1, X, Y) like tensor, the real weights.
            mask (torch.Tensor): Tensor of the same shape as weights, the mask(0 or 1).
        """
        assert weights.shape == mask.shape, f"The shape of mask({mask.shape}) and weights({weights.shape}) must match."
        weights_np = weights.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_np, mask=mask_np == 0, cmap='Blues', annot=False, cbar=True, linewidths=0.5)

        # Overlay solid red color for mask=0
        weights_np[mask_np == 0] = np.nan  # Set masked values to NaN to overlay red color

        plt.title("Heatmap with Masked Regions")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.savefig(os.path.join(log_path, "heatmap_with_masked_regions.png"))
        plt.close()

if __name__ == '__main__':
    import argparse
    import torch
    import os
    import sys
    
    # 添加项目根目录到路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import CONF
    from models import get_model
    from data import get_dataset
    
    # 设置参数解析
    parser = argparse.ArgumentParser(description='测试WeightsCutter类')
    parser.add_argument('--model', type=str, default='Qwen2.5-3B', help='模型名称')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='数据集名称')
    parser.add_argument('--task_type', type=str, default='language_modeling', help='任务类型(为None则自动推断)')
    parser.add_argument('--log_path', type=str, default=None, help='日志文件路径')
    parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('--n_batches', type=int, default=2, help='用于校准的批次数')
    args = parser.parse_args()
    
    print(f"正在测试WeightsCutter类，使用模型: {args.model}, 数据集: {args.dataset}")
    
    # 加载模型和分词器
    print(f"正在加载模型和分词器...")
    model, tokenizer = get_model(args.model, cache_dir=CONF.cache_dir)
    
    # 加载数据集
    print(f"正在加载数据集: {args.dataset}...")
    dataset = get_dataset(args.dataset)
    
    # 初始化WeightsCutter
    cutter = WeightsCutter(model, tokenizer, log_path=args.log_path)
    print(f"WeightsCutter初始化完成")
    
    # 打印模型信息
    print("模型信息:")
    cutter._print_model_info()
    
    # 测试状态检查 - 这会触发错误，因为模型尚未训练
    print("\n1. 测试状态检查 (预期会失败):")
    try:
        masks = cutter.threshold_prune_grad(threshold=0.1)
        print("  错误: 状态检查未能阻止在未训练状态下进行梯度剪枝")
    except ValueError as e:
        print(f"  通过: {e}")
    
    # 校准模型 (这会执行前向和反向传播)
    print("\n2. 校准模型:")
    try:
        calibration_result = cutter.calibrate(
            calibration_dataset=dataset,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            task_type=args.task_type
        )
        print(f"  校准完成，结果: {calibration_result}")
    except Exception as e:
        print(f"  校准失败: {e}")
    
    print("  模型状态已设置为:", cutter.status)
    
    # 测试threshold_prune_para
    print("\n3. 测试基于阈值的参数剪枝:")
    masks = cutter.threshold_prune_para(threshold=0.1, mode='lt')
    print(f"  成功获取参数剪枝掩码: {len(masks)}个层")
    
    # 测试threshold_prune_grad
    print("\n4. 测试基于阈值的梯度剪枝:")
    try:
        masks = cutter.threshold_prune_grad(threshold=0.1, mode='lt')
        print(f"  成功获取梯度剪枝掩码: {len(masks)}个层")
    except Exception as e:
        print(f"  梯度剪枝失败: {e}")
    
    # 测试prune_para_by_weights
    print("\n5. 测试基于权重的参数剪枝:")
    # 使用随机权重
    masks = cutter.prune_para_by_weights(prune_rate=0.2, memory_efficient=True)
    print(f"  成功获取基于权重的参数剪枝掩码: {len(masks)}个层")
    
    # 测试prune_grad_by_weights
    print("\n6. 测试基于权重的梯度剪枝:")
    try:
        # 使用随机权重
        masks = cutter.prune_grad_by_weights(prune_rate=0.2)
        print(f"  成功获取基于权重的梯度剪枝掩码: {len(masks)}个层")
    except Exception as e:
        print(f"  基于权重的梯度剪枝失败: {e}")
    
    # 测试特定模块的剪枝
    print("\n7. 测试对特定模块的剪枝:")
    # 获取模型的一个层的名称
    layer_names = [name for name, _ in model.named_modules() if 'transformer' in name and 'layer' in name]
    if layer_names:
        test_layer = layer_names[0]
        print(f"  选择的测试层: {test_layer}")
        masks = cutter.threshold_prune_para(module=test_layer, threshold=0.1)
        print(f"  成功对特定层进行剪枝")
    else:
        print("  没有找到合适的测试层")
    
    # 测试show_mask功能
    if hasattr(cutter, 'show_mask'):
        print("\n8. 测试掩码可视化:")
        # 获取一个参数和对应的掩码
        params = cutter.get_para(type_='list', verbose=False)
        if params:
            # 创建一个简单的掩码用于测试
            mask = (params[0] > params[0].mean()).float()
            try:
                cutter.show_mask(params[0], mask)
                print("  掩码可视化成功")
            except Exception as e:
                print(f"  掩码可视化失败: {e}")
        else:
            print("  没有可用的参数进行掩码可视化测试")
    
    # 测试零剪枝率情况
    print("\n9. 测试零剪枝率边界情况:")
    masks = cutter.prune_para_by_weights(prune_rate=0.0)
    # 检查所有掩码是否都是全1
    all_ones = all(torch.all(mask == 1).item() for mask in masks)
    print(f"  零剪枝率测试: {'通过' if all_ones else '失败'}")


