import os
import torch
from torch import nn
from typing import Union, Generator, Optional
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from torchsummary import summary
from utils.io import LogRedirectMixin, log, generate_name
from config import CONF
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Callable
import math

def format_params(n):
    """
    Convert parameter count n to a more readable format, e.g.:
      1234 -> "1.23K"
      1234567 -> "1.23M"
      1234567890 -> "1.23B"
    """
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)

def llm_summary(model, seq_len, batch_size=1, device="cuda"):
    """
    Summarize the LLM model's layers and parameters.

    Args:
        model: Model instance to be summarized, should be a LLM model based on Transformers.
        seq_len: Length of input sequence (e.g., 128, 256).
        batch_size: Batch size (default to 1).
        device: Device used ("cuda" or "cpu").
    
    Returns:
        None, print the summary of each layer's output shape, number of parameters, etc.
    """
    device = device.lower()

    # Get vocab_size from model config, default to 1000 if not found
    vocab_size = getattr(model.config, "vocab_size", 1000)
    
    # Construct dummy input: token_ids are random integers, attention_mask is all 1s
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
    dummy_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # OrderedDict to save information for each layer
    summary_dict = OrderedDict()
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            if not isinstance(input, tuple) or len(input) == 0:
                return
            try:
                first_input = input[0]
            except IndexError:
                return
            # If input is passed as a dictionary, take the "input_ids" tensor
            if isinstance(first_input, dict):
                inp_tensor = first_input.get("input_ids", None)
                if inp_tensor is None:
                    return
            else:
                inp_tensor = first_input

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_dict)
            m_key = f"{class_name}-{module_idx + 1}"
            summary_dict[m_key] = OrderedDict()

            # Record input shape
            summary_dict[m_key]["input_shape"] = list(inp_tensor.size())
            summary_dict[m_key]["input_shape"][0] = batch_size  # Set batch_size

            # Record output shape
            if isinstance(output, (list, tuple)):
                # If output is a tuple, take the first tensor
                out_tensor = output[0]
            else:
                out_tensor = output
            summary_dict[m_key]["output_shape"] = list(out_tensor.size())
            summary_dict[m_key]["output_shape"][0] = batch_size

            # Calculate parameter count
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.tensor(list(module.weight.size()), dtype=torch.long)).item()
                summary_dict[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.tensor(list(module.bias.size()), dtype=torch.long)).item()
            summary_dict[m_key]["nb_params"] = params

        # Register hook for non-container modules and the model itself
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and (module != model):
            hooks.append(module.register_forward_hook(hook))

    # Register hooks on the model
    model.apply(register_hook)

    # Perform one forward pass
    model.eval()
    with torch.no_grad():
        model(**dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print summary information
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary_dict:
        nb_params = summary_dict[layer]["nb_params"]
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary_dict[layer]["output_shape"]),
            # "{0:,}".format(summary_dict[layer]["nb_params"]),
            format_params(nb_params)
        )
        total_params += summary_dict[layer]["nb_params"]
        total_output += np.prod(summary_dict[layer]["output_shape"])
        if summary_dict[layer].get("trainable", False):
            trainable_params += summary_dict[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print("Total params: {} ({})".format(total_params, format_params(total_params)))
    print("Trainable params: {} ({})".format(trainable_params, format_params(trainable_params)))
    print("Non-trainable params: {} ({})".format(total_params - trainable_params, format_params(total_params - trainable_params)))
    print("----------------------------------------------------------------")

    # Estimate memory usage (in MB)
    total_input_size = abs(np.prod([batch_size, seq_len]) * 4.0 / (1024 ** 2))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2))
    total_size = total_input_size + total_output_size + total_params_size

    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")



class ModelInspector(LogRedirectMixin):
    def __init__(self, model:nn.Module, tokenizer:Optional[Callable]=None, log_path:Optional[str]=None) -> None:
        super().__init__(log_path)
        self.model = model
        self.tokenizer = tokenizer
        if self.model._model_type == 'LM':
            assert tokenizer is not None, ValueError("Tokenizer is required for language modeling tasks.")
        self.status = 'blank'  # blank, trained
        if log_path:
            self.log_path = log_path

    def summary(self, mock_inp:tuple=(3,224,224)):
        """Show model summary.

        Args:
            mock_inp (tuple, optional): Mock input tensor shape. Defaults to (3,224,224).
        """
        try:
            if self.model._model_type == 'CNN':
                # 尝试将整个模型移动到CPU以避免设备不一致问题
                try:
                    # 保存原始设备以便之后恢复
                    original_device = next(self.model.parameters()).device
                    self.model = self.model.cpu()
                    summary(self.model, mock_inp, device='cpu')
                    # 操作完成后将模型移回原设备
                    self.model = self.model.to(original_device)
                except Exception as e:
                    print(f"Error during model summary on CPU: {e}")
                    # 如果CPU摘要失败，则打印基本信息
                    self._print_model_info()
            elif self.model._model_type == 'LM':
                llm_summary(self.model, seq_len=128, batch_size=2, device=CONF.device)
        except Exception as e:
            print(f"Warning: Could not generate summary: {e}")
            self._print_model_info()

    def _print_model_info(self):
        """Print basic model information when detailed summary fails."""
        print("\nModel basic information:")
        print(f"Model type: {self.model._model_type}")
        print(f"Model class: {self.model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {format_params(total_params)}")
        print(f"Trainable parameters: {format_params(trainable_params)}")
        
        # Print top-level modules
        print("\nTop-level modules:")
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module.__class__.__name__} - {format_params(params)} parameters")

    def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size: int = 2, n_batches: int = 10, task_type: str = None):
        """Perform one forward and backward pass on calibration dataset to obtain model gradients.

        Args:
            calibration_dataset (torch.utils.data.Dataset): Dataset for calibration
            batch_size (int, optional): Batch size. Defaults to 32.
            task_type (str, optional): Task type. If None, will be inferred from model type. 
                Available values: 'classification', 'language_modeling', 'sequence_classification', etc.
        """
        if task_type is None:
            task_type = self._infer_task_type()
        
        dataloader = torch.utils.data.DataLoader(
            calibration_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        self.model.train()  # Ensure model is in training mode
        self.model.zero_grad()

        # Get one batch of data
        batch = next(iter(dataloader))
        
        # Process data and compute loss based on task type
        if task_type == 'classification':  # TODO: NOT TESTED
            raise NotImplementedError("Classification task is not implemented yet.")
            # inputs, labels = batch
            # inputs = inputs.to(CONF.device)
            # labels = labels.to(CONF.device)
            # outputs = self.model(inputs)
            # loss = nn.CrossEntropyLoss()(outputs, labels)
        
        elif task_type == 'sequence_classification':  # TODO: N_BATCHES
            raise NotImplementedError("Sequence classification task is not implemented yet.")
            # inputs, labels = batch['input_ids'], batch['labels']
            # if isinstance(inputs, torch.Tensor):
            #     inputs = inputs.to(CONF.device)
            # elif isinstance(inputs, list) and isinstance(inputs[0], str):
            #     inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
            #     inputs = inputs.to(CONF.device)
            # else:
            #     raise ValueError(f"Unsupported input format: {type(inputs)}")
            # if isinstance(labels, torch.Tensor):
            #     labels = labels.to(CONF.device)
            # outputs = self.model(**inputs)
            # last_logits = outputs.logits[:, -1, :]
            # loss = nn.CrossEntropyLoss()(last_logits, labels)
        
        elif task_type == 'language_modeling':
            # 语言建模任务评估
            total_loss = 0.0
            total_tokens = 0
            
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                    
                try:
                    # 处理不同格式的输入
                    if isinstance(batch, dict):
                        # 检查input_ids是否为字符串列表
                        if 'input_ids' in batch and isinstance(batch['input_ids'], list) and isinstance(batch['input_ids'][0], str):
                            # 需要先使用tokenizer处理文本
                            if self.tokenizer is None:
                                log(f"输入是文本但未提供tokenizer")
                                continue
                                
                            # 将文本转换为token ids
                            encoded = self.tokenizer(
                                batch['input_ids'],  # 这里是文本列表
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=512  # 可根据需要调整
                            )
                            batch_on_device = {k: v.to(CONF.device) for k, v in encoded.items()}
                        else:
                            # 已经是tokenized的张量数据，只需移至正确设备
                            batch_on_device = {k: v.to(CONF.device) if isinstance(v, torch.Tensor) else v 
                                              for k, v in batch.items()}
                        
                        # 对于没有labels的数据集（常见情况），创建移位labels用于计算下一个token的预测
                        if 'input_ids' in batch_on_device:
                            # 将input_ids右移一位作为标签
                            input_ids = batch_on_device['input_ids']
                            
                            # 如果有attention_mask，使用它来计算有效token数
                            if 'attention_mask' in batch_on_device:
                                attention_mask = batch_on_device['attention_mask']
                                # 有效token计数（忽略padding）
                                batch_tokens = attention_mask.sum().item()
                            else:
                                # 如果没有mask，假设所有token都有效
                                batch_tokens = input_ids.numel()
                                
                            # 前向传播
                            outputs = self.model(**batch_on_device)
                            
                            # 如果模型已经计算了loss
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                batch_loss = outputs.loss * batch_tokens  # 转换为总损失
                            else:
                                # 否则手动计算loss
                                # 输入序列移位：预测目标是下一个token
                                shift_logits = outputs.logits[..., :-1, :].contiguous()
                                shift_labels = input_ids[..., 1:].contiguous()
                                
                                # 使用交叉熵损失
                                loss_fct = nn.CrossEntropyLoss(reduction='sum')  # 使用sum便于计算总token的损失
                                batch_loss = loss_fct(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1)
                                )
                                
                                # 计算有效token数量（排除padding）
                                if 'attention_mask' in batch_on_device:
                                    # 右移attention_mask来匹配shift_labels
                                    shift_mask = attention_mask[..., 1:].contiguous()
                                    batch_tokens = shift_mask.sum().item()
                                else:
                                    batch_tokens = shift_labels.numel()

                            # 反向传播
                            batch_loss.backward()

                            # 累加总损失和token数
                            total_loss += batch_loss.item()
                            total_tokens += batch_tokens
                            
                    elif isinstance(batch, list) and all(isinstance(item, str) for item in batch):
                        # 直接是字符串列表的情况
                        if self.tokenizer is None:
                            log(f"输入是文本但未提供tokenizer")
                            continue
                            
                        # 使用tokenizer处理文本
                        encoded = self.tokenizer(
                            batch,  # 文本列表
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512  # 可根据需要调整
                        )
                        batch_on_device = {k: v.to(CONF.device) for k, v in encoded.items()}
                        
                        # 获取input_ids用于后续处理
                        input_ids = batch_on_device['input_ids']
                        
                        # 如果有attention_mask，使用它来计算有效token数
                        if 'attention_mask' in batch_on_device:
                            attention_mask = batch_on_device['attention_mask']
                            batch_tokens = attention_mask.sum().item()
                        else:
                            batch_tokens = input_ids.numel()
                        
                        # 前向传播
                        outputs = self.model(**batch_on_device)
                        
                        # 处理损失计算与之前相同
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            batch_loss = outputs.loss * batch_tokens  # 转换为总损失
                        else:
                            shift_logits = outputs.logits[..., :-1, :].contiguous()
                            shift_labels = input_ids[..., 1:].contiguous()
                            
                            loss_fct = nn.CrossEntropyLoss(reduction='sum')
                            batch_loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            
                            if 'attention_mask' in batch_on_device:
                                shift_mask = attention_mask[..., 1:].contiguous()
                                batch_tokens = shift_mask.sum().item()
                            else:
                                batch_tokens = shift_labels.numel()

                        # 反向传播
                        batch_loss.backward()

                        # 累加总损失和token数
                        total_loss += batch_loss.item()
                        total_tokens += batch_tokens

                    else:
                        log(f"不支持的输入格式: {type(batch)}")
                        continue
                        
                except Exception as e:
                    log(f"批次处理错误: {str(e)}")
                    import traceback
                    log(traceback.format_exc())
                
            # 计算平均损失和困惑度
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                
                # 将困惑度归一化到0-1之间作为分数
                # 使用一个合理的阈值(比如20)作为高困惑度
                max_reasonable_ppl = 20.0
                score = max(0.0, 1.0 - min(perplexity / max_reasonable_ppl, 1.0))
                
                log(f"语言建模评估 - 总tokens: {total_tokens}")
                log(f"语言建模评估 - 平均损失: {avg_loss:.4f}")
                log(f"语言建模评估 - 困惑度(PPL): {perplexity:.4f}")
                log(f"语言建模评估 - 归一化分数: {score:.4f}")
                
                results = {
                    'loss': avg_loss,
                    'perplexity': perplexity,
                    'score': score,
                    'total_tokens': total_tokens
                }
                # Update status
                self.status = 'trained'
                
            else:
                log(f"没有处理任何有效tokens")
                results = {'loss': float('inf'), 'perplexity': float('inf'), 'score': 0.0, 'total_tokens': 0}
        
        # 返回score作为主要结果，同时提供完整结果字典供详细分析
        return results

    def _infer_task_type(self, dataset:torch.utils.data.Dataset=None):
        """Infer task type based on model characteristics"""
        if hasattr(self.model, '_model_type'):
            if self.model._model_type == 'CNN':
                return 'classification'
            elif self.model._model_type == 'LM':
                if dataset is not None:
                    dt = next(iter(dataset))
                    if isinstance(dt, dict):
                        if 'input_ids' in dt:
                            return 'sequence_classification'
                        elif 'text' in dt:
                            return 'language_modeling'
                    else:
                        raise ValueError(f"Unsupported dataset format: {type(dt)}")
                else:
                    return 'language_modeling'
        
        # Try to infer from model structure if no explicit model type is marked
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'problem_type'):
                return self.model.config.problem_type
            elif hasattr(self.model.config, 'architectures'):
                if any('ForCausalLM' in arch for arch in self.model.config.architectures):
                    return 'language_modeling'
                elif any('ForSequenceClassification' in arch for arch in self.model.config.architectures):
                    return 'sequence_classification'
        
        raise ValueError("Cannot infer task type, please specify task_type parameter explicitly")
    
    def get_layer(self, name:str, verbose=True) -> nn.Module:
        """Get a specific layer of model.

        Args:
            name (str): Layer name, split by dot(.), especially, can be 'all'.

        Returns:
            nn.Module: Target layer
        """
        if name == 'all':
            return self.model
        else:
            name_list = name.split('.')
            tar = self.model
            for i in name_list:
                try:
                    i = eval(i)
                except:
                    pass
                if isinstance(i, str):
                    tar = getattr(tar, i)
                elif isinstance(i, int):
                    tar = tar[i]
            if verbose:
                log(f"Target layer: \n {tar}")
            return tar
    
    def get_para(self, layer:Union[str, nn.Module]='all', type_:str='list', verbose=True) -> Union[list, Generator]:
        """Get target layer's parameters, be careful that this method would not return a dict.

        Args:
            layer (Union[str, nn.Module]): Layer name or layer module, name splited by '.', for example, 'layer1.0.conv1'.
            type_ (str, optional): Type of parameters, list or generator. Defaults to 'list'.

        Returns:
            Union[list, Generator]: Layer's parameters.
        """
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        para = layer.parameters()
        list_para = list(para)
        if verbose:
            for ind, p in enumerate(list_para):
                log(f"Index {ind}: shape {list(p.shape)}, min value {torch.min(p).item()}, max value {torch.max(p).item()}.")
        if type_ == 'list':
            return list_para
        elif type_ == 'generator':
            return para
        else:
            raise ValueError(f"Make sure that type_ is in 'list' or 'generator'.")
    
    def get_grad(self, layer:Union[str, nn.Module], type_:str='dict', verbose=True) -> Union[list, dict]:
        """Get target layer's gradients.
           Make sure that model has been trained.

        Args:
            layer (Union[str, nn.Module]): Layer name split by dot.
            verbose (bool, optional): If show verbose infomation. Defaults to True.

        Returns:
            Union[list, dict: Gradient list / dict.
        """
        gradients = {}
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose=False)
        for name, para in layer.named_parameters():
            if para.grad is not None:
                gradients[name] = para.grad
                if verbose:
                    log(f"Name {name}: shape {list(para.shape)}, min value {torch.min(para.grad).item()}, max value {torch.max(para.grad).item()}.")
            else:
                if verbose:
                    log(f"Name {name}: no gradients.")
        if type_ == 'dict':
            return gradients
        else:
            return [x[1] for x in gradients.items()]
    
    def plot_histogram(self, tensors:list[torch.Tensor], bin:int=30, save_path='tensor_hist.png'):
        """
        Plot a histogram of the values of a flattened tensor.
        
        Parameters:
        tensor (list[torch.Tensor]): A list of tensors.
        bins (int): Number of bins for the histogram.
        """
        assert len(tensors) != 0, ValueError(f"Make sure that tensor list has values, now it is {tensors}.")
        num_tensors = len(tensors)
        num_cols = ceil(sqrt(num_tensors))
        num_rows = ceil(num_tensors / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        
        # 处理单个子图的情况
        if num_rows == 1 and num_cols == 1:
            axes = [axes]  # 将单个Axes对象包装成列表
        else:
            # 将多维数组展平为一维
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes.reshape(-1)
        
        for i, tensor in enumerate(tensors):
            flattened_tensor = tensor.flatten().detach().cpu().numpy()
            axes[i].hist(flattened_tensor, bins=bin, edgecolor='k', alpha=0.7)
            axes[i].set_title(f'Tensor {i+1}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        
        # 只有在有多个子图的情况下才需要删除多余的子图
        if len(axes) > i + 1:
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path if hasattr(self, 'log_path') else '.', save_path))



if __name__ == "__main__":
    from data import get_dataset
    from models.model_zoo import get_model
    
    # # Test CV model with CIFAR-10 dataset
    # print("\n" + "="*50)
    # print("TESTING CV MODEL WITH TOY DATASET")
    # print("="*50)
    
    # # Get CIFAR-10 dataset
    # cifar10 = get_dataset('cifar10', train=True)
    # print(f"CIFAR-10 dataset loaded, size: {len(cifar10)}")
    
    # # Get ResNet18 model
    # resnet18 = get_model('resnet18', pretrained=True, num_classes=10)
    # resnet18.to(CONF.device)
    # print(f"ResNet18 model loaded on {CONF.device}")
    
    # # Use ModelInspector to check model
    # cv_inspector = ModelInspector(resnet18)
    # cv_inspector.summary((3, 32, 32))
    
    # # Check specific layer
    # cv_layer = 'layer1.0.conv1'
    # layer = cv_inspector.get_layer(cv_layer)
    # params = cv_inspector.get_para(cv_layer)
    # cv_inspector.plot_histogram(params, save_path='resnet18_params.png')
    
    # Test NLP model with IMDB dataset
    print("\n" + "="*50)
    print("TESTING LLM MODEL WITH HUGGINGFACE DATASET")
    print("="*50)
    
    # Get dataset
    # imdb = get_dataset('imdb')
    # print(f"IMDB dataset loaded, size: {len(imdb)}")
    wikitext2 = get_dataset('wikitext2')
    print(f"Wikitext2 dataset loaded, size: {len(wikitext2)}")
    
    # Get Llama model (or use a smaller model like GPT-2)
    llama2, tokenizer = get_model('Qwen2.5-3B', cache_dir=CONF.cache_dir)
    
    # # Try to move model to GPU
    # try:
    #     llama2.to(CONF.device)
    #     print(f"LLM model loaded on {CONF.device}")
    # except RuntimeError as e:
    #     print(f"Model too large for {CONF.device}, keeping on CPU: {e}")
    
    # Use ModelInspector to check model
    nlp_inspector = ModelInspector(llama2, tokenizer)
    nlp_inspector.summary()

    nlp_inspector.calibrate(wikitext2)
    
    # # Check specific layer
    # nlp_layer = 'model.layers.0.self_attn'
    # layer = nlp_inspector.get_layer(nlp_layer)
    # params = nlp_inspector.get_para(nlp_layer)
    
    # Only take the first two parameters to plot histogram, avoid memory issues
    # nlp_inspector.plot_histogram(params[:2], save_path='llama_params.png')
    
    print("\nTesting completed!")