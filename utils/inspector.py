import os
import torch
from torch import nn
from typing import Union, Generator, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from torchsummary import summary
from utils.io import LogRedirectMixin, log, generate_name
from config import CONF
from collections import OrderedDict

import torch
import transformers
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
                # Try to move the whole model to CPU to avoid device inconsistency problem
                try:
                    # Save the original device for later restoration
                    original_device = next(self.model.parameters()).device
                    self.model = self.model.cpu()
                    summary(self.model, mock_inp, device='cpu')
                    # Restore the model to the original device after operation
                    self.model = self.model.to(original_device)
                except Exception as e:
                    print(f"Error during model summary on CPU: {e}")
                    # If the CPU summary fails, print basic information
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
            # Language modeling task evaluation
            total_loss = 0.0
            total_tokens = 0
            
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                    
                try:
                    # Process different input formats
                    if isinstance(batch, dict):
                        # Check if input_ids is a list of strings
                        if 'input_ids' in batch and isinstance(batch['input_ids'], list) and isinstance(batch['input_ids'][0], str):
                            # Need to preprocess text first
                            if self.tokenizer is None:
                                log(f"输入是文本但未提供tokenizer")
                                continue
                                
                            # Convert text to token ids
                            encoded = self.tokenizer(
                                batch['input_ids'],  # Here is a list of texts
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=512  # Can be adjusted as needed
                            )
                            batch_on_device = {k: v.to(CONF.device) for k, v in encoded.items()}
                        else:
                            # The data is already tokenized, just move it to the correct device
                            batch_on_device = {k: v.to(CONF.device) if isinstance(v, torch.Tensor) else v 
                                              for k, v in batch.items()}
                        
                        # For datasets without labels (common case), create shifted labels for calculating next token predictions
                        if 'input_ids' in batch_on_device:
                            # Shift input_ids by one to create labels
                            input_ids = batch_on_device['input_ids']
                            
                            # If there is an attention_mask, use it to calculate the number of valid tokens
                            if 'attention_mask' in batch_on_device:
                                attention_mask = batch_on_device['attention_mask']
                                # Count valid tokens (ignoring padding)
                                batch_tokens = attention_mask.sum().item()
                            else:
                                # If there is no mask, assume all tokens are valid
                                batch_tokens = input_ids.numel()
                                
                            # Forward pass
                            outputs = self.model(**batch_on_device)
                            
                            # If the model has calculated the loss
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                batch_loss = outputs.loss * batch_tokens  # Convert to total loss
                            else:
                                # Otherwise, manually calculate the loss
                                # Shift the input sequence: the prediction target is the next token
                                shift_logits = outputs.logits[..., :-1, :].contiguous()
                                shift_labels = input_ids[..., 1:].contiguous()
                                
                                # Use cross-entropy loss
                                loss_fct = nn.CrossEntropyLoss(reduction='sum')  # Use sum for easier calculation of total token loss
                                batch_loss = loss_fct(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1)
                                )
                                
                                # Calculate the number of valid tokens (excluding padding)
                                if 'attention_mask' in batch_on_device:
                                    # Shift the attention_mask to match shift_labels
                                    shift_mask = attention_mask[..., 1:].contiguous()
                                    batch_tokens = shift_mask.sum().item()
                                else:
                                    batch_tokens = shift_labels.numel()

                            # Backward pass
                            batch_loss.backward()

                            # Accumulate total loss and token count
                            total_loss += batch_loss.item()
                            total_tokens += batch_tokens
                            
                    elif isinstance(batch, list) and all(isinstance(item, str) for item in batch):
                        # The case where the input is a list of strings
                        if self.tokenizer is None:
                            log(f"Input is text but no tokenizer is provided")
                            continue
                            
                        # Use tokenizer to process text
                        encoded = self.tokenizer(
                            batch,  # Text list
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512  # Can be adjusted as needed
                        )
                        batch_on_device = {k: v.to(CONF.device) for k, v in encoded.items()}
                        
                        # Get input_ids for subsequent processing
                        input_ids = batch_on_device['input_ids']
                        
                        # If there is an attention_mask, use it to calculate the number of valid tokens
                        if 'attention_mask' in batch_on_device:
                            attention_mask = batch_on_device['attention_mask']
                            batch_tokens = attention_mask.sum().item()
                        else:
                            batch_tokens = input_ids.numel()
                        
                        # Forward pass
                        outputs = self.model(**batch_on_device)
                        
                        # Process loss calculation as before
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            batch_loss = outputs.loss * batch_tokens  # Convert to total loss
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

                        # Backward pass
                        batch_loss.backward()

                        # Accumulate total loss and token count
                        total_loss += batch_loss.item()
                        total_tokens += batch_tokens

                    else:
                        log(f"Unsupported input format: {type(batch)}")
                        continue
                        
                except Exception as e:
                    log(f"Batch processing error: {str(e)}")
                    import traceback
                    log(traceback.format_exc())
                
            # Calculate average loss and perplexity
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                
                # Normalize perplexity to 0-1 as score
                # Use a reasonable threshold (e.g., 20) as high perplexity
                max_reasonable_ppl = 20.0
                score = max(0.0, 1.0 - min(perplexity / max_reasonable_ppl, 1.0))
                
                log(f"Language modeling evaluation - Total tokens: {total_tokens}")
                log(f"Language modeling evaluation - Average loss: {avg_loss:.4f}")
                log(f"Language modeling evaluation - Perplexity (PPL): {perplexity:.4f}")
                log(f"Language modeling evaluation - Normalized score: {score:.4f}")
                
                results = {
                    'loss': avg_loss,
                    'perplexity': perplexity,
                    'score': score,
                    'total_tokens': total_tokens
                }
                # Update status
                self.status = 'trained'
                
            else:
                log(f"No valid tokens were processed")
                results = {'loss': float('inf'), 'perplexity': float('inf'), 'score': 0.0, 'total_tokens': 0}
        
        # Return score as the main result, while providing the complete result dictionary for detailed analysis
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
        """Get gradients of layer parameters.

        Args:
            layer (Union[str, nn.Module]): Layer name or module.
            type_ (str, optional): Return type, 'list' or 'dict'. Defaults to 'dict'.
            verbose (bool, optional): Whether to print information. Defaults to True.

        Returns:
            Union[list, dict]: Gradients of layer parameters.
        """
        if isinstance(layer, str):
            layer = self.get_layer(layer, verbose)
        
        if not layer.parameters():
            if verbose:
                print("Layer has no parameters.")
            return [] if type_ == 'list' else {}
        
        if type_ == 'list':
            return [p.grad for p in layer.parameters() if p.grad is not None]
        else:
            return {name: p.grad for name, p in layer.named_parameters() if p.grad is not None}
    
    def get_representation(self, layers:list[str], inputs:Any, forward_type:Optional[str]=None, detach:bool=True, split_attention_heads:bool=False, num_heads:Optional[int]=None, head_dim:Optional[int]=None, verbose:bool=True) -> dict:
        """Get the representation of specified layers (the output of forward propagation).

        Args:
            layers (list[str]): The name list of layers to get the representation.
            inputs (Any): The input data.
            forward_type (Optional[str], optional): The type of forward propagation. If None, call the model directly;
                                                     if the input is a dictionary, unpack it and pass it in;
                                                     if it is a custom method name, call the corresponding method. Defaults to None.
            detach (bool, optional): Whether to detach the representation, avoid gradient calculation. Defaults to True.
            split_attention_heads (bool, optional): Whether to split the attention heads. Defaults to False.
            num_heads (Optional[int], optional): The number of attention heads. If None, try to infer from model configuration. Defaults to None.
            head_dim (Optional[int], optional): The dimension of each attention head. If None, try to calculate automatically. Defaults to None.
            verbose (bool, optional): Whether to print information. Defaults to True.
        Returns:
            dict: A dictionary mapping layer names to representations.
        """
        representations = {}
        handles = []
        representations['meta'] = {}
        representations['meta']['inputs'] = inputs
        
        # Register hooks for each layer
        for layer_name in layers:
            layer = self.get_layer(layer_name, verbose=verbose)
            
            def hook_fn(name):
                def hook(module, input, output):
                    if detach:
                        if isinstance(output, torch.Tensor):
                            representations[name] = output.detach()
                        else:
                            # Handle the case where the output is a tuple or list
                            representations[name] = output[0].detach() if isinstance(output, (tuple, list)) else output
                    else:
                        if isinstance(output, torch.Tensor):
                            representations[name] = output
                        else:
                            representations[name] = output[0] if isinstance(output, (tuple, list)) else output
                return hook
            
            handle = layer.register_forward_hook(hook_fn(layer_name))
            handles.append(handle)
        
        # Move inputs to device
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if verbose:
                    if isinstance(v, torch.Tensor):
                        log(f"Input {k}: shape {v.shape}")
                    elif isinstance(v, str):
                        log(f"Input {k}: {v}")
                    else:
                        log(f"Input {k}: {v}")
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(CONF.device)
                if 'max_new_tokens' in inputs:  # If the input is a dict, and contains 'max_new_tokens', then convert it to int
                    inputs['max_new_tokens'] = int(inputs['max_new_tokens'])
        else:
            inputs = inputs.to(CONF.device)
        
        # Execute forward propagation
        try:
            if forward_type is None:
                if isinstance(inputs, dict) or isinstance(inputs, transformers.BatchEncoding):
                    self.model(**inputs)
                else:
                    self.model(inputs)
            else:
                forward_method = getattr(self.model, forward_type)
                if isinstance(inputs, dict) or isinstance(inputs, transformers.BatchEncoding):
                    outputs = forward_method(**inputs)
                else:
                    outputs = forward_method(inputs)
                representations['meta']['outputs'] = outputs
                if verbose:
                    log(f"Outputs: {outputs}")
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # If split_attention_heads is True, split the attention heads
        if split_attention_heads:
            for layer_name in layers:
                if layer_name in representations:
                    representations[layer_name] = self.split_heads(
                        representations[layer_name], 
                        num_heads=num_heads, 
                        head_dim=head_dim
                    )
        
        return representations
    
    def split_heads(self, tensor: torch.Tensor, num_heads: Optional[int] = None, head_dim: Optional[int] = None) -> torch.Tensor:
        """Split the attention heads in the representation.

        Args:
            tensor (torch.Tensor): The tensor to split the attention heads.
            num_heads (Optional[int], optional): The number of attention heads. If None, try to infer from model configuration. Defaults to None.
            head_dim (Optional[int], optional): The dimension of each attention head. If None, try to calculate automatically. Defaults to None.

        Returns:
            torch.Tensor: The separated attention head representation, shape is [batch_size, seq_len, num_heads, head_dim].
        """
        # Check if the input is a tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("The input must be a torch.Tensor.")
            
        # Get the shape of the tensor
        shape = tensor.shape
        
        # If num_heads is None, try to infer from model configuration
        if num_heads is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "num_attention_heads"):
                num_heads = self.model.config.num_attention_heads
            else:
                raise ValueError("num_heads is not specified and cannot be inferred from model configuration, please specify it manually.")
        
        # Handle different shapes of tensors
        if len(shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = shape
            
            # If head_dim is None, calculate the dimension of each head
            if head_dim is None:
                if hidden_dim % num_heads != 0:
                    raise ValueError(f"The hidden dimension {hidden_dim} cannot be divided by the number of attention heads {num_heads}.")
                head_dim = hidden_dim // num_heads
            
            # Reshape the tensor, split the attention heads
            # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
            return tensor.view(batch_size, seq_len, num_heads, head_dim)
            
        elif len(shape) == 4 and shape[2] == shape[3]:  # It may be an attention matrix [batch_size, num_heads, seq_len, seq_len]
            # In this case, the tensor may already contain separated attention heads
            return tensor
            
        else:
            # If it is other shapes, try to split the tensor according to num_heads
            if len(shape) >= 2:
                # Assume the last dimension is hidden_dim
                hidden_dim = shape[-1]
                
                if head_dim is None:
                    if hidden_dim % num_heads != 0:
                        raise ValueError(f"The dimension {hidden_dim} cannot be divided by the number of attention heads {num_heads}.")
                    head_dim = hidden_dim // num_heads
                
                # Build a new shape, keep the previous dimensions unchanged, and split the last dimension
                new_shape = list(shape[:-1]) + [num_heads, head_dim]
                return tensor.view(*new_shape)
            else:
                raise ValueError(f"Cannot handle the tensor with shape {shape}, at least 2 dimensions are required.")
    
    def get_attention_scores(self, layers: list[str], inputs: Any, forward_type: Optional[str] = None) -> dict:
        """Get the attention matrix (attention scores).

        Args:
            layers (list[str]): The name list of layers to get the attention scores (usually self-attention layers).
            inputs (Any): The input data.
            forward_type (Optional[str], optional): The type of forward propagation, same as get_representation. Defaults to None.

        Returns:
            dict: A dictionary mapping layer names to attention scores.
        """
        attention_scores = {}
        handles = []
        
        # Register hooks for each layer
        for layer_name in layers:
            layer = self.get_layer(layer_name, verbose=False)
            
            # Set different hooks for different types of attention layers
            def hook_fn(name):
                def hook(module, input, output):
                    # The output format of different attention layers may be different
                    # Usually, the attention scores are in the first element of the output tuple
                    # Here, we assume that the attention scores are in the first element of the output tuple, and you may need to adjust it in actual cases
                    if isinstance(output, tuple) and len(output) > 1:
                        # For some models (e.g., BERT), the attention scores are in the second position
                        scores = output[1] if len(output) > 1 else output[0]
                        attention_scores[name] = scores.detach()
                    else:
                        # If the output is not a tuple, you may need additional logic to extract the attention scores
                        attention_scores[name] = output.detach() if isinstance(output, torch.Tensor) else None
                return hook
            
            # Register hooks
            handle = layer.register_forward_hook(hook_fn(layer_name))
            handles.append(handle)
        
        # Execute forward propagation
        try:
            # Reuse the logic in get_representation for forward propagation
            self.get_representation(layers=[], inputs=inputs, forward_type=forward_type)
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        return attention_scores

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
        
        # Handle the case where there is only one subplot
        if num_rows == 1 and num_cols == 1:
            axes = [axes]  # Wrap the single Axes object in a list
        else:
            # Flatten the multi-dimensional array into a one-dimensional array
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes.reshape(-1)
        
        for i, tensor in enumerate(tensors):
            flattened_tensor = tensor.flatten().detach().cpu().numpy()
            axes[i].hist(flattened_tensor, bins=bin, edgecolor='k', alpha=0.7)
            axes[i].set_title(f'Tensor {i+1}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        
        # Only delete extra subplots if there are multiple subplots
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
    qwen25, tokenizer = get_model('Qwen2.5-3B', cache_dir=CONF.cache_dir)

    case0 = "Hello, how are you?"
    encoded = tokenizer(case0,  # Here is a list of texts
                        return_tensors='pt')

    # Use ModelInspector to check model
    nlp_inspector = ModelInspector(qwen25, tokenizer)
    # nlp_inspector.summary()
    # nlp_inspector.calibrate(wikitext2)
    nlp_layers = ['model.layers.0.self_attn', 'model.layers.0.mlp']
    reps = nlp_inspector.get_representation(nlp_layers, encoded)
    print(reps)
    
    # # Check specific layer
    # nlp_layer = 'model.layers.0.self_attn'
    # layer = nlp_inspector.get_layer(nlp_layer)
    # params = nlp_inspector.get_para(nlp_layer)
    
    # Only take the first two parameters to plot histogram, avoid memory issues
    # nlp_inspector.plot_histogram(params[:2], save_path='llama_params.png')
    
    print("\nTesting completed!")