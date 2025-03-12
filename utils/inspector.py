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
    
    def get_representation(self, layers:list[str], inputs:Any, forward_type:Optional[str]=None, 
                           detach:bool=True, split_attention_heads:bool=False, num_heads:Optional[int]=None, 
                           head_dim:Optional[int]=None, verbose:bool=True, stack_outputs:bool=True) -> dict:
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
            stack_outputs (bool, optional): Whether to stack multiple outputs from repeated forward passes (e.g. in generate method). Defaults to True.
        Returns:
            dict: A dictionary mapping layer names to representations.
        """
        representations = {}
        handles = []
        representations['meta'] = {}
        representations['meta']['inputs'] = inputs
        
        # Initialize containers for layer representations
        for layer_name in layers:
            representations[layer_name] = [] if stack_outputs else None
        
        # Register hooks for each layer
        for layer_name in layers:
            layer = self.get_layer(layer_name, verbose=False)  # Dont show layer info
            
            def hook_fn(name):
                def hook(module, input, output):
                    # Process the output
                    if isinstance(output, torch.Tensor):
                        processed_output = output.detach() if detach else output
                    else:
                        # Handle the case where the output is a tuple or list
                        processed_output = output[0].detach() if isinstance(output, (tuple, list)) and detach else output[0] if isinstance(output, (tuple, list)) else output
                        if detach and isinstance(processed_output, torch.Tensor):
                            processed_output = processed_output.detach()
                    
                    # Store the output
                    if stack_outputs:
                        representations[name].append(processed_output)
                    else:
                        representations[name] = processed_output
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
        
        # If stacking is enabled, process the collected outputs
        if stack_outputs:
            for layer_name in layers:
                if layer_name in representations and representations[layer_name]:
                    # Check if tensors can be stacked (have same shape)
                    tensors = representations[layer_name]
                    if all(isinstance(t, torch.Tensor) for t in tensors) and all(t.shape == tensors[0].shape for t in tensors):
                        # Stack tensors if they have the same shape
                        representations[layer_name] = torch.stack(tensors, dim=0)
                    else:
                        # If shapes differ or not all are tensors, keep as list
                        if verbose:
                            log(f"Warning: Could not stack tensors for layer {layer_name} due to different shapes or types")
        
        # If split_attention_heads is True, split the attention heads
        if split_attention_heads:
            for layer_name in layers:
                if layer_name in representations:
                    representation = representations[layer_name]
                    
                    if stack_outputs:
                        # Case 1: If stacking succeeded and we have a stacked tensor
                        if isinstance(representation, torch.Tensor) and len(representation.shape) >= 3:
                            # For tensors with shape [num_steps, batch, seq_len, hidden_dim]
                            if len(representation.shape) == 4:
                                num_steps = representation.shape[0]
                                split_heads_results = []
                                
                                for i in range(num_steps):
                                    split_heads_results.append(self.split_heads(
                                        representation[i],
                                        num_heads=num_heads,
                                        head_dim=head_dim
                                    ))
                                
                                # Try to stack the results if shapes are consistent
                                if all(isinstance(t, torch.Tensor) for t in split_heads_results) and all(t.shape == split_heads_results[0].shape for t in split_heads_results):
                                    representations[layer_name] = torch.stack(split_heads_results, dim=0)
                                else:
                                    representations[layer_name] = split_heads_results
                            else:
                                # For a single tensor with appropriate dimensions
                                representations[layer_name] = self.split_heads(
                                    representation,
                                    num_heads=num_heads,
                                    head_dim=head_dim
                                )
                        
                        # Case 2: If stacking failed and we have a list of tensors
                        elif isinstance(representation, list):
                            split_heads_results = []
                            
                            for tensor in representation:
                                if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                                    split_heads_results.append(self.split_heads(
                                        tensor,
                                        num_heads=num_heads,
                                        head_dim=head_dim
                                    ))
                                else:
                                    # Skip non-tensor or inappropriate dimension tensors
                                    split_heads_results.append(tensor)
                                    if verbose:
                                        log(f"Warning: Could not split heads for a tensor in layer {layer_name}")
                            
                            representations[layer_name] = split_heads_results
                    
                    # Case 3: If stack_outputs is False, we just have a single tensor
                    else:
                        if isinstance(representation, torch.Tensor) and len(representation.shape) >= 2:
                            representations[layer_name] = self.split_heads(
                                representation,
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

    def show_attention(self, raw_inputs, attention_representation, **kwargs):
        """
        可视化模型中的注意力权重分布
        
        参数:
            raw_inputs (Union[str, dict]): 原始输入数据，可以是文本字符串或包含图像和文本的字典
            attention_representation (torch.Tensor): 注意力权重，形状为[1, head_num, token_num, token_num]
            **kwargs: 其他配置参数，包括：
                head_indices: 要可视化的注意力头索引，默认为None（使用所有头的平均值）
                tokenizer: 用于将token ID映射回文本的tokenizer
                image_processor: 用于处理图像token映射的函数
                is_multimodal: 是否为多模态输入，默认为False
                image_first: 多模态输入中图像是否在前，默认为True
                image_size: 输入图像的大小，默认为(448, 448)
                patch_size: 图像patch的大小，默认为14
                merged_patch_count: 每个token合并的patch数量，默认为4
                cmap: 热力图的颜色映射，默认为'viridis'
                alpha: 热力图的透明度，默认为0.7
                output_path: 保存可视化结果的路径，默认为None
                figsize: 图像大小，默认为None
                title: 图像的总标题，默认为None
        
        返回:
            matplotlib.figure.Figure: 可视化结果的Figure对象
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # 使用默认配置，可以被kwargs中的参数覆盖
        config = {
            'head_indices': None,
            'tokenizer': None,
            'image_processor': None,
            'is_multimodal': False,
            'image_first': True,
            'image_size': (448, 448),
            'patch_size': 14,
            'merged_patch_count': 4,
            'cmap': 'viridis',
            'alpha': 0.7,
            'output_path': None,
            'figsize': None,
            'title': None
        }
        
        # 使用传入的参数更新配置
        config.update(kwargs)
        
        # 确保输入批次大小为1
        if attention_representation.shape[0] != 1:
            raise ValueError("Batch size must be 1 for attention visualization")
        
        # 确保整个注意力表示先移动到CPU
        attention_representation = attention_representation.cpu()
        
        if config['head_indices'] is not None:
            if isinstance(config['head_indices'], int):
                config['head_indices'] = [config['head_indices']]
            # 确保先移动到CPU
            attention_weights = attention_representation[0, config['head_indices']].mean(dim=0).detach().numpy()
        else:
            # 确保先移动到CPU
            attention_weights = attention_representation[0].mean(dim=0).detach().numpy()
        
        # 设置图像大小
        if config['figsize'] is None:
            config['figsize'] = (18, 10) if config['is_multimodal'] else (12, 8)
        
        # 创建图像
        fig = plt.figure(figsize=config['figsize'])
        
        if not config['is_multimodal']:
            # 文本输入的可视化
            self._visualize_text_attention(raw_inputs, attention_weights, config['tokenizer'], 
                                          config['cmap'], config['alpha'], fig, config['title'])
        else:
            # 多模态输入的可视化
            self._visualize_multimodal_attention(raw_inputs, attention_weights, config['tokenizer'], 
                                               config['image_processor'], config['image_first'], 
                                               config['image_size'], config['patch_size'], 
                                               config['merged_patch_count'], config['cmap'], 
                                               config['alpha'], fig, config['title'])
        
        # 保存图像
        if config['output_path']:
            plt.savefig(config['output_path'], bbox_inches='tight', dpi=300)
            print(f"注意力可视化结果已保存至: {config['output_path']}")
            plt.close(fig)  # 关闭图形以释放资源
        
        return fig

    def _visualize_text_attention(self, text_input, attention_weights, tokenizer, cmap, alpha, fig, title=None):
        """可视化文本输入的注意力权重"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # 获取token和文本映射
        if isinstance(text_input, str) and tokenizer is not None:
            tokens = tokenizer.tokenize(text_input)
        elif hasattr(text_input, 'tokens'):
            tokens = text_input.tokens
        else:
            # 假设输入是已经分词的列表
            tokens = text_input if isinstance(text_input, list) else ["Token_" + str(i) for i in range(attention_weights.shape[0])]
        
        # 确保token数量与注意力矩阵维度匹配
        token_count = min(len(tokens), attention_weights.shape[0])
        
        # 创建子图网格
        rows = token_count
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # 为每个token创建一个文本条带
        y_positions = np.linspace(0, 1, rows + 2)[1:-1]  # 避免边缘
        
        # 显示每个token的文本
        for i, token in enumerate(tokens[:token_count]):
            # 计算该token关注其他token的权重
            attn_row = attention_weights[i, :token_count]
            max_attn = attn_row.max()
            norm_attn = attn_row / max_attn if max_attn > 0 else attn_row
            
            # 在每个token位置显示带有注意力颜色背景的文本
            for j, (t, weight) in enumerate(zip(tokens[:token_count], norm_attn)):
                x_start = j / token_count
                x_end = (j + 1) / token_count
                color_map = plt.cm.get_cmap(cmap)
                color = color_map(weight)
                
                # 绘制背景矩形
                rect = plt.Rectangle((x_start, y_positions[i] - 0.03), 
                                   (x_end - x_start), 0.06, 
                                   facecolor=color, alpha=alpha * weight)
                ax.add_patch(rect)
                
                # 添加文本
                ax.text((x_start + x_end) / 2, y_positions[i], t, 
                       ha='center', va='center', fontsize=12, 
                       color='black' if weight < 0.6 else 'white')
        
        # 设置子图标题
        ax.set_title("Attention Visualization for Each Token")
        
        # 设置总标题（如果提供）
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _visualize_multimodal_attention(self, inputs, attention_weights, tokenizer, image_processor, 
                                           image_first, image_size, patch_size, merged_patch_count,
                                           cmap, alpha, fig, title=None):
        """可视化多模态输入的注意力权重，左侧显示原始内容，右侧显示注意力热力图

        参数:
            inputs: 输入数据，可以是图像和文本的字典或元组
            attention_weights: 注意力权重，形状为[token_num, token_num]
            tokenizer: 用于将token ID映射回文本的tokenizer
            image_processor: 用于处理图像token映射的函数
            image_first: 多模态输入中图像是否在前
            image_size: 输入图像的大小
            patch_size: 图像patch的大小
            merged_patch_count: 每个token合并的patch数量
            cmap: 热力图的颜色映射
            alpha: 热力图的透明度
            fig: matplotlib图形对象
            title: 图像的总标题，默认为None
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # 解析多模态输入
        if isinstance(inputs, dict):
            image = inputs.get('image') or inputs.get('images')
            text = inputs.get('text') or inputs.get('texts')
        else:
            # 假设是元组或列表
            image, text = inputs if len(inputs) >= 2 else (inputs[0], None)
        
        # 获取图像和文本的token数量
        total_tokens = attention_weights.shape[0]
        
        # 计算图像token的数量
        patch_grid_size = int(image_size[0] / patch_size)
        original_patches = patch_grid_size * patch_grid_size  # 如448x448和14x14的patch得到32x32=1024个patch
        merged_token_count = original_patches // merged_patch_count  # 例如合并为256个token
        
        # 映射token到内容的对应关系
        token_mapping = self._map_tokens_to_content(
            image, text, tokenizer, image_first, 
            image_size, patch_size, merged_patch_count, 
            merged_token_count, total_tokens
        )
        
        # 创建一个新的等分网格布局，确保等大小的显示区域
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
        
        # 处理图像数据
        if isinstance(image, torch.Tensor):
            # 转移到CPU并转为numpy
            image_np = image.cpu().numpy()
            # 处理批次维度
            if len(image_np.shape) == 4:  # [batch, channel, height, width]
                image_np = image_np.squeeze(0)  # 移除批次维度
            # 转换通道顺序，从[C,H,W]变为[H,W,C]
            if image_np.shape[0] == 3 and len(image_np.shape) == 3:  # 如果是[C,H,W]格式
                image_np = np.transpose(image_np, (1, 2, 0))  # 转为[H,W,C]格式
        elif isinstance(image, np.ndarray):
            # 处理numpy数组
            if len(image.shape) == 4:  # [batch, channel, height, width]
                image_np = image.squeeze(0)
            # 转换通道顺序，从[C,H,W]变为[H,W,C]
            if image_np.shape[0] == 3 and len(image_np.shape) == 3:  # 如果是[C,H,W]格式
                image_np = np.transpose(image_np, (1, 2, 0))  # 转为[H,W,C]格式
            else:
                image_np = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 检查图像形状是否符合matplotlib要求
        print(f"处理后的图像形状: {image_np.shape}")
        assert len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] in [1, 3, 4]), \
            f"图像形状不符合matplotlib要求: {image_np.shape}"

        # 确保值在合理范围内并检查图像数据
        if np.issubdtype(image_np.dtype, np.floating):
            if image_np.max() > 1.0:
                image_np = image_np / 255.0  # 如果值超过1，假设是0-255范围，转为0-1
            
            # 打印图像统计信息以便调试
            print(f"图像数值范围: min={image_np.min()}, max={image_np.max()}, mean={image_np.mean()}")
            
            # 确保图像不是全黑
            if image_np.max() < 0.1:  # 如果图像几乎是全黑的
                print("警告: 图像可能过暗，尝试增强对比度")
                # 增强对比度
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

        # 计算注意力权重的均值 - 将矩阵压缩为一行
        mean_attention = np.mean(attention_weights, axis=0)  # 对所有token的注意力取平均
        
        # 打印注意力值统计信息以便调试
        print(f"注意力值范围: min={mean_attention.min()}, max={mean_attention.max()}, mean={mean_attention.mean()}")
        
        # 归一化注意力值，确保有足够的对比度
        if mean_attention.max() > mean_attention.min():
            # 使用更强的对比度拉伸
            normalized_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
            # 应用对比度增强 - 使用幂函数增强对比度，小值更小，大值更大
            normalized_attention = np.power(normalized_attention, 0.7)  # 0.7是一个可调参数，小于1增加对比度
        else:
            normalized_attention = np.zeros_like(mean_attention)
            
        print(f"归一化后注意力值范围: min={normalized_attention.min()}, max={normalized_attention.max()}, mean={normalized_attention.mean()}")
        
        # 获取色图
        cmap_func = plt.cm.get_cmap(cmap)
        
        # 创建与原图相同大小的空白注意力图
        h, w = image_np.shape[:2]
        attention_img = np.zeros((h, w, 4))  # RGBA图像
        
        # 根据token映射填充注意力图
        for token_idx, token_info in token_mapping.items():
            if token_info['type'] == 'image':
                # 获取像素区域
                top, left, bottom, right = token_info['pixel_region']
                
                # 获取该token的注意力值并使用归一化后的值
                if token_idx < len(normalized_attention):
                    attention_value = normalized_attention[token_idx]
                    
                    # 根据注意力值获取颜色
                    color = cmap_func(attention_value)
                    
                    # 确保区域大小在图像范围内
                    top = min(top, h-1)
                    bottom = min(bottom, h)
                    left = min(left, w-1)
                    right = min(right, w)
                    
                    if top < bottom and left < right:
                        # 填充对应区域
                        attention_img[top:bottom, left:right, :] = color
        
        # 设置总标题（如果提供）
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        
        # 创建具有相同大小的子图 - 图像区域（左上角）
        ax_img = fig.add_subplot(gs[0, 0])
        
        # 原始图像区域 - 左上角
        ax_img.imshow(image_np)
        ax_img.set_title("Input Image")
        ax_img.axis('off')
        
        # 创建具有相同大小的子图 - 注意力热力图区域（右上角）
        ax_img_attn = fig.add_subplot(gs[0, 1])
        
        # 注意力热力图 - 右上角，使用与原图相同的extent参数确保显示比例一致
        # 使用与原图完全相同的绘图参数
        extent = ax_img.get_xlim() + ax_img.get_ylim()  # 组合x和y的范围
        ax_img_attn.imshow(attention_img, extent=extent)
        ax_img_attn.set_title("Image Attention Heatmap")
        ax_img_attn.axis('off')
        
        # 确保两个子图具有相同的坐标系
        ax_img_attn.set_xlim(ax_img.get_xlim())
        ax_img_attn.set_ylim(ax_img.get_ylim())
        
        # 文本区域 - 左下角
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.axis('off')
        
        # 如果有文本，显示原始文本
        if text is not None:
            if isinstance(text, list):
                text = text[0]  # 取第一个文本
            ax_text.text(0.5, 0.5, text, 
                       ha='center', va='center', fontsize=12,
                       wrap=True)
        
        ax_text.set_title("Original Text")
        
        # 文本注意力热力图 - 右下角
        ax_text_attn = fig.add_subplot(gs[1, 1])
        ax_text_attn.axis('off')
        ax_text_attn.set_title("Text Attention Heatmap")
        
        # 提取文本token的注意力值并单独归一化以增强对比度
        text_token_indices = [idx for idx, info in token_mapping.items() if info['type'] == 'text']
        text_attention_values = [normalized_attention[idx] if idx < len(normalized_attention) else 0 for idx in text_token_indices]
        
        # 如果有文本token，重新归一化以增强对比度
        if text_attention_values:
            text_min = min(text_attention_values)
            text_max = max(text_attention_values)
            if text_max > text_min:
                # 对文本token的注意力值进行更强的归一化
                text_normalized = {idx: ((normalized_attention[idx] - text_min) / (text_max - text_min + 1e-8)) ** 0.5
                                   for idx in text_token_indices if idx < len(normalized_attention)}
                
                # 进一步增强对比度 - 应用更强的非线性变换
                for idx in text_normalized:
                    # 应用更强的非线性变换，保持0.5作为中心
                    text_normalized[idx] = 0.5 + (text_normalized[idx] - 0.5) * 1.5
                    # 裁剪到[0,1]范围
                    text_normalized[idx] = max(0, min(1, text_normalized[idx]))
            else:
                text_normalized = {idx: 0.5 for idx in text_token_indices if idx < len(normalized_attention)}
        
        # 创建文本token的注意力可视化
        text_tokens = [info for idx, info in token_mapping.items() if info['type'] == 'text']
        if text_tokens:
            text_token_count = len(text_tokens)
            
            for token_idx, token_info in token_mapping.items():
                if token_info['type'] == 'text':
                    # 获取token位置
                    position = token_info['position']
                    token = token_info['token']
                    
                    # 计算显示位置
                    x_start = position / text_token_count
                    x_width = 1.0 / text_token_count
                    
                    # 使用文本专用归一化的注意力值
                    if token_idx in text_normalized:
                        attention_value = text_normalized[token_idx]
                        
                        # 获取颜色 - 使用更明显的颜色范围
                        color = cmap_func(attention_value)
                        
                        # 创建背景矩形 - 增加高度使其更醒目
                        import matplotlib.patches as patches
                        rect = patches.Rectangle(
                            (x_start, 0.3), x_width, 0.4,  # 增加高度
                            linewidth=0, facecolor=color, alpha=min(alpha + 0.3, 1.0)  # 增加透明度上限
                        )
                        ax_text_attn.add_patch(rect)
                        
                        # 添加token文本
                        ax_text_attn.text(x_start + x_width/2, 0.5, token,
                                ha='center', va='center', fontsize=10,
                                color='black' if attention_value < 0.5 else 'white')
        
        # 应用紧凑布局，确保子图间的间隔最小
        plt.tight_layout(rect=[0, 0, 1, 0.96 if title else 1])  # 如果有总标题，留出空间
        
        # 不需要显示，直接保存到路径
        return token_mapping

    def _map_tokens_to_content(self, image, text, tokenizer, image_first, 
                              image_size, patch_size, merged_patch_count, 
                              image_token_count, total_tokens):
        """映射每个token到对应的图像区域或文本"""
        token_mapping = {}
        patch_grid_size = int(image_size[0] / patch_size)  # 例如448/14=32
        
        # 修正：计算每行应该有多少个token
        # 假设图像是正方形，则每行的token数量应该是 patch_grid_size 除以 合并因子的平方根
        # 例如，如果每4个patch合并为一个token（2x2），则每行token数为 patch_grid_size / 2
        # 通常情况下，对于448x448的图像和14x14的patch，patch_grid_size=32，如果每个token是一行4个patch，
        # 那么每行应该有 32/4 = 8 个token
        tokens_per_row = patch_grid_size // (merged_patch_count if merged_patch_count > 1 else 1)
        
        # 打印调试信息
        print(f"总token数: {total_tokens}, 图像token数: {image_token_count}")
        print(f"Patch网格大小: {patch_grid_size}x{patch_grid_size}, 每行token数: {tokens_per_row}")
        print(f"每{merged_patch_count}个patch合并为1个token")
        
        # 图像token映射
        if image_first:
            # 图像token在前
            for token_idx in range(image_token_count):
                # 计算token在网格中的位置
                token_row = token_idx // tokens_per_row
                token_col = token_idx % tokens_per_row
                
                # 计算原始patch的起始索引 - 每个token包含一个patch的高度和merged_patch_count个patch的宽度
                start_patch_row = token_row
                start_patch_col = token_col * merged_patch_count
                
                # 计算像素区域
                top = start_patch_row * patch_size
                left = start_patch_col * patch_size
                width = merged_patch_count * patch_size
                height = patch_size
                
                # 确保不超出图像边界
                bottom = min(top + height, image_size[0])
                right = min(left + width, image_size[1])
                
                print(f"Token {token_idx}: 位置({token_row},{token_col}), 区域({top},{left},{bottom},{right})")
                
                token_mapping[token_idx] = {
                    'type': 'image',
                    'position': (token_row, token_col),
                    'pixel_region': (top, left, bottom, right),
                    'patches': [(start_patch_row, start_patch_col + i) 
                               for i in range(merged_patch_count)]
                }
            
            # 文本token映射
            if tokenizer is not None and text is not None:
                # 使用tokenizer处理文本
                if hasattr(tokenizer, 'tokenize'):
                    text_to_tokenize = text[0] if isinstance(text, list) else text
                    tokens = tokenizer.tokenize(text_to_tokenize)
                else:
                    # 某些tokenizer没有tokenize方法
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    tokens = [tokenizer.decode([id]) for id in encoded]
                
                print(f"文本: {text}")
                print(f"分词结果: {tokens}")
                
                # 文本token从image_token_count开始
                for i, token in enumerate(tokens):
                    text_token_idx = image_token_count + i
                    if text_token_idx < total_tokens:
                        token_mapping[text_token_idx] = {
                            'type': 'text',
                            'token': token,
                            'position': i,
                            'original_text': text
                        }
        else:  # TODO: NOT TESTED
            # 文本token在前
            if tokenizer is not None and text is not None:
                # 使用tokenizer处理文本
                if hasattr(tokenizer, 'tokenize'):
                    text_to_tokenize = text[0] if isinstance(text, list) else text
                    tokens = tokenizer.tokenize(text_to_tokenize)
                else:
                    # 某些tokenizer没有tokenize方法
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    tokens = [tokenizer.decode([id]) for id in encoded]
                
                print(f"文本: {text}")
                print(f"分词结果: {tokens}")
                
                text_token_count = min(len(tokens), total_tokens - image_token_count)
                
                # 文本token在前
                for i, token in enumerate(tokens[:text_token_count]):
                    token_mapping[i] = {
                        'type': 'text',
                        'token': token,
                        'position': i,
                        'original_text': text
                    }
            
            # 图像token在后
            text_token_count = len(token_mapping)  # 已映射的文本token数量
            for token_idx in range(image_token_count):
                # 计算token在网格中的位置
                token_row = token_idx // tokens_per_row
                token_col = token_idx % tokens_per_row
                
                # 计算原始patch的起始索引
                start_patch_row = token_row
                start_patch_col = token_col * merged_patch_count  # 每个token对应连续的patch
                
                # 计算像素区域
                top = start_patch_row * patch_size
                left = start_patch_col * patch_size
                width = merged_patch_count * patch_size
                height = patch_size
                
                # 确保不超出图像边界
                bottom = min(top + height, image_size[0])
                right = min(left + width, image_size[1])
                
                image_token_idx = text_token_count + token_idx
                token_mapping[image_token_idx] = {
                    'type': 'image',
                    'position': (token_row, token_col),
                    'pixel_region': (top, left, bottom, right),
                    'patches': [(start_patch_row, start_patch_col + i) 
                               for i in range(merged_patch_count)]
                }
        
        # 打印token数量统计
        image_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'image']
        text_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'text']
        print(f"已映射图像token数: {len(image_tokens)}, 已映射文本token数: {len(text_tokens)}")
        
        return token_mapping

    def _overlay_attention_to_image(self, ax, token_info, weights, token_mapping, 
                                      cmap_func, alpha, row_idx):
        """将注意力权重叠加到图像上"""
        import numpy as np
        import matplotlib.patches as patches
        
        # 获取像素区域
        top, left, bottom, right = token_info['pixel_region']
        
        # 创建一个与图像大小相同的空白图层
        height, width = bottom - top, right - left
        
        # 为当前token绘制边框
        rect = patches.Rectangle(
            (left, top), width, height, 
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加每个被关注token的热力显示
        for target_idx, weight in enumerate(weights):
            if weight < 0.1:  # 忽略权重太小的
                continue
            
            target_info = token_mapping.get(target_idx)
            if target_info is None:
                continue
            
            # 只处理图像token之间的注意力
            if target_info['type'] == 'image':
                t_top, t_left, t_bottom, t_right = target_info['pixel_region']
                t_height, t_width = t_bottom - t_top, t_right - t_left
                
                # 颜色取决于注意力权重
                color = cmap_func(weight)
                
                # 创建一个半透明矩形
                rect = patches.Rectangle(
                    (t_left, t_top), t_width, t_height, 
                    linewidth=0, facecolor=color, alpha=alpha * weight
                )
                ax.add_patch(rect)
        
        # 在源token上添加索引标记
        ax.text(left + width/2, top + height/2, str(row_idx),
                ha='center', va='center', color='white',
                bbox=dict(facecolor='black', alpha=0.7))

    def _overlay_attention_to_text(self, ax, token_info, weights, token_mapping, 
                                 cmap_func, alpha, row_idx, total_tokens):
        """将注意力权重叠加到文本上"""
        import numpy as np
        
        # 计算该文本token的位置
        position = token_info['position']
        token = token_info['token']
        text_token_count = sum(1 for info in token_mapping.values() if info['type'] == 'text')
        
        # 计算文本布局参数
        y_pos = 0.5  # 垂直位置
        x_start = position / text_token_count
        x_width = 1.0 / text_token_count
        
        # 为当前token绘制一个边框
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (x_start, y_pos - 0.1), x_width, 0.2, 
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加该token文本
        ax.text(x_start + x_width/2, y_pos, token,
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # 添加每个被关注token的热力显示
        for target_idx, weight in enumerate(weights):
            if weight < 0.1:  # 忽略权重太小的
                continue
            
            target_info = token_mapping.get(target_idx)
            if target_info is None:
                continue
            
            # 只处理文本token之间的注意力
            if target_info['type'] == 'text':
                t_position = target_info['position']
                t_x_start = t_position / text_token_count
                t_x_width = 1.0 / text_token_count
                
                # 颜色取决于注意力权重
                color = cmap_func(weight)
                
                # 创建一个半透明矩形
                rect = patches.Rectangle(
                    (t_x_start, y_pos - 0.1), t_x_width, 0.2, 
                    linewidth=0, facecolor=color, alpha=alpha * weight
                )
                ax.add_patch(rect)
        
        # 在源token上添加索引标记
        ax.text(x_start + x_width/2, y_pos + 0.15, str(row_idx),
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1'))

    def split_image_tokens(self, attention_weights, image_size=(448, 448), patch_size=14, merged_patch_count=4):
        """
        根据图像和patch参数，计算图像token在注意力矩阵中的位置
        
        参数:
            attention_weights: 注意力权重矩阵
            image_size: 图像大小(高度,宽度)
            patch_size: 每个patch的大小
            merged_patch_count: 每个token合并的patch数量
            
        返回:
            image_token_indices: 图像token的索引范围
        """
        # 计算原始patch数量
        patch_grid_h = image_size[0] // patch_size
        patch_grid_w = image_size[1] // patch_size
        total_patches = patch_grid_h * patch_grid_w
        
        # 计算合并后的token数量
        image_token_count = total_patches // merged_patch_count
        
        # 假设图像token在序列开始位置
        return (0, image_token_count)


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