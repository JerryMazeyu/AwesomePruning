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
import matplotlib.patches as patches

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
        Visualize attention weights distribution in the model
        
        Args:
            raw_inputs (Union[str, dict]): Raw input data, can be text string or dictionary containing image and text
            attention_representation (torch.Tensor): Attention weights, shape [1, head_num, token_num, token_num] or [1, head_num, 1, token_num+i]
            **kwargs: Other configuration parameters, including:
                head_indices: Attention head indices to visualize, default is None (use average of all heads)
                tokenizer: Tokenizer for mapping token IDs back to text
                is_multimodal: Whether the input is multimodal, default is False
                img_placeholder: Placeholder string used in text for image insertion (e.g. "<ImageHere>"), default is None
                image_size: Input image size, default is (448, 448)
                patch_size: Image patch size, default is 14
                merged_patch_count: Number of patches merged for each token, default is 4
                cmap: Colormap for heatmap, default is 'viridis'
                alpha: Transparency of heatmap, default is 0.7
                output_path: Path to save visualization results, default is None
                figsize: Figure size, default is None
                title: Title of the figure, default is None
                split_attention: Whether to split attention matrix, default is False (only applicable to shape [1, head_num, token_num, token_num])
        
        Returns:
            matplotlib.figure.Figure: Figure object of visualization results
        """
        
        # Default configuration, can be overridden by kwargs
        config = {
            'head_indices': None,
            'tokenizer': None,
            'is_multimodal': False,
            'img_placeholder': None,  # 替换掉image_first，改为img_placeholder
            'image_size': (448, 448),
            'patch_size': 14,
            'merged_patch_count': 4,
            'cmap': 'viridis',
            'alpha': 0.7,
            'output_path': None,
            'figsize': None,
            'title': None,
            'split_attention': False
        }
        
        # Update configuration with passed parameters
        config.update(kwargs)
        
        # Ensure batch size is 1
        if attention_representation.shape[0] != 1:
            raise ValueError("Batch size must be 1 for attention visualization")
        
        # Move attention representation to CPU
        attention_representation = attention_representation.cpu()
        
        # Check if attention matrix can be split (must be shape [1, head_num, token_num, token_num])
        is_splittable = config['split_attention'] and len(attention_representation.shape) == 4 and attention_representation.shape[2] == attention_representation.shape[3]
        
        if config['split_attention'] and not is_splittable:
            print(f"Warning: Cannot split attention matrix, current shape is {attention_representation.shape}, need shape [1, head_num, token_num, token_num]")
            
        # Calculate attention weights
        if config['head_indices'] is not None:
            if isinstance(config['head_indices'], int):
                config['head_indices'] = [config['head_indices']]
            attention_weights = attention_representation[0, config['head_indices']].mean(dim=0).detach().numpy()
        else:
            attention_weights = attention_representation[0].mean(dim=0).detach().numpy()
        
        # Set figure size
        if config['figsize'] is None:
            if config['is_multimodal']:
                cols = 4 if is_splittable else 2
                config['figsize'] = (9 * cols, 10)
            else:
                config['figsize'] = (12, 8)
        
        # Create figure
        fig = plt.figure(figsize=config['figsize'])
        
        if not config['is_multimodal']:
            # Text input visualization
            self._visualize_text_attention(raw_inputs, attention_weights, config['tokenizer'], 
                                          config['cmap'], config['alpha'], fig, config['title'])
        else:
            # Use multimodal visualization with img_placeholder
            self._visualize_multimodal_attention(raw_inputs, attention_weights, config['tokenizer'], 
                                               config['img_placeholder'], config['image_size'], 
                                               config['patch_size'], config['merged_patch_count'], 
                                               config['cmap'], config['alpha'], fig, config['title'],
                                               split_attention=is_splittable)
        
        # Save image
        if config['output_path']:
            plt.savefig(config['output_path'], bbox_inches='tight', dpi=300)
            print(f"Attention visualization saved to: {config['output_path']}")
            plt.close(fig)  # Close figure to release resources
        
        return fig

    def _create_attention_subplot(self, image, text, token_mapping, attention_values, 
                                  cmap_func, alpha, subplot_title=None, subplot_ax=None):
        """
        Create a subplot with image and text, optionally with attention heatmap
        
        Args:
            image (np.ndarray): Image data
            text (str): Text data
            token_mapping (dict): Mapping from token indices to content
            attention_values (np.ndarray, optional): Attention values for tokens, length token_num
            cmap_func (Callable): Colormap function
            alpha (float): Transparency of heatmap
            subplot_title (str, optional): Title of the subplot
            subplot_ax (matplotlib.axes.Axes, optional): Parent axes for the subplot, will create two subaxes
            
        Returns:
            tuple: Tuple of (image_axis, text_axis) - the created subplot axes
        """
        # Check if subplot_ax is None or not a valid Axes object
        if subplot_ax is None or not hasattr(subplot_ax, 'figure') or subplot_ax.figure is None:
            fig = plt.figure(figsize=(4.5, 5))
            subplot_ax = fig.add_subplot(111)
            standalone = True
        else:
            standalone = False
        
        
        # Create subgrid - divide the main coordinate area into two parts
        import matplotlib.gridspec as gridspec
        
        if hasattr(subplot_ax, 'get_position'):
            # Get the position of the current axis in the figure
            position = subplot_ax.get_position()
            
            # Clear the current axis and hide it
            subplot_ax.set_axis_off()
            
            # Manually create two sub-axes
            fig = subplot_ax.figure
            
            # Create the upper axis (image)
            ax_img = fig.add_axes([position.x0, position.y0 + position.height * 0.25, 
                                   position.width, position.height * 0.75])
            ax_img.set_axis_off()
            
            # Create the lower axis (text)
            ax_text = fig.add_axes([position.x0, position.y0, 
                                    position.width, position.height * 0.25])
            ax_text.set_axis_off()
        else:
            # If subplot_ax does not have the get_position method, create a new layout
            fig = subplot_ax.figure
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
            ax_img = fig.add_subplot(gs[0])
            ax_text = fig.add_subplot(gs[1])
            ax_img.set_axis_off()
            ax_text.set_axis_off()
        
        # Process image
        if image is not None:
            h, w = image.shape[:2]
            
            # Display the original image
            ax_img.imshow(image)
            
            # If attention values are provided, create attention heatmap
            if attention_values is not None:
                # Create an empty attention image
                attention_img = np.zeros((h, w, 4))  # RGBA image
                
                # Fill attention image based on token mapping
                for token_idx, token_info in token_mapping.items():
                    if token_info['type'] == 'image':
                        # Get pixel region
                        top, left, bottom, right = token_info['pixel_region']
                        
                        # Get attention value for this token
                        if token_idx < len(attention_values):
                            attention_value = attention_values[token_idx]
                            
                            # Get color based on attention value
                            color = cmap_func(attention_value)
                            
                            # Ensure region size is within image bounds
                            top = min(top, h-1)
                            bottom = min(bottom, h)
                            left = min(left, w-1)
                            right = min(right, w)
                            
                            if top < bottom and left < right:
                                # Fill corresponding region
                                attention_img[top:bottom, left:right, :] = color
            
                # Display heatmap over the image
                ax_img.imshow(attention_img, alpha=alpha)  # Show attention heatmap
        
        # Set title for image area
        if subplot_title:
            ax_img.set_title(subplot_title)
        
        # Display text - 始终显示原始完整文本
        if text is not None:
            if isinstance(text, list):
                text = text[0]  # Take the first text
            ax_text.text(0.5, 0.9, text, 
                      ha='center', va='top', fontsize=10,
                      wrap=True)
        
        # If attention values are provided, visualize text attention
        if attention_values is not None and text is not None:
            # Get text tokens
            text_tokens = [info for idx, info in token_mapping.items() if info['type'] == 'text']
            text_token_indices = [idx for idx, info in token_mapping.items() if info['type'] == 'text']
            
            if text_tokens:
                text_token_count = len(text_tokens)
                
                # 计算最大的文本position值，用于正确缩放位置
                max_position = max(info['position'] for info in text_tokens)
                
                # Normalize text attention values
                text_attention_values = [attention_values[idx] if idx < len(attention_values) else 0 
                                        for idx in text_token_indices]
                
                if text_attention_values:
                    text_min = min(text_attention_values)
                    text_max = max(text_attention_values)
                    
                    if text_max > text_min:
                        # Normalize text token attention values more strongly
                        text_normalized = {idx: ((attention_values[idx] - text_min) / (text_max - text_min + 1e-8)) ** 0.5
                                          for idx in text_token_indices if idx < len(attention_values)}
                        
                        # Apply stronger contrast enhancement
                        for idx in text_normalized:
                            text_normalized[idx] = 0.5 + (text_normalized[idx] - 0.5) * 1.5
                            text_normalized[idx] = max(0, min(1, text_normalized[idx]))
                    else:
                        text_normalized = {idx: 0.5 for idx in text_token_indices if idx < len(attention_values)}
                    
                    # Visualize text token attention
                    for token_idx, token_info in token_mapping.items():
                        if token_info['type'] == 'text':
                            # Get token position and text
                            position = token_info['position']
                            token = token_info['token']
                            
                            # 计算在文本区域中的显示位置（使用全局位置，确保连续）
                            x_start = position / (max_position + 1)
                            x_width = 1.0 / (max_position + 1)
                            
                            # Use text-specific normalized attention value
                            if token_idx in text_normalized:
                                attention_value = text_normalized[token_idx]
                                
                                # Get color
                                color = cmap_func(attention_value)
                                
                                # Create background rectangle
                                rect = patches.Rectangle(
                                    (x_start, 0.3), x_width, 0.4,
                                    linewidth=0, facecolor=color, alpha=min(alpha + 0.3, 1.0)
                                )
                                ax_text.add_patch(rect)
                                
                                # Add token text
                                ax_text.text(x_start + x_width/2, 0.5, token,
                                        ha='center', va='center', fontsize=10,
                                        rotation=90,  # 添加90度旋转
                                        color='black' if attention_value < 0.5 else 'white')
        
        # Set title for text area
        # ax_text.set_title("Text")
        
        # 如果是独立创建的figure，则应用tight_layout
        if standalone:
            plt.tight_layout()
        
        return ax_img, ax_text

    def _visualize_multimodal_attention(self, inputs, attention_weights, tokenizer, 
                                        img_placeholder, image_size, patch_size, merged_patch_count,
                                        cmap, alpha, fig, title=None, split_attention=False):
        """
        Visualize attention weights for multimodal input
        
        Args:
            inputs: Input data, can be dictionary or tuple of image and text
            attention_weights: Attention weights, shape [token_num, token_num]
            tokenizer: Tokenizer for mapping token IDs back to text
            img_placeholder: Placeholder string used in text for image insertion (e.g. "<ImageHere>"), default is None
            image_size: Input image size
            patch_size: Image patch size
            merged_patch_count: Number of patches merged for each token
            cmap: Colormap for heatmap
            alpha: Transparency of heatmap
            fig: Matplotlib figure object
            title: Title of the figure, default is None
            split_attention: Whether to split attention matrix, default is False
        """
        
        # Parse multimodal input
        if isinstance(inputs, dict):
            image = inputs.get('image') or inputs.get('images')
            text = inputs.get('text') or inputs.get('texts')
        else:
            # Assume it's a tuple or list
            image, text = inputs if len(inputs) >= 2 else (inputs[0], None)
        
        # Get number of tokens
        total_tokens = attention_weights.shape[1]
        
        # Calculate number of image tokens
        patch_grid_size = int(image_size[0] / patch_size)
        original_patches = patch_grid_size * patch_grid_size  # e.g., 448x448 and 14x14 patch gives 32x32=1024 patches
        merged_token_count = original_patches // merged_patch_count  # e.g., merging to 256 tokens
        
        # Map tokens to content
        token_mapping = self._map_tokens_to_content(
            image, text, tokenizer, img_placeholder, 
            image_size, patch_size, merged_patch_count, 
            merged_token_count, total_tokens
        )
        
        # Process image data
        if isinstance(image, torch.Tensor):
            # Move to CPU and convert to numpy
            image_np = image.cpu().numpy()
            # Process batch dimension
            if len(image_np.shape) == 4:  # [batch, channel, height, width]
                image_np = image_np.squeeze(0)  # Remove batch dimension
            # Convert channel order from [C,H,W] to [H,W,C]
            if image_np.shape[0] == 3 and len(image_np.shape) == 3:  # If format is [C,H,W]
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert to [H,W,C] format
            elif isinstance(image, np.ndarray):
                # Process numpy array
                if len(image.shape) == 4:  # [batch, channel, height, width]
                    image_np = image.squeeze(0)
                # Convert channel order from [C,H,W] to [H,W,C]
                if image_np.shape[0] == 3 and len(image_np.shape) == 3:  # If format is [C,H,W]
                    image_np = np.transpose(image_np, (1, 2, 0))  # Convert to [H,W,C] format
                else:
                    image_np = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        # Check if image shape meets matplotlib requirements
        assert len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] in [1, 3, 4]), \
            f"Image shape does not meet matplotlib requirements: {image_np.shape}"

        # Ensure values are in reasonable range
        if np.issubdtype(image_np.dtype, np.floating):
            if image_np.max() > 1.0:
                image_np = image_np / 255.0  # If values exceed 1, assume 0-255 range, convert to 0-1
            
            # Ensure image is not too dark
            if image_np.max() < 0.1:  # If image is almost all black
                print("Warning: Image may be too dark, trying to enhance contrast")
                # Enhance contrast
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Set up color map
        cmap_func = plt.cm.get_cmap(cmap)
        
        # Calculate mean attention
        mean_attention = np.mean(attention_weights, axis=0)  # Average over all tokens
        
        # Normalize attention values to ensure sufficient contrast
        if mean_attention.max() > mean_attention.min():
            # Stronger contrast stretch
            normalized_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
            # Apply contrast enhancement - use power function to enhance contrast
            normalized_attention = np.power(normalized_attention, 0.7)  # 0.7 is an adjustable parameter, less than 1 increases contrast
        else:
            normalized_attention = np.zeros_like(mean_attention)
        
        # Identify image and text tokens - 现在使用token_mapping中的类型来分类
        image_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'image']
        text_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'text']
        
        print(f"Image tokens: {len(image_tokens)}, Text tokens: {len(text_tokens)}")
        
        # 设置网格布局 - 1行n列
        cols = 4 if split_attention else 2
        fig.clf()  # 清空图形
        
        # 创建网格布局
        import matplotlib.gridspec as gridspec
        grid = gridspec.GridSpec(1, cols, figure=fig, wspace=0.3, hspace=0.3)
        
        # If title is provided, set it
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        
        # 1. 原始内容子图
        ax1 = fig.add_subplot(grid[0])
        self._create_attention_subplot(
            image_np, text, token_mapping, None, 
            cmap_func, alpha, "Original Content", ax1
        )
        
        # 2. 完整注意力子图
        ax2 = fig.add_subplot(grid[1])
        self._create_attention_subplot(
            image_np, text, token_mapping, normalized_attention, 
            cmap_func, alpha, "Complete Attention", ax2
        )
        
        # 如果需要分割注意力，计算self-attention和cross-attention
        if split_attention:
            # 计算self-attention和cross-attention
            if image_tokens and text_tokens:
                # 将token索引排序
                img_indices = sorted(image_tokens)
                txt_indices = sorted(text_tokens)
                
                # 检查attention_weights的维度情况
                if len(attention_weights.shape) == 1:
                    print(f"注意：attention_weights是一维数组，形状为{attention_weights.shape}，无法进行矩阵分块")
                    # 对于一维数组，直接基于索引提取对应的值
                    self_attention = np.zeros_like(normalized_attention)
                    cross_attention = np.zeros_like(normalized_attention)
                    
                    # 根据索引提取对应区域的值
                    for idx in img_indices:
                        if idx < len(attention_weights):
                            self_attention[idx] = attention_weights[idx]
                    
                    for idx in txt_indices:
                        if idx < len(attention_weights):
                            cross_attention[idx] = attention_weights[idx]
                else:
                    # 创建子块矩阵 - 使用np.ix_进行索引
                    try:
                        # 尝试创建子矩阵块
                        img_img_block = attention_weights[np.ix_(img_indices, img_indices)]  # 图像到图像
                        img_txt_block = attention_weights[np.ix_(img_indices, txt_indices)]  # 图像到文本
                        txt_img_block = attention_weights[np.ix_(txt_indices, img_indices)]  # 文本到图像
                        txt_txt_block = attention_weights[np.ix_(txt_indices, txt_indices)]  # 文本到文本
                        
                        print(f"子矩阵块大小: img_img={img_img_block.shape}, img_txt={img_txt_block.shape}, txt_img={txt_img_block.shape}, txt_txt={txt_txt_block.shape}")
                    except Exception as e:
                        print(f"创建子矩阵块时出错: {e}")
                        # 发生错误时使用零矩阵
                        img_img_block = np.zeros((len(img_indices), len(img_indices)))
                        img_txt_block = np.zeros((len(img_indices), len(txt_indices)))
                        txt_img_block = np.zeros((len(txt_indices), len(img_indices)))
                        txt_txt_block = np.zeros((len(txt_indices), len(txt_indices)))
                    
                    # 计算self-attention: 对两个对角块矩阵行平均，然后分配到相应位置
                    self_attention = np.zeros_like(normalized_attention)
                    
                    # 图像部分的self-attention - 对img_img_block按行平均
                    img_self_attn = np.mean(img_img_block, axis=1)  # 行平均
                    for i, idx in enumerate(img_indices):
                        if idx < len(self_attention):
                            self_attention[idx] = img_self_attn[i]
                    
                    # 文本部分的self-attention - 对txt_txt_block按行平均
                    txt_self_attn = np.mean(txt_txt_block, axis=1)  # 行平均
                    for i, idx in enumerate(txt_indices):
                        if idx < len(self_attention):
                            self_attention[idx] = txt_self_attn[i]
                    
                    # 计算cross-attention: 分别处理图像和文本部分
                    cross_attention = np.zeros_like(normalized_attention)
                    
                    # 图像部分的cross-attention
                    # 图像-文本块的行平均 (每个图像token对所有文本token的平均关注度)
                    if img_txt_block.size > 0:
                        img_to_txt_attn = np.mean(img_txt_block, axis=1)  # 行平均
                        for i, idx in enumerate(img_indices):
                            if idx < len(cross_attention) and i < len(img_to_txt_attn):
                                cross_attention[idx] = img_to_txt_attn[i]
                    
                    # 文本部分的cross-attention
                    # 文本-图像块的行平均 (每个文本token对所有图像token的平均关注度)
                    if txt_img_block.size > 0:
                        txt_to_img_attn = np.mean(txt_img_block, axis=1)  # 行平均
                        for i, idx in enumerate(txt_indices):
                            if idx < len(cross_attention) and i < len(txt_to_img_attn):
                                cross_attention[idx] = txt_to_img_attn[i]
                
                # 归一化self-attention
                if self_attention.max() > self_attention.min():
                    self_attention = (self_attention - self_attention.min()) / (self_attention.max() - self_attention.min() + 1e-8)
                    self_attention = np.power(self_attention, 0.7)
                
                # 归一化cross-attention
                if cross_attention.max() > cross_attention.min():
                    cross_attention = (cross_attention - cross_attention.min()) / (cross_attention.max() - cross_attention.min() + 1e-8)
                    cross_attention = np.power(cross_attention, 0.7)
            else:
                # 如果只有图像或只有文本，则自注意力就是完整注意力，交叉注意力为0
                self_attention = normalized_attention.copy()
                cross_attention = np.zeros_like(normalized_attention)
            
            # 3. Self-Attention子图
            ax3 = fig.add_subplot(grid[2])
            self._create_attention_subplot(
                image_np, text, token_mapping, self_attention, 
                cmap_func, alpha, "Self-Attention", ax3
            )
            
            # 4. Cross-Attention子图
            ax4 = fig.add_subplot(grid[3])
            self._create_attention_subplot(
                image_np, text, token_mapping, cross_attention, 
                cmap_func, alpha, "Cross-Attention", ax4
            )
        
        # 应用紧凑布局
        fig.tight_layout(rect=[0, 0, 1, 0.95 if title else 1])  # 为标题留出空间
        
        return token_mapping

    def _visualize_text_attention(self, text_input, attention_weights, tokenizer, cmap, alpha, fig, title=None):
        """Visualize the attention weights of text input"""
        
        # Get token and text mapping
        if isinstance(text_input, str) and tokenizer is not None:
            tokens = tokenizer.tokenize(text_input)
        elif hasattr(text_input, 'tokens'):
            tokens = text_input.tokens
        else:
            # Assume input is already tokenized list
            tokens = text_input if isinstance(text_input, list) else ["Token_" + str(i) for i in range(attention_weights.shape[0])]
        
        # Ensure token count matches attention matrix dimension
        token_count = min(len(tokens), attention_weights.shape[0])
        
        # Create subplot grid
        rows = token_count
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create a text strip for each token
        y_positions = np.linspace(0, 1, rows + 2)[1:-1]  # Avoid edge
        
        # Display text for each token
        for i, token in enumerate(tokens[:token_count]):
            # Calculate the weight of this token on other tokens
            attn_row = attention_weights[i, :token_count]
            max_attn = attn_row.max()
            norm_attn = attn_row / max_attn if max_attn > 0 else attn_row
            
            # Display text with attention color background at each token position
            for j, (t, weight) in enumerate(zip(tokens[:token_count], norm_attn)):
                x_start = j / token_count
                x_end = (j + 1) / token_count
                color_map = plt.cm.get_cmap(cmap)
                color = color_map(weight)
                
                # Draw background rectangle
                rect = plt.Rectangle((x_start, y_positions[i] - 0.03), 
                                   (x_end - x_start), 0.06, 
                                   facecolor=color, alpha=alpha * weight)
                ax.add_patch(rect)
                
                # Add text
                ax.text((x_start + x_end) / 2, y_positions[i], t, 
                       ha='center', va='center', fontsize=12, 
                       color='black' if weight < 0.6 else 'white')
        
        # Set subplot title
        ax.set_title("Attention Visualization for Each Token")
        
        # Set total title (if provided)
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _map_tokens_to_content(self, image, text, tokenizer, img_placeholder, 
                              image_size, patch_size, merged_patch_count, 
                              image_token_count, total_tokens):
        """Map each token to the corresponding image region or text"""
        token_mapping = {}
        patch_grid_size = int(image_size[0] / patch_size)  # For example, 448/14=32
        tokens_per_row = patch_grid_size // (merged_patch_count if merged_patch_count > 1 else 1)
        
        print(f"Total token count: {total_tokens}, Image token count: {image_token_count}")
        print(f"Patch grid size: {patch_grid_size}x{patch_grid_size}, Tokens per row: {tokens_per_row}")
        print(f"Merge {merged_patch_count} patches into 1 token")
        
        # Decide the processing method based on whether the image placeholder is provided
        if img_placeholder is None:
            # Traditional method: image first or text first
            # Image tokens first
            for token_idx in range(image_token_count): 
                # Calculate the position of the token in the grid
                token_row = token_idx // tokens_per_row
                token_col = token_idx % tokens_per_row
                
                # Calculate the starting index of the original patch - each token contains a patch height and merged_patch_count patch widths
                start_patch_row = token_row
                start_patch_col = token_col * merged_patch_count
                
                # Calculate the pixel region
                top = start_patch_row * patch_size
                left = start_patch_col * patch_size
                width = merged_patch_count * patch_size
                height = patch_size
                
                # Ensure not exceeding image boundaries
                bottom = min(top + height, image_size[0])
                right = min(left + width, image_size[1])
                
                print(f"Token {token_idx}: Position ({token_row},{token_col}), Region ({top},{left},{bottom},{right})")
                
                token_mapping[token_idx] = {
                    'type': 'image',
                    'position': (token_row, token_col),
                    'pixel_region': (top, left, bottom, right),
                    'patches': [(start_patch_row, start_patch_col + i) 
                               for i in range(merged_patch_count)]
                }
            
            # Text token mapping
            if tokenizer is not None and text is not None:
                # Use tokenizer to process text
                if hasattr(tokenizer, 'tokenize'):
                    text_to_tokenize = text[0] if isinstance(text, list) else text
                    tokens = tokenizer.tokenize(text_to_tokenize)
                else:
                    # Some tokenizers do not have the tokenize method
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    tokens = [tokenizer.decode([id]) for id in encoded]
                
                print(f"Text: {text}")
                print(f"Tokenization result: {tokens}")
                
                # Text tokens start from image_token_count
                for i, token in enumerate(tokens):
                    text_token_idx = image_token_count + i
                    if text_token_idx < total_tokens:
                        token_mapping[text_token_idx] = {
                            'type': 'text',
                            'token': token,
                            'position': i,  # 文本token位置，连续编号
                            'original_text': text
                        }
        else:
            # New method: image insertion based on placeholder
            if tokenizer is None or text is None:
                raise ValueError("Tokenizer and text are required when using img_placeholder")
                
            # Prepare image token data
            def create_image_token_data(start_idx):
                """Create mapping data for each image token"""
                image_tokens = {}
                for token_offset in range(image_token_count):
                    token_idx = start_idx + token_offset
                    
                    # Calculate the position in the image grid
                    token_row = token_offset // tokens_per_row
                    token_col = token_offset % tokens_per_row
                    
                    # Calculate the starting index of the original patch
                    start_patch_row = token_row
                    start_patch_col = token_col * merged_patch_count
                    
                    # Calculate the pixel region
                    top = start_patch_row * patch_size
                    left = start_patch_col * patch_size
                    width = merged_patch_count * patch_size
                    height = patch_size
                    
                    # Ensure not exceeding image boundaries
                    bottom = min(top + height, image_size[0])
                    right = min(left + width, image_size[1])
                    
                    image_tokens[token_idx] = {
                        'type': 'image',
                        'position': (token_row, token_col),
                        'pixel_region': (top, left, bottom, right),
                        'patches': [(start_patch_row, start_patch_col + i) 
                                  for i in range(merged_patch_count)]
                    }
                return image_tokens
            
            # Process text segmentation and image insertion
            text_to_tokenize = text[0] if isinstance(text, list) else text
            
            if img_placeholder in text_to_tokenize:
                # Use placeholder to split text
                text_segments = text_to_tokenize.split(img_placeholder)
                print(f"Text split into {len(text_segments)} segments using placeholder '{img_placeholder}'")
                
                # Get tokens for each text segment
                segment_tokens = []
                for i, segment in enumerate(text_segments):
                    if hasattr(tokenizer, 'tokenize'):
                        tokens = tokenizer.tokenize(segment)
                    else:
                        encoded = tokenizer.encode(segment, add_special_tokens=(i==0))
                        tokens = [tokenizer.decode([id]) for id in encoded]
                    segment_tokens.append(tokens)
                
                # Create token mapping
                token_idx = 0
                text_position = 0  # Track the overall position of all text tokens
                
                # First text segment
                for i, token in enumerate(segment_tokens[0]):
                    token_mapping[token_idx] = {
                        'type': 'text',
                        'token': token,
                        'position': text_position,  # Use continuous global position
                        'segment': 0,  # Record the segment
                        'original_text': text_segments[0]
                    }
                    token_idx += 1
                    text_position += 1  # Text position increases
                
                # Process subsequent image and text segments
                for seg_idx in range(1, len(text_segments)):
                    # Insert image tokens
                    image_tokens = create_image_token_data(token_idx)
                    token_mapping.update(image_tokens)
                    token_idx += image_token_count
                    
                    # Add subsequent text segment
                    seg_tokens = segment_tokens[seg_idx]
                    for i, token in enumerate(seg_tokens):
                        if token_idx < total_tokens:
                            token_mapping[token_idx] = {
                                'type': 'text',
                                'token': token,
                                'position': text_position,  # Use continuous global position
                                'segment': seg_idx,  # Record the segment
                                'original_text': text_segments[seg_idx]
                            }
                            token_idx += 1
                            text_position += 1  # Text position increases
            else:
                # If the text does not contain the placeholder, use the default method to process
                print(f"Warning: img_placeholder '{img_placeholder}' not found in text, using default token mapping")
                
                # Process all text tokens first
                if hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(text_to_tokenize)
                else:
                    encoded = tokenizer.encode(text_to_tokenize, add_special_tokens=False)
                    tokens = [tokenizer.decode([id]) for id in encoded]
                
                # Create text token mapping first
                text_position = 0
                for i, token in enumerate(tokens):
                    if i < total_tokens - image_token_count:
                        token_mapping[i] = {
                            'type': 'text',
                            'token': token,
                            'position': text_position,
                            'original_text': text_to_tokenize
                        }
                        text_position += 1
                
                # Then add image tokens
                image_tokens = create_image_token_data(len(tokens))
                token_mapping.update(image_tokens)
        
        # Print token count statistics
        image_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'image']
        text_tokens = [idx for idx, info in token_mapping.items() if info['type'] == 'text']
        print(f"Total mapped image tokens: {len(image_tokens)}, Total mapped text tokens: {len(text_tokens)}")
        
        return token_mapping


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