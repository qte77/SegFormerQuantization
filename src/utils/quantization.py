"""
Quantization module for SegFormer models.

This module provides functions for quantizing SegFormer models
using various quantization methods supported by the Quanto library.

Functions:
    quantize_models: Quantize a base SegFormer model using multiple quantization levels.

The module uses Quanto for quantization and supports float8, int8, int4, and int2 quantization.
"""

from transformers import QuantoConfig, SegformerForSemanticSegmentation
from torch import device
from typing import Dict
import quanto

def quantize_models(
    base_model: SegformerForSemanticSegmentation,
    model_name: str,
    model_save_path: str,
    torch_device: device
) -> Dict[str, SegformerForSemanticSegmentation]:
    """
    Quantize the base model using various quantization methods.
    
    Args:
        base_model (SegformerForSemanticSegmentation): The base model to quantize.
        model_name (str): Name of the model.
        model_save_path (str): Path to save quantized models.
        torch_device (torch.device): Device to load models to.
    
    Returns:
        Dict[str, SegformerForSemanticSegmentation]: Dictionary of quantized models.
    """

    models = {}
    config_quanto = {}
    bits_quanto_w = ['float8', 'int8', 'int4', 'int2']
    for nbits in bits_quanto_w:
        config_quanto[nbits] = QuantoConfig(weights=nbits)
    
    for nbits in bits_quanto_w:
        model_htype = f"{model_name}-quanto-{nbits}"
        model_save_path_quanto = f"{model_save_path}/{model_name}-{model_htype}"
        try:
            print(f"loading local {model_htype}")
            models[model_htype] = SegformerForSemanticSegmentation.from_pretrained(model_save_path_quanto)
        except Exception:
            try:
                print(f"loading local {model_name}")
                models[model_htype] = SegformerForSemanticSegmentation.from_pretrained(
                    model_save_path,
                    local_files_only=True,
                    torch_dtype=base_model.config.torch_dtype,
                    quantization_config=config_quanto[nbits],
                )
            except Exception:
                print(f"loading online {model_name}")
                models[model_htype] = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    torch_dtype=base_model.config.torch_dtype,
                    quantization_config=config_quanto[nbits],
                )
        quanto.freeze(models[model_htype])
        models[model_htype] = models[model_htype].to(torch_device)
    return models

