"""
Quantization module for SegFormer models.

This module provides functions for quantizing SegFormer models
using various quantization methods supported by the Quanto library.

Functions:
    quantize_models: Quantize a base SegFormer model using multiple quantization levels.

The module uses Quanto for quantization and supports float8, int8, int4, and int2 quantization.
"""

from transformers import QuantoConfig, SegformerForSemanticSegmentation
import quanto

def quantize_models(base_model, model_name, model_save_path, device):
    """
    Quantize the base model using various quantization methods.
    
    Args:
        base_model (SegformerForSemanticSegmentation): The base model to quantize.
        model_name (str): Name of the model.
        model_save_path (str): Path to save quantized models.
        device (torch.device): Device to load models to.
    
    Returns:
        dict: Dictionary of quantized models.
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
        models[model_htype] = models[model_htype].to(device)
    return models

