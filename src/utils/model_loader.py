"""
Model loading module for SegFormer evaluation pipeline.

This module provides functions for loading and initializing SegFormer models
and image processors, with support for local and remote loading.

Functions:
    load_base_model: Load or download and save the base SegFormer model.
    load_image_processor: Load or download and save the image processor.

The module uses the transformers library for model and processor management.
"""

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch import dtype, device
from os.path import exists
from os import makedirs

def load_base_model(
    model_name: str,
    model_save_path: str,
    compute_dtype: dtype, 
    device: device
) -> SegformerForSemanticSegmentation:
    """
    Load or download and save the base SegFormer model.
    
    Args:
        model_name (str): Name of the model to load.
        model_save_path (str): Path to save/load the model.
        compute_dtype (torch.dtype): Computation dtype for the model.
        device (torch.device): Device to load the model to.
    
    Returns:
        SegformerForSemanticSegmentation: The loaded model.
    """

    try:
        print("loading from disk")
        model = SegformerForSemanticSegmentation.from_pretrained(model_save_path)
    except Exception as e:
        print(
            f"Error loading local model: {e}."
            f"Loading from source and saving to disk."
        )
        if not exists(model_save_path):
            makedirs(model_save_path, exist_ok=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
        )
        model.save_pretrained(model_save_path)
    return model.to(device)

def load_image_processor(
    model_name: str,
    tokenizer_save_path: str
) -> SegformerImageProcessor:
    """
    Load or download and save the image processor.
    
    Args:
        model_name (str): Name of the model to load the processor for.
        tokenizer_save_path (str): Path to save/load the processor.
    
    Returns:
        SegformerImageProcessor: The loaded image processor.
    """

    try:
        return SegformerImageProcessor.from_pretrained(tokenizer_save_path)
    except Exception as e:
        print(
            f"Error loading local ImageProcessor: {e}."
            f"Loading from source and saving to disk."
        )
        image_processor = SegformerImageProcessor.from_pretrained(model_name)
        image_processor.save_pretrained(tokenizer_save_path)
        return image_processor