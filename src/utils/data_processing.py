"""
Data processing module for SegFormer evaluation pipeline.

This module contains functions for loading, preprocessing, and handling
dataset operations for semantic segmentation tasks using SegFormer models.

Functions:
    load_dataset: Load or download and save the dataset.
    get_processed_inputs: Process and prepare inputs for model inference.
    convert_to_RGB: Convert dataset images and annotations to RGB and grayscale respectively.

The module uses the Hugging Face datasets library and image processing
tools to prepare data for SegFormer model evaluation.
"""

from datasets import load_dataset, load_from_disk, Dataset
from datasets.utils.logging import set_verbosity_error
from transformers import SegformerImageProcessor
from typing import List, Dict, Tuple
from torch import dtype, device, Tensor, half, float16

def load_dataset_custom(
    dataset_save_path: str,
    dataset_name: str
):
    """
    Load or download and save the dataset.
    
    Args:
        dataset_save_path (str): Path to save/load the dataset.
        dataset_name (str): Name of the dataset to load.
    
    Returns:
        Dataset: The loaded dataset.
    """
    
    try:
        return load_from_disk(dataset_save_path)
    except Exception:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        dataset.save_to_disk(dataset_save_path)
        return dataset

def get_processed_inputs(
    dataset: Dataset,
    image_processor: SegformerImageProcessor,
    device: device,
    bias_dtype: dtype = None
) -> Tuple[Tensor, Tensor]:
    """
    Process and prepare inputs for model inference.
    
    Args:
        dataset (Dataset): The dataset to process.
        image_processor (SegformerImageProcessor): The image processor to use.
        device (torch.device): The device to load tensors to.
        bias_dtype (torch.dtype, optional): Dtype for bias, if any.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Processed pixel values and labels.
    """
    
    set_verbosity_error()
    dataset = dataset.map(convert_to_RGB, batched=True)
    dataset = image_processor.preprocess(
        images=dataset['image'],
        segmentation_maps=dataset['annotation'],
        return_tensors="pt",
    )
    pixel_values = dataset['pixel_values'].to(device)
    labels = dataset['labels'].to(device)
    if bias_dtype in [half, float16]:
        pixel_values = pixel_values.half()
    return pixel_values, labels

def convert_to_RGB(
    dataset: Dict[List, List]
) -> Dict[List, List]:
    """
    Convert dataset images and annotations to RGB and grayscale respectively.
    
    Args:
        dataset (Dict): Dataset containing 'image' and 'annotation' keys.
    
    Returns:
        Dict[List, List]: Processed dataset with converted images and annotations.
    """

    images = [img.convert("RGB") for img in dataset['image']]
    annotations = [img.convert("L") for img in dataset['annotation']]
    return {'image': images, 'annotation': annotations}