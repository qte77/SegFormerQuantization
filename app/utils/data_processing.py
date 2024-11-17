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

from datasets import load_dataset, load_from_disk
from datasets.utils.logging import set_verbosity_error
import torch

def load_dataset_custom(dataset_save_path, dataset_name):
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
    except Exception as e:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        dataset.save_to_disk(dataset_save_path)
        return dataset

def get_processed_inputs(dataset, image_processor, device, bias_dtype=None):
    """
    Process and prepare inputs for model inference.
    
    Args:
        dataset (Dataset): The dataset to process.
        image_processor (ImageProcessor): The image processor to use.
        device (torch.device): The device to load tensors to.
        bias_dtype (torch.dtype, optional): Dtype for bias, if any.
    
    Returns:
        tuple: Processed pixel values and labels.
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
    if bias_dtype in [torch.half, torch.float16]:
        pixel_values = pixel_values.half()
    return pixel_values, labels

def convert_to_RGB(dataset):
    """
    Convert dataset images and annotations to RGB and grayscale respectively.
    
    Args:
        dataset (dict): Dataset containing 'image' and 'annotation' keys.
    
    Returns:
        dict: Processed dataset with converted images and annotations.
    """

    images = [img.convert("RGB") for img in dataset['image']]
    annotations = [img.convert("L") for img in dataset['annotation']]
    return {'image': images, 'annotation': annotations}

