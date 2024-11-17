"""
Evaluator module for SegFormer model inference and metric computation.

This module provides functions for model inference and evaluation
on semantic segmentation tasks using SegFormer models.

Functions:
    infer_model: Perform model inference and return loss and logits.
    evaluate_model: Evaluate the model on a dataset shard and compute metrics.

The module uses PyTorch for model inference and the 'evaluate' library
for computing semantic segmentation metrics.
"""

import torch
from evaluate import load

def infer_model(model, pixel_values, labels):  
    """
    Perform model inference and return loss and logits.
    
    Args:
        model (SegformerForSemanticSegmentation): The model to use for inference.
        pixel_values (torch.Tensor): Input pixel values.
        labels (torch.Tensor): Ground truth labels.
    
    Returns:
        tuple: Model loss and logits.
    """
  
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
    return outputs.loss, outputs.logits

def evaluate_model(model, dataset_shard, image_processor, device, metric_name, id2label):
    """
    Evaluate the model on a dataset shard and compute metrics.
    
    Args:
        model (SegformerForSemanticSegmentation): The model to evaluate.
        dataset_shard (Dataset): A shard of the dataset to evaluate on.
        image_processor (SegformerImageProcessor): The image processor to use.
        device (torch.device): The device to run evaluation on.
        metric_name (str): Name of the metric to use.
        id2label (dict): Mapping of label IDs to label names.
    
    Returns:
        dict: Computed evaluation metrics.
    """

    pixel_values, labels = get_processed_inputs(dataset_shard, image_processor, device, model.config.torch_dtype)
    loss, logits = infer_model(model, pixel_values, labels)
    predictions = torch.nn.functional.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear", align_corners=False
    ).argmax(dim=1)
    metric = load(metric_name)
    results = metric.compute(
        predictions=predictions.cpu().numpy(),
        references=labels.cpu().numpy(),
        num_labels=len(id2label),
        ignore_index=model.config.semantic_loss_ignore_index
    )
    return results

