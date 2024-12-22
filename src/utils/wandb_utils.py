"""
Weights & Biases utility module for SegFormer evaluation pipeline.

Provides functions for managing W&B runs, including initialization,
metadata setting, and result logging for the SegFormer evaluation pipeline.

Functions:
    create_wandb_run: Initialize a new W&B run.
    create_wandb_run_meta: Set metadata for a W&B run.
    log_wandb_results: Log evaluation results to W&B.

Note: Requires WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY environment variables.
"""

import wandb
from datetime import datetime

def create_wandb_run(project, entity, name, group):
    """
    Initialize and create a new Weights & Biases run.
    
    Args:
        project (str): Name of the W&B project.
        entity (str): Name of the W&B entity (team or user).
        name (str): Name of the run.
        group (str): Group name for the run.
    
    Returns:
        wandb.Run: The created W&B run object.
    """

    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group
    )
    assert wandb_run is wandb.run
    return wandb_run

def create_wandb_run_meta(
    wandb_run, model_name, dataset_name, device, wandb_tag_mode,
    quant_used, model_used, ds_num_shards, ds_shards_mod
):
    """
    Set metadata for the Weights & Biases run.
    
    Args:
        wandb_run (wandb.Run): The W&B run object.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        device (torch.device): Device used for computation.
        wandb_tag_mode (str): Tag for the run mode.
        quant_used (str): Quantization method used, if any.
        model_used (SegformerForSemanticSegmentation): The model being evaluated.
        ds_num_shards (int): Number of dataset shards.
        ds_shards_mod (float): Modulo for dataset shard logging.
    
    Returns:
        wandb.Run: The updated W&B run object.
    """

    wandb_tags = [model_name, dataset_name, device.type, wandb_tag_mode]
    if quant_used:
        wandb_tags += [quant_used]
    else:
        wandb_tags += [str(model_used.config.torch_dtype)]
    wandb_run.tags = wandb_tags
    wandb_run.notes = f"{datetime.now().isoformat()}, " \
        f"model size {model_used.get_memory_footprint()*1.0e-6:.2f} MB, " \
        f"{ds_num_shards=}, {ds_shards_mod=}"
    return wandb_run

def log_wandb_results(results, model):
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        results (dict): Evaluation results to log.
        model (SegformerForSemanticSegmentation): The model being evaluated.
    """

    wandb.log({
        'mean_iou': results['mean_iou'],
        'mean_accuracy': results['mean_accuracy'],
        'overall_accuracy': results['overall_accuracy'],
        'memory_footprint_MB': float(f"{model.get_memory_footprint()*1.0e-6:.2f}")
    })
