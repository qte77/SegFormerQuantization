"""
SegFormer evaluation pipeline for semantic segmentation tasks.

This module orchestrates the evaluation of SegFormer models with various
quantization methods on the Scene Parse 150 dataset. It handles model loading,
quantization, dataset preparation, evaluation, and result logging using
Weights & Biases.

The pipeline supports multiple quantization levels and efficient processing
through dataset sharding. Results are tracked and visualized for performance
analysis across different model configurations.

Usage:
    python app.py

Environment variables:
    WANDB_API_KEY: API key for Weights & Biases logging
    WANDB_PROJECT: Name of the W&B project
    WANDB_ENTITY: Name of the W&B entity (team or user)
"""

from config import *
from .utils.data_processing import load_dataset, get_processed_inputs
from .utils.model_loader import load_base_model, load_image_processor
from .utils.quantization import quantize_models
from .utils.evaluator import evaluate_model
from .utils.wandb_utils import create_wandb_run, create_wandb_run_meta, log_wandb_results
from .utils.general_utils import print_model_sizes
import wandb

def main():
    """
    Main execution function for the SegFormer evaluation pipeline.
    
    This function orchestrates the entire evaluation process, including
    model loading, dataset preparation, evaluation, and logging results.
    """	
    
    base_model = load_base_model(model_name, model_save_path, compute_dtype, device)
    image_processor = load_image_processor(model_name, tokenizer_save_path)
    dataset = load_dataset(dataset_save_path, dataset_name)
    wandb.login(relogin=True, force=True, key=environ['WANDB_API_KEY'])
    
    # Quantize models
    models = quantize_models(base_model, model_name, model_save_path, device)
    print_model_sizes(models)
    
    # Evaluation loop
    ds_subset = 'validation'
    for model_name, model in models.items():
        model.eval()
        for k in range(0, ds_num_shards):
            if k == 0 or k % ds_shards_mod == 0:
                print(f"Evaluating {model_name}, shard {k}/{ds_num_shards}")
            wandb_run = create_wandb_run(
                environ['WANDB_PROJECT'], environ['WANDB_ENTITY'],
                model_name, dataset_name
            )
            create_wandb_run_meta(
                wandb_run, model_name, dataset_name, device, wandb_tag_mode,
                model_name, model, ds_num_shards, ds_shards_mod
            )
            dataset_shard = dataset[ds_subset].shard(
                num_shards=ds_num_shards, index=k
            )
            results = evaluate_model(
                model, dataset_shard, image_processor, device,
                metric_name, model.config.id2label
            )
            log_wandb_results(results, model)
            wandb.finish()

if __name__ == "__main__":
    main()

