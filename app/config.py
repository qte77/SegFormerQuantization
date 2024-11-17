"""Configuration settings for the SegFormer evaluation pipeline."""

import torch
from os import environ

# Global configuration
wandb_tag_mode = "eval"
wandb_tag_runmode = "multiple-runs"
dataset_name = "scene_parse_150"
model_name_short = "b2"
model_name = f"nvidia/segformer-{model_name_short}-finetuned-ade-512-512"
metric_name = 'mean_iou'
ds_num_shards = 100
ds_shards_mod = ds_num_shards / 10

# Paths
model_save_path = f"./models/{model_name}"
tokenizer_save_path = f"./tokenizers/{model_name}"
dataset_save_path = f"./datasets/{dataset_name}"

# WandB configuration
environ['WANDB_PROJECT'] = f'segformer-{dataset_name}-{wandb_tag_mode}-{wandb_tag_runmode}'
environ['WANDB_ENTITY'] = 'ba-segformer'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_dtype = torch.float32

