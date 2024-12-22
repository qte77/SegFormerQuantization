"""Configuration settings for the SegFormer evaluation pipeline."""

from torch import device, float32
from torch.cuda import is_available
from os import environ

# Global configuration
wandb_tag_mode = "eval"
wandb_tag_runmode = "multiple-runs"
dataset_name = "scene_parse_150"
ds_num_shards = 100
ds_shards_mod = ds_num_shards / 10
model_name_short = "b2"
model_name = f"nvidia/segformer-{model_name_short}-finetuned-ade-512-512"
metric_name = 'mean_iou'

# Paths
dataset_save_path = f"./datasets/{dataset_name}"
model_save_path = f"./models/{model_name}"
tokenizer_save_path = f"./tokenizers/{model_name}"

# WandB configuration
environ['WANDB_PROJECT'] = f'segformer-{dataset_name}-{wandb_tag_mode}-{wandb_tag_runmode}'
environ['WANDB_ENTITY'] = 'ba-segformer'

# Device configuration
compute_dtype = float32
device = device("cuda" if is_available() else "cpu")
