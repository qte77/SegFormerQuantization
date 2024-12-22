# %% [markdown]
# # SegFormer baseline to fine-tune model and quantize weights
# 
# This notebook fine-tunes a SegFormer on a (toy) dataset and test on a single image to establish a baseline. It also uses [torch.Tensor.half](https://pytorch.org/docs/stable/generated/torch.Tensor.half.html) to quantize the wights to fp16 as a baseline as well. 
# 
# - [Results of this baseline experiment](https://qte77.github.io/SegFormerBaseline-FineTuning-results/)
# - [SegFormer Part 1, Description](https://qte77.github.io/SegFormer-Part1-Description/)
# - [SegFormer Part 2, Quantization Description](https://qte77.github.io/SegFormer-Part3-Quantization-Description/)
# - [Fine-tune SegFormer on a custom dataset](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb)
# - [Hugging Face - SegFormerForSemanticSegmentation](https://huggingface.co/docs/transformers/main/model_doc/segformer#transformers.SegformerForSemanticSegmentation)
#

# %% [code]
# https://peps.python.org/pep-0008/#maximum-line-length
# get basic libraries
import requests
import zipfile
import io
import os
import json
import copy
from PIL import Image
# from subprocess import check_call
# from sys import executable

# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %% [code]
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# %% [code]
# TODO implement !pip install for notebooks
# Using pip from your program
# https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
# def import_or_install(package):
#    try:
#        __import__(package)
#    except ImportError:
#        pip.main(['install', '-qq', package])
# for package in ['transformers', 'datasets', 'evaluate']:
#    # import_or_install(package)
#    # check_call([executable, '-m', 'pip', 'install', '-qq', module])
#    pip.main(['install', '-qq', package])
import evaluate
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation

# %% [code]
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
model_name = "nvidia/mit-b0"
metric_used = "mean_iou"
train_n_epochs = 1000
dataset_toy_url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
load_full_dataset = False
dataset_full_name = "scene_parse_150"
# root_dir = './ADE20k_toy_dataset' # kaggle: /kaggle/working, colab: /content
root_dir = './ADE20k_toy_dataset'
test_image = './ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg'
test_image_annotation = './ADE20k_toy_dataset/annotations/training/ADE_train_00000001.png'

# %% [code]
def download_data():
    url = dataset_toy_url
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

download_data()

# %% [code]
if load_full_dataset:
  dataset = load_dataset(dataset_full_name, trust_remote_code=True, data_dir=".")

# %% [markdown]
# ## Define PyTorch dataset and dataloaders

# %% [code]
#@title `class SemanticSegmentationDataset(Dataset)`
#TODO export to utils
class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

# %% [code]
image_processor = SegformerImageProcessor(do_reduce_labels=True)

# %% [code]
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)

# %% [code]
batch = next(iter(train_dataloader))
mask = (batch["labels"] != 255)

# %% [code]
#TODO export to utils
def print_data_info():
    encoded_inputs = train_dataset[0]
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))
    print(f"{encoded_inputs['pixel_values'].shape=}")
    print(f"{encoded_inputs['labels'].shape=}")
    print(f"{encoded_inputs['labels']=}")
    print(f"{encoded_inputs['labels'].squeeze().unique()=}")
    for k,v in batch.items():
      print(k, v.shape)
    print(f"{batch['labels'].shape=}")
    print(f"{mask=}")
    print(f"{batch['labels'][mask]=}")

print_data_info()

# %% [markdown]
# ## Define the model

# %% [code]
# load id2label mapping from a JSON on the hub
id2label = json.load(open(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"), "r"))
id2label = { int(k): v for k, v in id2label.items() }
label2id = { v: k for k, v in id2label.items() }

# %% [code]
# %%time
# define model
model_orig = SegformerForSemanticSegmentation.from_pretrained(
    model_name, num_labels=150, id2label=id2label, label2id=label2id,
    # torch_dtype=torch.float16,
)
model_fined = copy.deepcopy(model_orig)
model_orig == model_fined

# %% [markdown]
# ## Fine-tune

# %% [code]
metric = evaluate.load(metric_used)

# %% [code]
# image_processor.do_reduce_labels

# %% [code]
# define optimizer
optimizer = torch.optim.AdamW(model_fined.parameters(), lr=0.00006)
# move model to GPU
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

model_orig.to(device)
model_fined.to(device)

# %% [code]
# %%time
#@title `train model`
#TODO export to utils
model_fined.train()
for epoch in tqdm(range(train_n_epochs), position=0, leave=True):  # loop over the dataset multiple times
   for idx, batch in enumerate(train_dataloader): # tqdm(train_dataloader)
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_fined(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)

          # note that the metric expects predictions + labels as numpy arrays
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if epoch % 5 == 0 and idx % 100 == 0:
          # currently using _compute instead of compute
          # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
          metrics = metric._compute(
                  predictions=predicted.cpu(),
                  references=labels.cpu(),
                  num_labels=len(id2label),
                  ignore_index=255,
                  reduce_labels=False, # we've already reduced the labels ourselves
              )
          print("Epoch:", epoch)
          print("Loss:", loss.item())
          print("Mean_iou:", metrics["mean_iou"])
          print("Mean accuracy:", metrics["mean_accuracy"])

# %% [markdown]
# ## half fine-tuned

# %% [code]
model_fined_half = copy.deepcopy(model_fined).half()

# %% [code]
model_sizes = {
    "orig": model_orig.get_memory_footprint(),
    "fined": model_fined.get_memory_footprint(),
    "fined_half": model_fined_half.get_memory_footprint()
}

# %% [markdown]
# ## Inference

# %% [code]
image = Image.open(test_image)
# image

# %% [code]
# prepare the image for the model
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)

# %% [code]
# forward pass
with torch.no_grad():
    pred_outputs = {
        "orig": model_orig(pixel_values=pixel_values),
        "fined": model_fined(pixel_values=pixel_values),
        "fined_half": model_fined_half(pixel_values=pixel_values.half())
    }
# logits are of shape (batch_size, num_labels, height/4, width/4)
[print(f"pred_{k}_logits.shape={pred_outputs[k].logits.cpu().shape}") for k in pred_outputs.keys()]

# %% [code]
pred_seg_map = {
    k:(
        image_processor.post_process_semantic_segmentation(v, target_sizes=[image.size[::-1]])[0]
    ).cpu().numpy()
    for k,v in pred_outputs.items()
}

# %% [code]
#TODO export to utils
def ade_palette(num_labels: int = 150, num_colors: int = 256):
    """ADE20K palette that maps each class of 'num_labels' to 'num_colors' values."""
    scale = num_colors / num_labels
    return [[int(k * scale) for i in range(0,3)] for k in range(0, num_labels)]

# %% [code]
#TODO export to utils
def plot_pred_map(predicted_segmentation_map, ade_palette):
    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                          predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    palette = np.array(ade_palette)
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()
    
def get_annotation_map(plot=False):
    map = Image.open(test_image_annotation)
    # convert map to NumPy array
    map = np.array(map)
    map[map == 0] = 255 # background class is replaced by ignore_index
    map = map - 1 # other classes are reduced by one
    map[map == 254] = 255
    if plot:
        map
    return map

def get_anno_map_unique_classes(map, print_cls=True):
    classes_map = np.unique(map).tolist()
    unique_classes = [model_fined.config.id2label[idx] if idx!=255 else None for idx in classes_map]
    if print:
        print(f"Classes in this image: {unique_classes}")
    return unique_classes

def plot_coloured_map(map, image, ade_palette):
    color_seg = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette)
    for label, color in enumerate(palette):
        color_seg[map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

# %% [code]
map = get_annotation_map()
unique_cls = get_anno_map_unique_classes(map)
unique_cls_ids = [label2id[k] for k in unique_cls if k is not None]
# ade_palette = ade_palette()
# plot_pred_map(predicted_segmentation_map, ade_palette)
# plot_coloured_map(map, image, ade_palette)

# %% [code]
# metric expects a list of numpy arrays for both predictions and references
metrics_param = {
  "references": [map],
  "num_labels": len(id2label),
  "ignore_index": 255,
  "reduce_labels": False, # we've already reduced the labels ourselves
}
metrics = { k:metric._compute(predictions=[v], **metrics_param) for k,v in pred_seg_map.items() }

# %% [code]
# overall metrics
#TODO print model sizes in MiB
for m in metrics.keys():
    print(m)
    print(f"\tsize={(model_sizes[m]/8192):.2f}")
    for k,v in list(metrics[m].items())[:3]:
        print(f"\t{k}={v}")
        
# per category
metric_table = dict()
for m in metrics.keys():
    metric_table[m] = {
        id2label[i]:[
            metrics[m]["per_category_iou"][i],
            metrics[m]["per_category_accuracy"][i]
        ]
        for i in unique_cls_ids
    }
    print("---------------------")
    print(m)
    print(pd.DataFrame.from_dict(metric_table[m], orient="index", columns=["IoU", "Acc"]))
