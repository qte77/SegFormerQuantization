from os.path import exists
from timeit import repeat
from time import perf_counter
from psutil import virtual_memory
from statistics import mean, stdev
from datasize import DataSize
from torch import (
  float64, float32, float16,
  bfloat16, int8, uint8, bool,
  device, rand, randint
)
from torch.cuda import (
    memory_allocated, mem_get_info,
    is_available, device_count
)
from datasets import (
  Dataset, DatasetDict,
  load_from_disk, load_dataset
)
from transformers import (
  # SegformerForSemanticSegmentation,
  SegformerForImageClassification,
  AutoImageProcessor, SegformerImageProcessor
)

def get_device_info() -> dict:
  """
  Returns device info for CPU and GPU (cuda). TPU and others not implemented.
  """
  d = device("cuda" if is_available() else "cpu")
  di = {}
  di["devicename"] = d
  cpu_max_memory = int(virtual_memory().total/2.**30)
  di["cpu_max_memory"] = { "cpu" : f"{cpu_max_memory}GiB" }

  if d.type == "cuda":
    di["cuda_free_mem"] = int(mem_get_info()[0]/1024**3)
    cuda_max_memory = int(di["cuda_free_mem"])
    di["cuda_n_gpus"] = device_count()
    di["cuda_max_memory"] = {
      i : f"{cuda_max_memory}GiB"
      for i in range(di["cuda_n_gpus"])
    }
  
  return di

def get_dataset(ds_path: str, dataset_url: str = "") -> DatasetDict:
  """Loads dataset."""
  # load a custom dataset from local/remote files/folders using ImageFolder
  # local/remote files (supporting : tar, gzip, zip, xz, rar, zstd)
  # local: dataset = load_dataset("imagefolder", data_dir="path_to_folder")
  # hub, e.g.: dataset = load_dataset("cifar10")
  if exists(ds_path):
    return load_from_disk(ds_path)
  else:
    dataset = load_dataset("imagefolder", data_files=dataset_url)
    dataset.save_to_disk(ds_path)
    return dataset

def get_split_dataset(
  ds_path: str, dataset_url: str = "", test_size: float = 0.1
) -> dict[Dataset]:
  """Returns split dataset."""
  dataset = get_dataset(ds_path, dataset_url)
  return {
    k:v for k,v in
    dataset["train"].train_test_split(test_size=0.1).items()
  }

def get_image_processor(
  model_checkpoint: str, save_path: str
) -> SegformerImageProcessor:
  """Loads and saves ImageProcessor, i.e Tokenizer."""
  # AutoImgProc errors when no ImProc in checkpoint config
  # but loads base SegformerImageProcessor
  tok_loc = save_path if exists(save_path) else model_checkpoint
  print(f"Loading from {tok_loc}.")
  image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
  if tok_loc != save_path: image_processor.save_pretrained(save_path)
  return image_processor

def get_and_save_sf_model(
  model_checkpoint: str, save_path: str,
  max_memory: int, kwparams: dict, device_map: str = "cuda:0"
) -> SegformerForImageClassification:
  """
  Downloads and saves or SegformerForImageClassification model
  or loads it from local path if possible.
  """

  if "half" in kwparams.keys():
    half = True if kwparams["half"] else False
    kwparams.pop("half", None)
  else:
    half = False

  model_location = f"{save_path}/{model_checkpoint}"
  if not exists(model_location):
    model_location = model_checkpoint

  model = SegformerForImageClassification.from_pretrained(
    model_location,
    # device_map={'':torch.cuda.current_device()}
    # device_map=['auto','sequential']
    device_map=device_map,
    max_memory=max_memory,
    **kwparams
  )

  if half:
    model.half() # from f32 to f16
    model.is_quantized = True

  if model_location == model_checkpoint:
    model.save_pretrained(save_path)

  return model

def get_model_as_dict(
  model_checkpoint: str, save_path: str,
  max_memory: int, kwparams: dict, device_map: str = "cuda:0"
) -> dict:
  """Returns a dict containing model, load_time and memory_footprint."""
  m = {}
  m["load_time"] = perf_counter()
  m["model"] = get_and_save_sf_model(
    model_checkpoint, save_path, max_memory, kwparams
  )
  m["load_time"] = perf_counter() - m["load_time"]
  m["memory_footprint"] = m["model"].get_memory_footprint()
  return m

def print_models_stats(models_stats: dict, model_base_size: float) -> None:
  """
  Prints table of model stats.
  Expects dict of model names with dict of
  'load_time', 'memory_footprint', 'is_quantized' and 'hf_device_map'
  as well as 'model_base_size' for ratio of model sizes.
  """
  sep = "\t"
  format = "MiB"
  h = f"name{sep*2}size [{format}]{sep}ratio{sep}time"
  h = f"{h}{sep}device_map" # {sep}is_quantized
  print(h)
  for k in models_stats.keys():
    s = models_stats[k]["memory_footprint"] \
      if "memory_footprint" in models_stats[k].keys() else -1
    sprn = f"{DataSize(s):.2{format}}".replace(format, "")
    ratio_base = round(models_stats[k]["memory_footprint"] / model_base_size, 2)
    time = round(models_stats[k]["load_time"], 2)
    dm = models_stats[k]["device_map"]
    # iq = models_stats[k]["is_quantized"]
    print(f"{k}{sep}{sprn}{sep*2}{ratio_base}{sep}{time}{sep}{dm}") # {sep}{iq}

def get_calculation_speed(
  device_to_use: str, runs: int = 7, number: int = 77, tee: bool = False
) -> dict:
  """
  Creates tensors of size '(100, 100, 100)' and  runs them 'number' times within 'runs' and with dtypes
  'float64, float32, float16, bfloat16, int8, uint8, bool'.
  TPU not implemented.
  """
  tsize = (100, 100, 100)
  dtypes = (
    float64, float32, float16,
    bfloat16, int8, uint8, bool
  )
  timings = {}

  if device_to_use == "cuda":
    devices = ["cuda", "cpu"]
  elif device_to_use == "cpu":
    devices = [device_to_use]
  else:
    raise NotImplementedError

  for dev in devices:
    if tee: print(f"{dev=}")
    timings[dev] = {}

    for dt in dtypes:
      if tee: print(dt)
      dt_str = str(dt)
      timings[dev][dt_str] = {}
      with device(dev):
        if "float" in dt_str:
          o = rand(tsize, dtype=dt)
        elif "bool" in dt_str:
          o = rand(tsize) < 0.5
        else:
          o = randint(0, 1, tsize, dtype=dt)
        #TODO line magic not valid in py
        #t = %timeit -r 7 -n 77 -o o**2
        t = repeat(stmt='o**2', repeat=runs, number=number, globals=locals())
        timings[dev][dt_str]["mean"] = mean(t)
        timings[dev][dt_str]["stdev"] = stdev(t)
        timings[dev][dt_str]["raw"] = t
        if tee:
          print(
            "mean: %f, stdev: %f"
            % (
              timings[dev][dt_str]["mean"],
              timings[dev][dt_str]["stdev"]
            )
          )

  return timings
