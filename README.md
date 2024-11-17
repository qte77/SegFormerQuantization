# SegFormer Quantization pipeline

An end-to-end evaluation pipeline for SegFormer models on semantic segmentation tasks, with support for various quantization methods.

[![CodeFactor](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization/badge)](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization)
[![Ruff](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yml)
[![Links (Fail Fast)](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yml)
[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://open.vscode.dev/qte77/SegFormerQuantization)

## Status

[DRAFT] [WIP] ----> Not fully implemented yet

The current version is <0.1.0>. For version history have a look at CHANGELOG.md.

## Features

- Model loading and quantization (float8, int8, int4, int2)
- Dataset processing and sharding
- Evaluation metrics computation (mean IoU, mean accuracy, overall accuracy)
- Integration with Weights & Biases for experiment tracking

## Project Structure

```
/
├── app/
│ ├── app.py
│ ├── config.py
│ └── utils/
│   ├── data_processing.py
│   ├── evaluator.py
│   ├── general_utils.py
│   ├── model_loader.py
│   ├── quantization.py
│   └── wandb_utils.py
└── requirements.txt
```
	
## Setup

1. Install dependencies: `poetry install` or `pip install -r requirements.txt`
2. Set up Weights & Biases API key in environment variables

## Usage

Run the main script: `python app/app.py`


## Configuration

Adjust settings in `app/config.py` for model, dataset, and evaluation parameters.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
For third-party licenses, see the [LICENSES](LICENSES) file.

