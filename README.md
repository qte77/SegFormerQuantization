# SegFormer Quantization Pipeline

<!-- mkdocs only start
## TOC
mkdocs only start -->

An end-to-end evaluation pipeline for SegFormer models on semantic segmentation tasks, with support for various quantization methods.

[![CodeFactor](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization/badge)](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization)
[![Ruff](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yml)
[![Links (Fail Fast)](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yml)
[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://vscode.dev/github/qte77/SegFormerQuantization)

## Status

(DRAFT) (WIP) ----> Not fully implemented yet

The current version is <0.5.7>. For version history have a look at the [CHANGELOG](CHANGELOG.md).

## TOC <!-- mkdocs exclude { data-search-exclude } -->

* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Documentation](#documentation)
* [UML](#uml)
* [TODO](#todo)
* [License](#license)

## Features

- Model loading and quantization (float8, int8, int4, int2)
- Dataset processing and sharding
- Evaluation metrics computation (mean IoU, mean accuracy, overall accuracy)
- Integration with Weights & Biases for experiment tracking

[↑](#toc)
	
## Setup

1. Install dependencies: `pip install poetry==1.8.4 && poetry install`.
2. Set up Weights & Biases API key in environment variables

[↑](#toc)

## Usage

### Python

If inside `poetry` venv

```sh
python -m app
```

or if outside

```sh
poetry run python -m app
```

### Docker

```bash
docker build -t segformer-quant-eval .
docker run segformer-quant-eval
```

[↑](#toc)

## Configuration

Adjust settings in `app/config.py` for model, dataset, and evaluation parameters.

## Documentation

[Documentation SegFormer Quantization Pipeline](https://qte77.github.io/SegFormerQuantization/)

[↑](#toc)

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
└── pyproject.toml
```

[↑](#toc)

## UML

<img src="assets/SegFormerQuantization.UML.dark.png#gh-dark-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" /> <!-- mkdocs exclude { data-search-exclude } -->
<img src="assets/SegFormerQuantization.UML.light.png#gh-light-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" />

[↑](#toc)

## TODO

- [ ] Implement tests before concrete function (TDD)
	- test_model_loading, test_image_preprocessing
	- test_quantization, test_predict, test_end_to_end
- [ ] Include option to call HF API instead of saving model locally
- [ ] Use pydantic and python typing
- [ ] Insert link to and report of WandB project
- mkdocs
	- [x] Add .md to LICENSE/LICENSES to avoid download instead of open
	- [x] Remove/Change #href ↑(#toc) to avoid conflict with gh-pages
	- [x] Remove/Change #href for light/dark png to avoid conflict with gh-pages

[↑](#toc)

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE.md) file for details.
For third-party licenses, see the [LICENSES](LICENSES.md) file.

[↑](#toc)

