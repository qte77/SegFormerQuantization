# SegFormer Quantization Pipeline

An end-to-end evaluation pipeline for SegFormer models on semantic segmentation tasks, with support for various quantization methods.

![Version](https://img.shields.io/badge/version-0.7.3-8A2BE2)
[![CodeFactor](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization/badge)](https://www.codefactor.io/repository/github/qte77/SegFormerQuantization)
[![CodeQL](https://github.com/qte77/SegFormerQuantization/actions/workflows/codeql.yaml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/codeql.yaml)
[![ruff](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yaml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/ruff.yaml)
[![pytest](https://github.com/qte77/SF-quant-temp/actions/workflows/pytest.yaml/badge.svg)](https://github.com/qte77/SF-quant-temp/actions/workflows/pytest.yaml)
[![Link Checker](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yaml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/links-fail-fast.yaml)
[![Deploy Docs](https://github.com/qte77/SegFormerQuantization/actions/workflows/generate-deploy-mkdocs-ghpages.yaml/badge.svg)](https://github.com/qte77/SegFormerQuantization/actions/workflows/generate-deploy-mkdocs-ghpages.yaml)
[![vscode.dev](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=vscode.dev&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://vscode.dev/github/qte77/SegFormerQuantization)

## Status

(DRAFT) (WIP) ----> Not fully implemented yet

For version history have a look at the [CHANGELOG](CHANGELOG.md).

## TOC

- [SegFormer Quantization Pipeline](#segformer-quantization-pipeline)
  - [Status](#status)
  - [TOC](#toc)
  - [Features](#features)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Python](#python)
    - [Docker](#docker)
    - [Test](#test)
  - [Configuration](#configuration)
  - [Documentation](#documentation)
  - [Project Structure](#project-structure)
  - [Architecture](#architecture)
    - [System](#system)
    - [Code](#code)
  - [TODO](#todo)
  - [DONE](#done)
  - [License](#license)

## Features

- Model loading and quantization (float8, int8, int4, int2)
- Dataset processing and sharding
- Evaluation metrics computation (mean IoU, mean accuracy, overall accuracy)
- Integration with Weights & Biases for experiment tracking

[↑](#toc)

## Setup

1. Install uv: `pip install uv`
2. Install dependencies: `uv sync [--frozen]`
3. Set up Weights & Biases API key in environment variables

[↑](#toc)

## Usage

### Python

```sh
uv run [--locked] python -m src 
```

### Docker

```sh
docker build -t segformer-quant-eval .
docker run segformer-quant-eval
```

To build with different python version

```sh
docker build --build-arg
  PYTHON_VERSION=<py_version> \
  .
```

### Test

```sh
uv sync --only-group dev
uv run pytest tests/
```

[↑](#toc)

## Configuration

Adjust settings in `src/config.py` for model, dataset, and evaluation parameters.

## Documentation

[Documentation SegFormer Quantization Pipeline](https://qte77.github.io/SegFormerQuantization/)

[↑](#toc)

## Project Structure

```sh
/
├── src/
│ ├─ utils/
│ │ ├── data_processing.py
│ │ ├── evaluator.py
│ │ ├── general_utils.py
│ │ ├── model_loader.py
│ │ ├── quantization.py
│ │ └── wandb_utils.py
│ ├── app.py
│ └── config.py
└── pyproject.toml
```

[↑](#toc)

## Architecture

### System

<img src="assets/images/SegFormerQuantization.C4.System.dark.png#gh-dark-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" />
<img src="assets/images/SegFormerQuantization.C4.System.light.png#gh-light-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" />

### Code

<img src="assets/images/SegFormerQuantization.C4.Code.dark.png#gh-dark-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" />
<img src="assets/images/SegFormerQuantization.C4.Code.light.png#gh-light-mode-only" alt="SegFormerQuantization" title="SegFormerQuantization" width="60%" />

[↑](#toc)

## TODO

- [ ] TDD
  - Implement tests before implementing concrete function
  - test_model_loading, test_image_preprocessing
  - test_quantization, test_predict, test_end_to_end
  - Use pydantic and python typing
- [ ] mkdocs
  - Extend workflow to copy only files in nav of mkdocs.yaml
- [ ] bump
  - Check steps summary output json, possible bug in GHA
  - Check if version tag exists, abort if so to avoid association of tag with an unrelated commit
- [ ] `README.md`
  - Include badge for tests
  - Insert link to report and project within WandB
- [ ] Optional
  - Hugging Face
    - Include option to call HF API instead of saving model locally
      - Might be useful for evaluation purposes
  - Docker
    - Evaluate `callisto` for fast cloud-native builds

## DONE

- [x] Use pt or cuda images to reduce loading time size, e.g.
  - `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
  - `nvidia/12.6.3-base-ubuntu24.04`
- [x] mkdocs
  - Add .md to LICENSE/LICENSES to avoid download instead of open
  - Remove/Change #href ↑(#toc) to avoid conflict with gh-pages
  - Remove/Change #href for light/dark png to avoid conflict with
  - Fix mkdocs not indenting checkbox ul
  - Fix mkdocs not including png with plain in-line html, assets/ not copied by mkdocsgh-pages
- [x] bump
  - Fix `bump-my-version.yaml` rollback step to delete auto-created branch after `failure()`
    - Handle error `fatal: could not read Username` & `Error: Process completed with exit code 128.`
- [x] `CHANGELOG.md` auto-generate
  - Solution: do manual bump occasionally and populate `CHANGELOG.md` before
  - Conventional Commits `.gitmessage`
  - Tools like `git-changelog`
- [x] Docker
  - Where are site-packages in Dockerfile for copy to runtime located?
    - Solution: `.venv`
- [x] Branch protection rules
  - Push to main with PR only
  - Use dedicated branch `dev-auto-push-to-main`
  - Incorporate branch to workflow `bump-my-version.yaml`
  - Create workflow `update_changelog.yaml`

[↑](#toc)

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE.md) file for details Third-party licenses might also apply.

[↑](#toc)
