# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), i.e. MAJOR.MINOR.PATCH (Breaking.Feature.Patch).

Types of changes:

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [Unreleased]

---

## [0.5.8] - 2024-12-09

### Added

- Optional step to explicitly push to workflow bump-my-version

### Changed

- ghpages action back to simple version
- Import of config.py to relative
- bump-my-version targets in files
- README.md usage infos to contain venv and test, TODO to use pytorch or cuda images
- Renamed workflow bump2version to bump-my-version

### Fixed

- Moved package files to ./app
- Removed typos from Dockerfile
- Disabled GHA bump-my-version trigger pull-request and push to main

---

## [0.5.6] - 2024-11-20

## [0.5.5] - 2024-11-20

### Fixed

- ghpages script indent

## [0.5.4] - 2024-11-20

### Fixed

- ghpages docstring nav file

## [0.5.3] - 2024-11-20

### Fixed

- Script gen docstring to eclude __init__.py

### [0.5.2] - 2024-11-20

### [0.5.1] - 2024-11-20

### Fixed

- Typo in ghpages workflow

### Changed

- Separated patch and PR bump type

## [0.5.0] - 2024-11-20

### Fixed

- Indenting in ghpages workflow

## [0.4.0] - 2024-11-20

### Added

- `gen_mkdocs_docstring_pages.py` containing script to recurse package for mkdocs docstrings

### Changed

- Workflow ghpages generate and deploy to include recurse docstring script

### Fixed

- Image size in README
- Check `bump_type` instead of `input` in push and PR

## [0.3.0] - 2024-11-20

## [0.2.2] - 2024-11-20

## [0.2.1] - 2024-11-17

### Added

- Dockerfile
- Bumpversion for push and PR on `main`

### Changed

- Extension .md for LICENSE(S) for mkdocs output

## [0.2.0] - 2024-11-17

### Added

- Management: `pyproject.toml`
- Documentation: `PRD.md`, `FRD.md`, `DRD.md`
- Documentation gh-pages: `generate-deploy-mkdocs-ghpages.yml`
- Versioning: `CHANGELOG.md`, GHA Bump2version
- Standarization: `.gitmessage`
- Third-party `LICENSES`
- mkdocs: `__init__.py`
- Packaging: `__main__.py`

### Changed

- `LICENSE`
- `.gitignore`
- Actions: `links-fail-fast.yml`, `codeql.yml`

## Removed

- `.bumpversion.cfg`
- `requirements.txt`
