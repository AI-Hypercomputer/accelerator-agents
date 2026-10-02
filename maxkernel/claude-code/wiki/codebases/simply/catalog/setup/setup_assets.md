---
title: 'Module: setup/setup_assets.py'
type: catalog
provenance: extracted
module: setup/setup_assets.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `setup.setup_assets`/
symbols:
  main: main().
  download_models: download_models().
  download_datasets: download_datasets().
  VOCABS_DIR: VOCABS_DIR.
  MODELS_REPO: MODELS_REPO.
  DATASETS_REPO: DATASETS_REPO.
  MODELS_DIR: MODELS_DIR.
  DATASETS_DIR: DATASETS_DIR.
  check_existing: check_existing().
  reorganize_gemma_models: reorganize_gemma_models().
  setup_gemma_vocabs: setup_gemma_vocabs().
  setup_qwen_vocabs: setup_qwen_vocabs().
---
# Module: [`setup/setup_assets.py`](../../../../../raw/code/simply/setup/setup_assets.py)

## Functions
- `check_existing(directory: str)` — [`L225`](../../../../../raw/code/simply/setup/setup_assets.py#L225) — Check if directory exists and has content.
- `download_datasets(datasets_dir: str, repo: str = DATASETS_REPO)` — [`L204`](../../../../../raw/code/simply/setup/setup_assets.py#L204) — Download datasets from HuggingFace.
- `download_models(models_dir: str, repo: str = MODELS_REPO)` — [`L173`](../../../../../raw/code/simply/setup/setup_assets.py#L173) — Download pretrained models from HuggingFace.
- `main()` — [`L233`](../../../../../raw/code/simply/setup/setup_assets.py#L233)
- `reorganize_gemma_models(models_dir: str)` — [`L47`](../../../../../raw/code/simply/setup/setup_assets.py#L47) — Reorganize Gemma model directories for Simply's checkpoint format.
- `setup_gemma_vocabs(models_dir: str, vocabs_dir: str)` — [`L114`](../../../../../raw/code/simply/setup/setup_assets.py#L114) — Copy Gemma tokenizer files to vocabs directory.
- `setup_qwen_vocabs(models_dir: str, vocabs_dir: str)` — [`L142`](../../../../../raw/code/simply/setup/setup_assets.py#L142) — Copy Qwen tokenizer files to vocabs directory.

## Module values
- `DATASETS_DIR` — [`L39`](../../../../../raw/code/simply/setup/setup_assets.py#L39)
- `DATASETS_REPO` — [`L44`](../../../../../raw/code/simply/setup/setup_assets.py#L44)
- `MODELS_DIR` — [`L38`](../../../../../raw/code/simply/setup/setup_assets.py#L38)
- `MODELS_REPO` — [`L43`](../../../../../raw/code/simply/setup/setup_assets.py#L43)
- `VOCABS_DIR` — [`L40`](../../../../../raw/code/simply/setup/setup_assets.py#L40)

