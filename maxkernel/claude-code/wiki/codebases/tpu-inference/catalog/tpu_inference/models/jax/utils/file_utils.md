---
title: 'Module: tpu_inference/models/jax/utils/file_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/utils/file_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.utils.file_utils`/
symbols:
  is_hf_repo: is_hf_repo().
  download_model_weights_from_hf: download_model_weights_from_hf().
  logger: logger.
  hfs: hfs.
  LOCK_DIR: LOCK_DIR.
  run_cmd: run_cmd().
  delete_file: delete_file().
  list_files: list_files().
  get_lock: get_lock().
  get_free_disk_size: get_free_disk_size().
  list_hf_repo: list_hf_repo().
  get_hf_model_weights_size: get_hf_model_weights_size().
  DisabledTqdm: DisabledTqdm#
  DisabledTqdm.__init__: DisabledTqdm#__init__().
---
# Module: [`tpu_inference/models/jax/utils/file_utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py)

## Classes
### `DisabledTqdm`
- def: [`tpu_inference/models/jax/utils/file_utils.py:93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L93)
- signature: `class DisabledTqdm(tqdm):`
- protocol/private: `__init__`[`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L95)

## Functions
- `delete_file(path: str)` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L42)
- `download_model_weights_from_hf(model_path: str, cache_dir: Optional[str], weights_format: str)` — [`L99`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L99)
- `get_free_disk_size(path: str='/')` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L68)
- `get_hf_model_weights_size(repo_id: str, weights_format: str)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L85)
- `get_lock(model_name_or_path: str)` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L54)
- `is_hf_repo(repo_id: str)` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L76)
- `list_files(dir: str, pattern: str='*')` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L49)
- `list_hf_repo(repo_id: str, pattern: str='**')` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L80)
- `run_cmd(cmd: str, *args, **kwargs)` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L38)

## Module values
- `LOCK_DIR` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L33)
- `hfs` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L31)
- `logger` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/file_utils.py#L29)

