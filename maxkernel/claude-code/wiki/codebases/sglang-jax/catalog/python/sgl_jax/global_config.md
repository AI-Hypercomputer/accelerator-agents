---
title: 'Module: python/sgl_jax/global_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/global_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.global_config`/
symbols:
  global_config: global_config.
  GlobalConfig.retract_decode_steps: GlobalConfig#retract_decode_steps.
  GlobalConfig: GlobalConfig#
  GlobalConfig.default_init_new_token_ratio: GlobalConfig#default_init_new_token_ratio.
  GlobalConfig.default_min_new_token_ratio_factor: GlobalConfig#default_min_new_token_ratio_factor.
  GlobalConfig.default_new_token_ratio_decay_steps: GlobalConfig#default_new_token_ratio_decay_steps.
  GlobalConfig.__init__: GlobalConfig#__init__().
  GlobalConfig.verbosity: GlobalConfig#verbosity.
  GlobalConfig.default_backend: GlobalConfig#default_backend.
  GlobalConfig.flashinfer_workspace_size: GlobalConfig#flashinfer_workspace_size.
  GlobalConfig.skip_special_tokens_in_output: GlobalConfig#skip_special_tokens_in_output.
  GlobalConfig.spaces_between_special_tokens_in_out: GlobalConfig#spaces_between_special_tokens_in_out.
  GlobalConfig.enable_precache_with_tracing: GlobalConfig#enable_precache_with_tracing.
  GlobalConfig.enable_parallel_encoding: GlobalConfig#enable_parallel_encoding.
  GlobalConfig.tpu_scatter_min_local_size: GlobalConfig#tpu_scatter_min_local_size.
---
# Module: [`python/sgl_jax/global_config.py`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py)

## Classes
### `GlobalConfig`
- def: [`python/sgl_jax/global_config.py:6`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L6)
- doc: Store some global constants.
- signature: `class GlobalConfig:`
- members:
  - `default_backend` — [`L21`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L21)
  - `default_init_new_token_ratio` — [`L24`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L24)
  - `default_min_new_token_ratio_factor` — [`L27`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L27)
  - `default_new_token_ratio_decay_steps` — [`L30`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L30)
  - `enable_parallel_encoding` — [`L46`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L46)
  - `enable_precache_with_tracing` — [`L45`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L45)
  - `flashinfer_workspace_size` — [`L36`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L36)
  - `retract_decode_steps` — [`L35`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L35)
  - `skip_special_tokens_in_output` — [`L41`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L41)
  - `spaces_between_special_tokens_in_out` — [`L42`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L42)
  - `tpu_scatter_min_local_size` — [`L51`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L51)
  - `verbosity` — [`L18`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L18)
- protocol/private: `__init__`[`L14`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L14)
- used by: `retract_decode`, `release_req`, [`init_new_token_ratio`](srt/managers/scheduler.md#Scheduler.init_new_token_ratio), [`global_config`](global_config.md#global_config), [`new_token_ratio_decay`](srt/managers/scheduler.md#Scheduler.new_token_ratio_decay), [`min_new_token_ratio`](srt/managers/scheduler.md#Scheduler.min_new_token_ratio)

## Module values
- `global_config` — [`L54`](../../../../../../raw/code/sglang-jax/python/sgl_jax/global_config.py#L54)

