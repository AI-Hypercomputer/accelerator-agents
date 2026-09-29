---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_tpu_common`/
symbols:
  build_splash_kernel: build_splash_kernel().
  NUM_LANES.NUM_LANES: NUM_LANES.NUM_LANES.
  check_inputs_support: check_inputs_support().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py)

## Functions
- `build_splash_kernel(mask: base.Mask, splash_config: splash.SplashConfig, q_seq_len: int, kv_seq_len: int, is_mqa: bool, save_residuals: bool)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py#L65) — Returns the SplashAttention kernel maker and splash mask.
- `check_inputs_support(*x, bias, logits_dtype, dropout_mask, dropout_rate, paging_info, mask)` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py#L30) — Checks that Mosaic TPU supports the given inputs.

## Module values
- `NUM_LANES` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py#L27)

