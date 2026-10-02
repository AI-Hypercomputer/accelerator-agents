---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v2.bench_compare`/
symbols:
  v1_bc: v1_bc.
  ep_sharding: ep_sharding.
  run_v2: run_v2().
  v1_bc_eff: v1_bc_eff.
  v2_bc_eff: v2_bc_eff.
  run_v1: run_v1().
  v2_bc: v2_bc.
  bts2: bts2.
  log: log().
  v2_pallas_times: v2_pallas_times.
  label: label.
  v1_pallas_times: v1_pallas_times.
  pct: pct.
  num_tokens: num_tokens.
  v2_ms: v2_ms.
  w1: w1.
  w2: w2.
  w3: w3.
  qbk_arg: qbk_arg.
  shard_key: shard_key.
  v1_trace: v1_trace.
  v2_trace: v2_trace.
  trace_timeit: trace_timeit().
  tokens: tokens.
  gating: gating.
  padded_nt: padded_nt.
  v2_pallas_avg: v2_pallas_avg.
  make_sharded: make_sharded().
  gating_local_shape: gating_local_shape.
  v2_tuned_table: v2_tuned_table.
  local_nt_raw: local_nt_raw.
  pad_local: pad_local.
  diff: diff.
  topk_idx: topk_idx.
  ep_size: ep_size.
  v1_times: v1_times.
  v2_times: v2_times.
  token_candidates: token_candidates.
  mesh: mesh.
  v1_avg: v1_avg.
  v2_avg: v2_avg.
  E: E.
  _event_device_interval_us: _event_device_interval_us().
  _extract_fair_device_spans_ms: _extract_fair_device_spans_ms().
  topk_logits: topk_logits.
  aligned_local_nt: aligned_local_nt.
  k5: k5.
  bse: bse.
  quantize_shard_map: quantize_shard_map().
  d: d.
  warmup: warmup.
  iters: iters.
  use_trace: use_trace.
  topk_wts: topk_wts.
  nt: nt.
  P: P.
  num_devices: num_devices.
  f: f.
  top_k: top_k.
  quant_block_k: quant_block_k.
  v1_fused_ep_moe: v1_fused_ep_moe.
  V1BlockConfig: V1BlockConfig.
  devices: devices.
  _device_duration_us: _device_duration_us().
  v1_pallas_avg: v1_pallas_avg.
  direct_scaled_dot: direct_scaled_dot.
  w3_scale_s: w3_scale_s.
  w2_scale_s: w2_scale_s.
  w1_scale_s: w1_scale_s.
  results: results.
  v1_ms: v1_ms.
  use_fp8: use_fp8.
  v2_bf_override: v2_bf_override.
  v2_bt_override: v2_bt_override.
  v2_btc_override: v2_btc_override.
  v2_bts_override: v2_bts_override.
  V1_TUNED: V1_TUNED.
  wall_timeit: wall_timeit().
  _as_float: _as_float().
  gating_per_dev: gating_per_dev.
  bt2: bt2.
  bf2: bf2.
  btc2: btc2.
  t0: t0.
  TRACE_ROOT: TRACE_ROOT.
  v2_decode_mode: v2_decode_mode.
  parse_csv_int: parse_csv_int().
  V2_TUNED: V2_TUNED.
  V2_DIRECT_SCALED_DOT_TUNED: V2_DIRECT_SCALED_DOT_TUNED.
  align_local_tokens_for_v2: align_local_tokens_for_v2().
  unpack_v2_config: unpack_v2_config().
  key: key.
  _load_trace: _load_trace().
  _extract_pallas_durations_ms: _extract_pallas_durations_ms().
  _slowest_per_iteration: _slowest_per_iteration().
  k1: k1.
  k2: k2.
  k3: k3.
  k4: k4.
  i: i.
  dev: dev.
  bt: bt.
  bf: bf.
  bd1: bd1.
  bd2: bd2.
  bts: bts.
  btc: btc.
  bfc: bfc.
  bd1c: bd1c.
  bd2c: bd2c.
  bse2: bse2.
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py)

## Functions
- `_as_float(v: Any)` — [`L254`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L254)
- `_device_duration_us(e: dict[str, Any])` — [`L261`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L261)
- `_event_device_interval_us(e: dict[str, Any])` — [`L269`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L269)
- `_extract_fair_device_spans_ms(trace: dict[str, Any], kernel_name_re: re.Pattern[str], *, include_preceding_hlo_categories: tuple[str, ...] = (), max_preceding_gap_us: float = 5000)` — [`L286`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L286)
- `_extract_pallas_durations_ms(trace: dict[str, Any], kernel_name_re: re.Pattern[str])` — [`L224`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L224)
- `_load_trace(trace_root: str)` — [`L206`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L206)
- `_slowest_per_iteration(by_pid: dict[int, list[float]])` — [`L277`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L277)
- `align_local_tokens_for_v2(local_num_tokens: int)` — [`L135`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L135)
- `log(msg)` — [`L34`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L34) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `make_sharded(rng_key, shape, dtype, scale=1)` — [`L122`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L122)
- `parse_csv_int(env_key: str, default: list[int])` — [`L70`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L70)
- `quantize_shard_map(w)` — [`L172`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L172)
- `run_v1(bc=v1_bc)` — [`L413`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L413) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `run_v2(bc=v2_bc)` — [`L470`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L470) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `trace_timeit(run_fn, warmup_n, iters_n, kernel_name_re, step_name, *, include_preceding_hlo_categories: tuple[str, ...] = ())` — [`L339`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L339)
- `unpack_v2_config(cfg: tuple[int, ...])` — [`L141`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L141)
- `wall_timeit(run_fn, warmup_n, iters_n)` — [`L193`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L193)

## Module values
- `E` — [`L55`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L55)
- `P` — [`L47`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L47)
- `TRACE_ROOT` — [`L31`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L31)
- `V1BlockConfig` — [`L45`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L45)
- `V1_TUNED` — [`L79`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L79)
- `V2_DIRECT_SCALED_DOT_TUNED` — [`L98`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L98)
- `V2_TUNED` — [`L89`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L89)
- `aligned_local_nt` — [`L460`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L460)
- `bd1` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bd1c` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bd2` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bd2c` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bf` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bf2` — [`L449`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L449)
- `bfc` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bse` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bse2` — [`L449`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L449)
- `bt` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bt2` — [`L449`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L449)
- `btc` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `btc2` — [`L449`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L449)
- `bts` — [`L401`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L401)
- `bts2` — [`L449`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L449)
- `d` — [`L53`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L53)
- `dev` — [`L383`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L383)
- `devices` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L49)
- `diff` — [`L518`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L518)
- `direct_scaled_dot` — [`L62`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L62)
- `ep_sharding` — [`L109`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L109) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `ep_size` — [`L51`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L51)
- `f` — [`L54`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L54)
- `gating` — [`L391`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L391)
- `gating_local_shape` — [`L381`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L381)
- `gating_per_dev` — [`L382`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L382)
- `i` — [`L383`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L383)
- `iters` — [`L58`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L58)
- `k1` — [`L153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L153)
- `k2` — [`L153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L153)
- `k3` — [`L153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L153)
- `k4` — [`L153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L153)
- `k5` — [`L153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L153)
- `key` — [`L152`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L152)
- `label` — [`L509`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L509)
- `local_nt_raw` — [`L459`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L459)
- `mesh` — [`L50`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L50)
- `nt` — [`L526`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L526)
- `num_devices` — [`L48`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L48)
- `num_tokens` — [`L377`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L377) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `pad_local` — [`L461`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L461)
- `padded_nt` — [`L462`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L462)
- `pct` — [`L519`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L519)
- `qbk_arg` — [`L161`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L161)
- `quant_block_k` — [`L60`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L60)
- `results` — [`L375`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L375)
- `shard_key` — [`L384`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L384)
- `t0` — [`L30`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L30)
- `token_candidates` — [`L107`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L107)
- `tokens` — [`L380`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L380)
- `top_k` — [`L56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L56)
- `topk_idx` — [`L396`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L396)
- `topk_logits` — [`L397`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L397)
- `topk_wts` — [`L398`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L398)
- `use_fp8` — [`L59`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L59)
- `use_trace` — [`L61`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L61)
- `v1_avg` — [`L507`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L507)
- `v1_bc` — [`L402`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L402)
- `v1_bc_eff` — [`L405`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L405) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `v1_fused_ep_moe` — [`L44`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L44)
- `v1_ms` — [`L527`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L527)
- `v1_pallas_avg` — [`L513`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L513)
- `v1_pallas_times` — [`L442`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L442)
- `v1_times` — [`L441`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L441)
- `v1_trace` — [`L433`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L433)
- `v2_avg` — [`L508`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L508)
- `v2_bc` — [`L458`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L458) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `v2_bc_eff` — [`L463`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L463)
- `v2_bf_override` — [`L64`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L64)
- `v2_bt_override` — [`L65`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L65)
- `v2_btc_override` — [`L66`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L66)
- `v2_bts_override` — [`L67`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L67)
- `v2_decode_mode` — [`L63`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L63)
- `v2_ms` — [`L527`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L527)
- `v2_pallas_avg` — [`L514`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L514)
- `v2_pallas_times` — [`L498`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L498) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `v2_times` — [`L497`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L497)
- `v2_trace` — [`L490`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L490)
- `v2_tuned_table` — [`L448`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L448)
- `w1` — [`L156`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L156) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `w1_scale_s` — [`L160`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L160)
- `w2` — [`L157`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L157) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `w2_scale_s` — [`L160`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L160)
- `w3` — [`L158`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L158) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `w3_scale_s` — [`L160`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L160)
- `warmup` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_compare.py#L57)

