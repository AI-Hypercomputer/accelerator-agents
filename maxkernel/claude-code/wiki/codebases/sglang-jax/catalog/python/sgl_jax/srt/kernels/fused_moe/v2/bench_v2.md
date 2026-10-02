---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v2.bench_v2`/
symbols:
  run_fn: run_fn().
  result: result.
  log: log().
  ref_kwargs: ref_kwargs.
  e: e.
  ep_sharding: ep_sharding.
  tune_configs: tune_configs.
  tuned_cfg: tuned_cfg.
  w2s_sc: w2s_sc.
  resolved_key: resolved_key.
  generate_tune_candidates: generate_tune_candidates().
  tag: tag.
  _default_cfgs: _default_cfgs.
  tag_resolved: tag_resolved.
  avg: avg.
  bc0: bc0.
  timing_label: timing_label.
  bc_resolved: bc_resolved.
  ks3: ks3.
  wall_avg: wall_avg.
  num_tokens: num_tokens.
  topk_idx: topk_idx.
  ref: ref.
  best_times: best_times.
  gating: gating.
  nt: nt.
  times: times.
  w1: w1.
  w2: w2.
  w3: w3.
  shard_key: shard_key.
  sk: sk.
  qbk_arg: qbk_arg.
  wait_times: wait_times.
  pad_local: pad_local.
  w3_shared_scale: w3_shared_scale.
  w3_shared: w3_shared.
  tokens: tokens.
  bts: bts.
  E: E.
  tokens_c: tokens_c.
  gating_c: gating_c.
  rel_err: rel_err.
  trace_timeit: trace_timeit().
  d: d.
  topk_logits: topk_logits.
  mesh: mesh.
  ep_size: ep_size.
  quant_block_k: quant_block_k.
  P: P.
  top_k: top_k.
  make_sharded: make_sharded().
  invalid_modes: invalid_modes.
  valid_routing_modes: valid_routing_modes.
  _repl: _repl().
  gating_local_shape: gating_local_shape.
  configs_to_try: configs_to_try.
  gating_c_local: gating_c_local.
  result_f32: result_f32.
  bench_tol: bench_tol.
  f: f.
  local_nt_raw: local_nt_raw.
  entries: entries.
  tidx: tidx.
  se_inter: se_inter.
  interleave_bt_modes: interleave_bt_modes.
  num_devices: num_devices.
  direct_scaled_dot: direct_scaled_dot.
  auto_tuned_block: auto_tuned_block.
  gating_c_dev: gating_c_dev.
  twts: twts.
  max_err: max_err.
  bc: bc.
  bt_candidates: bt_candidates.
  bf_candidates: bf_candidates.
  btc_candidates: btc_candidates.
  bts_candidates: bts_candidates.
  token_candidates: token_candidates.
  interleave_bt: interleave_bt.
  routing_mode: routing_mode.
  use_fp8: use_fp8.
  _estimate_vmem_bytes_v2: _estimate_vmem_bytes_v2().
  make_deterministic_topk: make_deterministic_topk().
  make_hot_expert_topk: make_hot_expert_topk().
  w2_shared: w2_shared.
  w1_shared: w1_shared.
  repl_sharding: repl_sharding.
  repl_sharding2: repl_sharding2.
  local_nt: local_nt.
  padded_local_nt: padded_local_nt.
  aligned_local_nt: aligned_local_nt.
  w1s_sc: w1s_sc.
  w3s_sc: w3s_sc.
  cross_expert_prefetch_modes: cross_expert_prefetch_modes.
  make_replicated: make_replicated().
  block_configs_to_try: block_configs_to_try.
  nt0: nt0.
  use_shared_expert: use_shared_expert.
  w1_shared_scale: w1_shared_scale.
  padded_nt: padded_nt.
  d_avg: d_avg.
  w_avg: w_avg.
  ref_f32: ref_f32.
  k5: k5.
  parse_csv_int: parse_csv_int().
  enable_bt_scatter_overlap: enable_bt_scatter_overlap.
  _align_to: _align_to().
  w3_scale_s: w3_scale_s.
  w2_scale_s: w2_scale_s.
  w1_scale_s: w1_scale_s.
  quantize_shard_map: quantize_shard_map().
  w2_shared_scale: w2_shared_scale.
  results.results: results.results.
  xprefetch_mode: xprefetch_mode.
  dispatch_times: dispatch_times.
  _extract_durations_ms: _extract_durations_ms().
  devices: devices.
  bse: bse.
  SWIGLU_LIMIT: SWIGLU_LIMIT.
  SHARED_SWIGLU_LIMIT: SHARED_SWIGLU_LIMIT.
  tune_mode: tune_mode.
  enable_act_quant: enable_act_quant.
  _quant_repl_pc: _quant_repl_pc().
  topk_wts: topk_wts.
  i: i.
  dev: dev.
  by_tokens: by_tokens.
  align_local_tokens_for_v2: align_local_tokens_for_v2().
  warmup: warmup.
  iters: iters.
  _qbk_str: _qbk_str.
  use_wall: use_wall.
  use_split: use_split.
  timeit_fn: timeit_fn.
  key: key.
  gating_per_dev: gating_per_dev.
  seen_resolved_configs: seen_resolved_configs.
  k1: k1.
  t0: t0.
  KERNEL_NAME_RE: KERNEL_NAME_RE.
  TRACE_ROOT: TRACE_ROOT.
  _load_trace: _load_trace().
  _build_xprof_counter_options: _build_xprof_counter_options().
  wall_timeit: wall_timeit().
  split_timeit: split_timeit().
  parse_csv_str: parse_csv_str().
  parse_csv_int_or_none: parse_csv_int_or_none().
  parse_csv_bool: parse_csv_bool().
  check_correctness: check_correctness.
  valid_cross_expert_prefetch_modes: valid_cross_expert_prefetch_modes.
  _explicit_block_shape: _explicit_block_shape.
  _pow2_floor: _pow2_floor().
  _pow2_ceil: _pow2_ceil().
  _aligned_divisors: _aligned_divisors().
  k2: k2.
  k3: k3.
  k4: k4.
  ks1: ks1.
  ks2: ks2.
  w1s_q: w1s_q.
  w3s_q: w3s_q.
  w2s_q: w2s_q.
  bt: bt.
  bf: bf.
  btc: btc.
  best_tag: best_tag.
  best_avg: best_avg.
  _ladder_div2: _ladder_div2().
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py)

## Functions
- `_align_to(x, a)` — [`L345`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L345)
- `_aligned_divisors(n, alignment=8)` — [`L372`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L372) — All divisors of n that are multiples of alignment, descending.
- `_build_xprof_counter_options()` — [`L109`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L109) — Periodic TPU TensorCore perf-counter sampling for xprof (notebook-style).
- `_estimate_vmem_bytes_v2(*, bt, bf, btc, bse, bts, hidden_size, intermediate_size, num_experts, top_k, ep_size, num_tokens, use_fp8=False, quant_block_k=128, direct_scaled_dot=True, interleave_bt=True, enable_bt_scatter_overlap=True, verbose=False)` — [`L387`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L387)
- `_extract_durations_ms(trace: dict[str, Any])` — [`L76`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L76) — Extract per-iteration device durations for the v2 kernel from trace.
- `_ladder_div2(start)` — [`L361`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L361)
- `_load_trace(trace_root: str)` — [`L58`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L58)
- `_pow2_ceil(x)` — [`L355`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L355)
- `_pow2_floor(x)` — [`L349`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L349)
- `_quant_repl_pc(w)` — [`L885`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L885)
- `_repl(x)` — [`L893`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L893)
- `align_local_tokens_for_v2(local_num_tokens: int)` — [`L238`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L238)
- `generate_tune_candidates(intermediate_size, hidden_size, local_num_tokens, ep_size, num_experts, top_k, *, use_fp8=False, quant_block_k=128, direct_scaled_dot=True, interleave_bt=True, enable_bt_scatter_overlap=True, vmem_budget=64 * 1024 * 1024, vmem_headroom=0.95, max_configs=48, bse=256, use_shared_expert=False, se_inter=0, verbose=False)` — [`L536`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L536) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `log(msg)` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L49)
- `make_deterministic_topk(num_tokens, top_k, num_experts)` — [`L748`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L748)
- `make_hot_expert_topk(num_tokens, top_k, num_experts, hot_frac=0.1, hot_load=0.7)` — [`L774`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L774) — Hot-expert routing: hot_load fraction of (token, k) slots route to
- `make_replicated(rng_key, shape, dtype, scale=1)` — [`L815`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L815)
- `make_sharded(rng_key, shape, dtype, scale=1)` — [`L735`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L735)
- `parse_csv_bool(env_key: str, default: list[bool])` — [`L222`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L222)
- `parse_csv_int(env_key: str, default: list[int])` — [`L201`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L201)
- `parse_csv_int_or_none(env_key: str)` — [`L215`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L215)
- `parse_csv_str(env_key: str, default: list[str])` — [`L208`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L208)
- `quantize_shard_map(w)` — [`L841`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L841)
- `run_fn()` — [`L1059`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1059) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `split_timeit(run_fn, warmup: int, iters: int)` — [`L178`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L178) — Measure dispatch and block_until_ready separately.
- `trace_timeit(run_fn, warmup: int, iters: int)` — [`L135`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L135) — Warmup then profile *iters* calls, return per-iter device durations (ms).
- `wall_timeit(run_fn, warmup: int, iters: int)` — [`L165`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L165)

## Module values
- `E` — [`L267`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L267)
- `KERNEL_NAME_RE` — [`L45`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L45)
- `P` — [`L259`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L259)
- `SHARED_SWIGLU_LIMIT` — [`L279`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L279)
- `SWIGLU_LIMIT` — [`L276`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L276)
- `TRACE_ROOT` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L46)
- `_default_cfgs` — [`L935`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L935)
- `_explicit_block_shape` — [`L339`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L339)
- `_qbk_str` — [`L285`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L285)
- `aligned_local_nt` — [`L1027`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1027)
- `auto_tuned_block` — [`L342`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L342)
- `avg` — [`L1113`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1113) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `bc` — [`L1013`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1013)
- `bc0` — [`L1145`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1145) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `bc_resolved` — [`L1033`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1033) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `bench_tol` — [`L1230`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1230)
- `best_avg` — [`L1133`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1133)
- `best_tag` — [`L1133`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1133)
- `best_times` — [`L1133`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1133) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `bf` — [`L1017`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1017)
- `bf_candidates` — [`L329`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L329)
- `block_configs_to_try` — [`L933`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L933)
- `bse` — [`L270`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L270)
- `bt` — [`L1017`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1017)
- `bt_candidates` — [`L328`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L328)
- `btc` — [`L1017`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1017)
- `btc_candidates` — [`L330`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L330)
- `bts` — [`L1017`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1017)
- `bts_candidates` — [`L331`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L331)
- `by_tokens` — [`L1128`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1128)
- `check_correctness` — [`L275`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L275)
- `configs_to_try` — [`L1002`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1002)
- `cross_expert_prefetch_modes` — [`L293`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L293)
- `d` — [`L265`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L265)
- `d_avg` — [`L1093`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1093)
- `dev` — [`L985`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L985)
- `devices` — [`L261`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L261)
- `direct_scaled_dot` — [`L290`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L290)
- `dispatch_times` — [`L1091`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1091)
- `e` — [`L960`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L960) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `enable_act_quant` — [`L291`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L291)
- `enable_bt_scatter_overlap` — [`L292`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L292)
- `entries` — [`L1132`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1132)
- `ep_sharding` — [`L721`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L721)
- `ep_size` — [`L263`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L263)
- `f` — [`L266`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L266)
- `gating` — [`L993`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L993)
- `gating_c` — [`L1164`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1164)
- `gating_c_dev` — [`L1155`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1155)
- `gating_c_local` — [`L1154`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1154)
- `gating_local_shape` — [`L983`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L983)
- `gating_per_dev` — [`L984`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L984)
- `i` — [`L985`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L985)
- `interleave_bt` — [`L1015`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1015)
- `interleave_bt_modes` — [`L294`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L294)
- `invalid_modes` — [`L299`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L299)
- `iters` — [`L274`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L274)
- `k1` — [`L732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L732)
- `k2` — [`L732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L732)
- `k3` — [`L732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L732)
- `k4` — [`L732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L732)
- `k5` — [`L732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L732)
- `key` — [`L731`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L731)
- `ks1` — [`L822`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L822)
- `ks2` — [`L822`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L822)
- `ks3` — [`L822`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L822)
- `local_nt` — [`L914`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L914)
- `local_nt_raw` — [`L1026`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1026)
- `max_err` — [`L1227`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1227)
- `mesh` — [`L262`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L262)
- `nt` — [`L1129`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1129)
- `nt0` — [`L1152`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1152)
- `num_devices` — [`L260`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L260)
- `num_tokens` — [`L910`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L910)
- `pad_local` — [`L1028`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1028)
- `padded_local_nt` — [`L915`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L915)
- `padded_nt` — [`L1025`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1025)
- `qbk_arg` — [`L828`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L828)
- `quant_block_k` — [`L286`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L286)
- `ref` — [`L1213`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1213) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `ref_f32` — [`L1226`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1226)
- `ref_kwargs` — [`L1199`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1199) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `rel_err` — [`L1228`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1228)
- `repl_sharding` — [`L813`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L813)
- `repl_sharding2` — [`L883`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L883)
- `resolved_key` — [`L1045`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1045)
- `result` — [`L1172`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1172) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `result_f32` — [`L1223`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1223)
- `results` — [`L908`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L908)
- `routing_mode` — [`L269`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L269)
- `se_inter` — [`L272`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L272)
- `seen_resolved_configs` — [`L1010`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1010)
- `shard_key` — [`L986`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L986)
- `sk` — [`L1157`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1157)
- `t0` — [`L44`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L44)
- `tag` — [`L1018`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1018)
- `tag_resolved` — [`L1038`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1038)
- `tidx` — [`L1169`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1169)
- `timeit_fn` — [`L314`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L314)
- `times` — [`L1110`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1110)
- `timing_label` — [`L315`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L315)
- `token_candidates` — [`L332`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L332)
- `tokens` — [`L977`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L977)
- `tokens_c` — [`L1153`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1153)
- `top_k` — [`L268`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L268)
- `topk_idx` — [`L979`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L979)
- `topk_logits` — [`L999`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L999)
- `topk_wts` — [`L979`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L979)
- `tune_configs` — [`L916`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L916)
- `tune_mode` — [`L287`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L287)
- `tuned_cfg` — [`L944`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L944)
- `twts` — [`L1170`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1170)
- `use_fp8` — [`L284`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L284)
- `use_shared_expert` — [`L271`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L271)
- `use_split` — [`L289`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L289)
- `use_wall` — [`L288`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L288)
- `valid_cross_expert_prefetch_modes` — [`L298`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L298)
- `valid_routing_modes` — [`L307`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L307)
- `w1` — [`L805`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L805)
- `w1_scale_s` — [`L827`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L827)
- `w1_shared` — [`L810`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L810)
- `w1_shared_scale` — [`L880`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L880)
- `w1s_q` — [`L897`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L897)
- `w1s_sc` — [`L897`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L897)
- `w2` — [`L806`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L806)
- `w2_scale_s` — [`L827`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L827)
- `w2_shared` — [`L810`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L810)
- `w2_shared_scale` — [`L880`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L880)
- `w2s_q` — [`L899`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L899)
- `w2s_sc` — [`L899`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L899)
- `w3` — [`L807`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L807)
- `w3_scale_s` — [`L827`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L827)
- `w3_shared` — [`L810`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L810)
- `w3_shared_scale` — [`L880`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L880)
- `w3s_q` — [`L898`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L898)
- `w3s_sc` — [`L898`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L898)
- `w_avg` — [`L1094`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1094)
- `wait_times` — [`L1091`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1091)
- `wall_avg` — [`L1095`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1095) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_v2.md)
- `warmup` — [`L273`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L273)
- `xprefetch_mode` — [`L1014`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_v2.py#L1014)

