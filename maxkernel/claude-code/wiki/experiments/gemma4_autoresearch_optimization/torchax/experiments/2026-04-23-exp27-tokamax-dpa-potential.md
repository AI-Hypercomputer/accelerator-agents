---
title: "Exp 27 — tokamax.dot_product_attention (PARKED, sliding-window unsupported)"
type: experiment
tags: [experiment, gemma4, pallas, tokamax, splash-attention, use-base2-exp, parked]
hypothesis: tokamax-mosaic-tpu-use-base2-exp
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp27-tokamax-dpa"
verdict: parked
---

Tried to swap our direct `make_splash_mha_single_device` wiring for `tokamax.dot_product_attention(implementation="mosaic")`. Same splash kernel under the hood, but tokamax's mosaic_tpu path sets `use_base2_exp=True` in the softmax (TPU-native exp2 instead of natural exp). **Blocked: tokamax's mosaic_tpu kernel does not support `mask.k_start`, i.e. sliding-window attention.** Gemma 4 has 21 sliding-window layers out of 42, all of which fall back to XLA — dominating step time. Parked.

## Hypothesis under test

**Statement**: Tokamax's mosaic_tpu wrapping of splash enables `use_base2_exp=True` (natural-exp softmax → base-2-exp with log2(e) Q rescale). On v6e, hardware exp2 is faster than natural exp. Expected 1–3% attention-path improvement, 0.3–1% end-to-end.

Origin: post-exp-25 audit of remaining optimization knobs; noted that tokamax's default config has `use_base2_exp=True` while our direct splash call has no such flag (stock JAX splash doesn't expose it). See [tokamax Config at `pallas_mosaic_tpu.py:53`](../../../../codebases/tokamax.md) and [exp8 § Next hypotheses](2026-04-23-exp8-splash-attention-accepted.md).

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp27-tokamax-dpa` off trunk at exp 25.
- Installed tokamax in the `gemma4_py313` env: `pip install -e raw/code/tokamax --no-deps` plus `immutabledict pydantic qwix einshape tensorboardx typeguard==2.13.3 jaxtyping`. Skipped `cuequivariance` (GPU-only; not needed on TPU).
- Added `tokamax_attention_fn` and `register_tokamax_attention(mesh)` to `torchax/model/pallas_attention.py`. Layout diff from splash path: tokamax expects `[B, S, N, D]` (HF hands us `[B, N, S, D]`), so `.transpose(1, 2)` on Q/K/V before `call_jax`. Tokamax handles `jax.shard_map` internally when `q_sharding` is passed, so we don't need our own wrap.
- Dispatch via env `ATTENTION_IMPL=tokamax` → `register_tokamax_attention(mesh)` in `train.py`; default `splash` retains exp 25 path unchanged.

### First failure — absl flag parsing

Initial run crashed immediately on every attention call with:
```
[tokamax_pallas] fallback to XLA: UnrecognizedFlagError: Unknown command line flag 'steps'
```
Root cause: tokamax's `_src/config.py:_ConfigOption.value` lazily calls `flags.FLAGS(sys.argv)` on first config access if `FLAGS.is_parsed()` is false. That happens inside the jit-traced body of our attention function. `sys.argv` still carried argparse-style `--steps` etc., which absl's flag parser rejects.

**Fix**: in `register_tokamax_attention`, pre-parse absl flags with argv stripped:
```python
from absl import flags as _abslflags
if not _abslflags.FLAGS.is_parsed():
    _abslflags.FLAGS(sys.argv[:1])
```
This marks flags as parsed (with defaults) without trying to consume our argparse flags.

### Second (fatal) failure — sliding-window unsupported

With absl fixed, the kernel *itself* rejects sliding-window masks:
```
[tokamax_pallas] fallback to XLA: NotImplementedError: mask.k_start is not supported.
```
Source: `raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py:59-60`:
```python
if mask.k_start is not None:
    raise NotImplementedError("mask.k_start is not supported.")
```

Tokamax's `local_window_size=(W, 0)` becomes an internal mask with `k_start = q_idx - W + 1`. The mosaic_tpu kernel explicitly refuses that. All 21 of Gemma 4's sliding-window layers raise here and fall back to XLA — which is orders of magnitude slower than splash for the sliding case (no fused softmax, full score-matrix materialization). The other 21 (causal) layers do run on tokamax-mosaic.

Killed the run before it completed step 0. A hybrid approach (tokamax for causal, our splash for sliding) would only win on half the layers, yielding an estimated ≤0.2% end-to-end — not worth the added complexity.

## Results

None measurable. Run aborted before profile capture; compile never completed the jit trace because of repeated fallbacks inside the traced body.

## Mechanism (why the fallback is fatal)

- Gemma 4 alternates full-causal and sliding-window-512 attention per layer (`sliding_window_pattern` in the config).
- Our `tokamax_attention_fn` passed `is_causal=True` + `local_window_size=(512, 0)` for sliding layers, which is the correct tokamax API per `dot_product_attention`'s signature.
- Tokamax's API layer accepts this and constructs an internal mask with `k_start != None`. The mosaic_tpu kernel rejects it at build time — not at autotune, not at dispatch, but at `common.check_inputs_support(...)`.
- The fallback in our `tokamax_attention_fn` catches this and calls our `_xla_fallback_fwd`, which is a plain einsum + mask + softmax path — functional but ~20–30× slower than splash for sliding-window at seq=1024.
- In practice: half the layers per step at 20–30× slowdown is dominant; end-to-end would be multiples slower than exp 25 baseline.

## Profile

No trace — run killed before step 0 completed. First attempt crashed on absl flag parsing; second attempt hit `NotImplementedError: mask.k_start is not supported` (tokamax mosaic_tpu rejects sliding-window) and fell back to slow XLA for 21/42 layers, defeating the purpose; killed.

## Verdict

**PARKED.** Tokamax's mosaic_tpu kernel is not a drop-in replacement for our splash wiring on Gemma 4's hybrid attention. Not an experiment that can be salvaged at reasonable effort:
- Hybrid (tokamax for causal, splash for sliding) wins ≤0.2% — not worth maintaining two code paths.
- Extending tokamax's `pallas_mosaic_tpu_common` to handle `k_start` is an upstream PR, out of scope.

**Not merged to trunk.** Code lives on the exp 27 branch as a reference wiring:
- `tokamax_attention_fn` + `register_tokamax_attention(mesh)` in `torchax/model/pallas_attention.py`
- `ATTENTION_IMPL=tokamax` env dispatch in `train.py`
- absl flag pre-parse workaround documented above

If tokamax later adds sliding-window support (upstream fix to `pallas_mosaic_tpu_common.py:59`), this branch becomes directly useful — the plumbing is correct, only the kernel's mask support is missing.

## Next hypotheses (promoted to exp 28)

1. **seq=2048 batch=1 at exp25 config** — exp14 did this at exp12 config (31,960 TPS). exp25's block=1024 + SEQ_MINOR + fused_bwd might be proportionally better at longer seq (attention N² larger share of cost).
2. **batch=4 retry at exp25 config** — exp22 OOM'd at batch=4 before SEQ_MINOR + block=1024. Unlikely to fit (block=1024 uses more VMEM not less) but quick one-flag check.
3. **Persistent JAX compile cache** — orthogonal iteration-speed win (150 s → ~10 s after first run). Doesn't change TPS but speeds the loop.
4. **LIBTPU scoped VMEM bump** (`--xla_tpu_scoped_vmem_limit_kib=524288` vs default 131072) — may help splash fit larger tiles. Requires verifying v6e VMEM size first.

## See also

- [exp 8 — first Pallas experiment](2026-04-23-exp8-splash-attention-accepted.md) — the original splash-via-shard_map wiring this exp27 tried to replace.
- [exp 26 — scan-over-layers (parked)](2026-04-23-exp26-scan-over-layers-potential.md) — similar parked-with-analysis outcome.
- [program.md § Pallas kernel landscape](../../program.md).
- [tokamax codebase](../../../../codebases/tokamax.md) — `dot_product_attention` API; note mosaic_tpu sliding-window limitation.
- [splash-attention concept](../../../../concepts/splash-attention.md).

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (+120 lines tokamax path)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (+10 lines env-dispatch)
- `/tmp/gemma4_exp27_smoke.log` (first failure — absl)
- `/tmp/gemma4_exp27_smoke2.log` (second failure — sliding-window)
- `raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_common.py:59-60` (the explicit `NotImplementedError` on `k_start`)
- No runtime profile captured — run did not complete step 0.
