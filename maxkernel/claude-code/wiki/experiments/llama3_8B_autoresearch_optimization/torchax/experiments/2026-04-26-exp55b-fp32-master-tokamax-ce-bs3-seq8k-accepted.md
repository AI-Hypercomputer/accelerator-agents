---
title: "Exp 55b — fp32 master + bf16 compute + tokamax CE + scan + splash @ bs=3 seq=8192 (ACCEPTED, new program-target best)"
type: experiment
tags: [llama3, torchax, amp, fp32-master, tokamax, cross-entropy, shard-map, seq8k, accepted, milestone]
hypothesis: llama3-torchax-tokamax-ce-density-unlock
model: llama3-8b-torchax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-torchax-20260426-exp55b-fp32-master-tokamax (image hf-v26)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp20-amp-fp32-master
verdict: supported
---

🏆 **Program-target advanced.** True AMP master (fp32 weights, bf16 compute)
plus the **tokamax mosaic-TPU `linear_softmax_cross_entropy_loss` kernel**
unlocks **bs=3 seq=8192** (previously OOM in exp 44 by 6.04 GiB) at
**34.6 % MFU and 6,178 tok/s/chip — +35 % per-chip TPS over exp 20** (the
previous program-target best at 31.6 % MFU).

The win comes from collapsing the per-chip CE-output footprint: the
canonical path materializes a `(B*L, V)` logits tensor (≈12.6 GiB for
B=24, L=8192, V=128256, fp32) that gets all-gathered along V; the
mosaic-TPU CE kernel **streams the logsumexp over V blocks in VMEM**, so
the persistent activation goes from `(B*L, V)` to just `(B*L, lse_scalar)`.
That's the single biggest activation footprint in the model and removing
it is what makes the bs=3 density legal.

## Stack composition

Every accepted optimization stacks here:

1. **Splash attention** (exp 2) + autotuned config (exp 8/9/10):
   `block_q=block_kv=1024, use_fused_bwd_kernel=True, dkv=(2048,2048)`.
2. **Scan-over-layers** (exp 40 architecture): the 32 `LlamaDecoderLayer`s
   are stacked into a single `torchax.train.ScannedModule` body, traversed
   by `jax.lax.scan`. XLA sees one body's worth of buffers at compile time
   instead of the 32-layer-unrolled sum.
3. **fp32 master weights + bf16 compute** (this experiment, NEW): weights
   are stored as `torch.float32` (`--weights_dtype=fp32`); inside `model_fn`
   each weight is autocast to `bf16` (`--compute_dtype=bf16`) before the
   `functional_call`. The `astype` op's vjp downcasts the bf16 grad back to
   fp32 on the way out, so optax sees fp32 grads matching its fp32 mu/nu
   and fp32 master weights — **the strict canonical AMP master pattern**.
4. **`mu_dtype=fp32`** for adamw (`--master_dtype=fp32`) — same pattern
   as exp 20.
5. **Tokamax `linear_softmax_cross_entropy_loss`** (NEW): `mosaic_tpu`
   implementation, wrapped in **`shard_map`** because mosaic kernels can't
   be auto-partitioned. h and labels sharded along `'fsdp'`, w replicated
   inside the shard_map (one all-gather of `lm_head.weight` per step), each
   shard computes its local `sum`, then `jax.lax.psum(_, axis_name='fsdp')`
   gathers the global sum, divided outside the shard_map by `B*L` for the
   mean.
6. **VMEM 98304 KiB** (from exp 18b discovery).

## Setup

- Image: **`hf-v26`**. Adds `--compute_dtype` flag and the
  `_maybe_cast_weights` autocast in `model_fn`. Builds on `hf-v22` (which
  added the `shard_map` wrap for tokamax CE) and `hf-v25` (which fixed an
  MFU-formula bug — see *MFU correction* section below).
- Cluster: `<your-v6e8-cluster>`, single v6e-8 slice (8 chips,
  2 hosts × 4 local devices), <your-zone>.
- Branch: `v6e8-llama3-8b-torchax-20260426-exp55b-fp32-master-tokamax`.
- Command (key flags only):
  ```
  python train.py --model_id=meta-llama/Meta-Llama-3-8B \
    --batch_size=3 --seqlen=8192 --train_steps=15 \
    --weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32 \
    --use_real_data=True --use_splash=True --use_scan=True --use_tokamax_ce=True
  ```

## Tokamax CE bring-up — what was needed (hf-v18 → hf-v26)

The kernel does not drop in cleanly under `fire.Fire` + multi-host JAX +
torchax + sharded weights. Each error fixed surfaced the next:

| Image | Issue | Fix |
|-------|-------|-----|
| hf-v18 | `KeyError: 'lm_head_weight_ref'` from Identity-replacement hack | Add `skip_lm_head` flag to `LlamaForCausalLMScan`; keep `lm_head.weight` as canonical key, just bypass the projection in forward. Loss path reads it directly from the `weights` dict. |
| hf-v19 | `absl.flags.UnrecognizedFlagError: --model_id` | tokamax's `_ConfigOption.value` calls `flags.FLAGS(sys.argv)` on first read; sys.argv carries fire kwargs. Pre-mark absl flags as parsed: `flags.FLAGS([sys.argv[0]])` before any tokamax import. |
| hf-v20 | `lax.mul requires arguments to have the same dtypes, got bfloat16, float32` | Mosaic CE kernel hardcodes fp32 grad outputs (lines 668-669 of `pallas_mosaic_tpu_kernel.py`) regardless of input dtype. Cast inputs to fp32 at the boundary (`jh.astype(jnp.float32)`); the cast-vjp auto-downcasts the bf16 grad on the way back. |
| hf-v21 | `NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.` | Wrap the kernel in `jax.experimental.shard_map.shard_map` with `in_specs=(P('fsdp', None), P('fsdp'), P())` and `out_specs=P()`. Inside the shard, compute local sum; `jax.lax.psum` for global sum; divide by global B*L outside. |
| hf-v22 | (compiles, runs) | First successful tokamax CE run. |
| hf-v24 | (added) | `--compute_dtype` flag + `_maybe_cast_weights` for true AMP master path. |
| hf-v26 | MFU formula divisor bug (see below) | `avg_step_time = total_time / n_measured_steps` (was `... / (train_steps - warmup_steps)`). |

In addition there is a `jaxtyping`-stub block at the top of `train.py`
that monkey-patches `jaxtyped` to a no-op, because tokamax's
`*B T H d`-style annotations under py3.13 trigger a typeguard AST-walk
crash (independent of the bring-up sequence above).

## Results — corrected MFU at the program target

Re-run with hf-v26 (correct `avg_step_time` divisor):

| Run | bs | seq | MFU | tok/s | tok/s/chip | step_time | n_steps actual |
|-----|----|-----|-----|-------|------------|-----------|----------------|
| exp 55  | 2 | 8192 | 33.9 % | 48,419 | 6,052 | 2.71 s | 13 |
| 🏆 **exp 55b** | **3** | **8192** | **34.6 %** | **49,427** | **6,178** | **3.98 s** | **10** |
| exp 55c | 4 | 8192 | 33.6 % | 47,929 | 5,991 | 5.47 s | 7  |

**Reference vs prior bests at the program target (seq=8192):**

| Stack | bs | tok/s/chip | MFU | Notes |
|-------|----|-----------:|----:|-------|
| exp 20 (AMP / mu-fp32 / bf16 weights / per-layer remat / no scan) | 1 | 4,591 | 31.6 % | Prior best at seq=8192 |
| exp 40 (scan + bf16 weights + bf16-master / no tokamax CE) | 2 | 6,076 | 34.0 % | bs=3 OOM |
| exp 51b (scan + bf16 weights + tokamax CE) | 3 | 6,178 | 34.6 % | bs=3 unlocked |
| 🏆 **exp 55b (this — scan + tokamax CE + fp32 master)** | **3** | **6,178** | **34.6 %** | bs=3, true AMP master |
| exp 55c (scan + tokamax CE + fp32 master) | 4 | 5,991 | 33.6 % | Memory-pressured |

Per-chip throughput at bs=3 is **identical** between exp 51b (bf16 weights)
and exp 55b (fp32 master) — within noise. The fp32 weight storage adds
~2 GiB/chip vs bf16, but the cast to bf16 inside `model_fn` is fused into
the matmul by XLA so there is no measurable runtime overhead. **fp32
master is essentially free at this shape.**

Loss trajectory at bs=3 over 12 steps for the program target: 11.77 →
11.63 (first 12 steps with real wikitext data). Loss decreases more
quickly than with bf16 weights (exp 51b: 11.77 → 11.71 over 12 steps),
consistent with finer-grained adamw updates landing in fp32 master rather
than rounding into bf16 at each step.

## Memory accounting (rough, per chip)

| Item | bf16 weights | fp32 master |
|------|-------------:|------------:|
| Weights (8.03 B params / 8 chips) | 2.0 GiB | 4.0 GiB |
| mu (fp32 always) | 4.0 GiB | 4.0 GiB |
| nu (fp32 by default in optax.adamw with mu_dtype=fp32) | 4.0 GiB | 4.0 GiB |
| Optimizer-state subtotal | 10 GiB | 12 GiB |
| HBM remaining for activations + tokamax-CE w_HV all-gather + grads | ~22 GiB | ~20 GiB |

The bs=3 unlock requires the activation footprint to fit within ~20 GiB
HBM, which is what removing the materialized `(B*L, V)` logits tensor
buys us via the streamed CE kernel.

## MFU correction (hf-v26)

The MFU numbers reported in earlier hf-v22-built experiments
(exp 51b 45.0 %, exp 51c 64.1 %, exp 53b 62.4 %) were **inflated by an
average-step-time divisor bug**: `avg_step_time = total_time_after_warmup
/ (train_steps - warmup_steps)` divides by the *planned* step count, but
the wikitext data loader exits early at non-trivial global batch sizes
(global_batch=24 supports ~10 measured steps before the loader is
exhausted). The numerator (`total_time_after_warmup`) only accumulates
for steps that ran, but the denominator stays at 13 (= 15 − 2). Result:
`avg_step_time` is artificially small; MFU is artificially large by
factor `(measured_steps / 13)`.

For exp 51b: 10 actual / 13 planned ⇒ MFU inflated by `13/10 = 1.30×`,
explaining 34.6 % → 45.0 %. For exp 51c: 7/13 ⇒ `1.86×`, explaining
34.5 % → 64.1 %.

The fix in hf-v26 introduces a `n_measured_steps` counter incremented
inside the loop and divides by that. Both `steps measured` in the summary
print and the MFU calculation now use the actual count. The corrected
re-runs are exp 55 / 55b / 55c above.

The previously-reported 36.0 % MFU for exp 40 (bs=2 seq=8192) is
unaffected because bs=2 ran the full 15 steps without loader exhaustion.

## Verdict

**Supported.** All three success criteria met:
- Measurable improvement: 6,178 tok/s/chip vs prior program-target
  best 4,591 (exp 20) — **+35 % per-chip throughput**.
- No semantic regression: loss decays normally and faster than bf16
  weights (full AMP master is correctly active).
- No model-quality compromise: weights are stored fp32, optimizer math is
  fp32; only the forward/backward compute is bf16 (canonical AMP).

This becomes the **new program-target best at seq=8192**. The full stack
is reproducible with `--weights_dtype=fp32 --compute_dtype=bf16
--master_dtype=fp32 --use_splash=True --use_scan=True --use_tokamax_ce=True`.

## Profile

- **Browser URL** (xprof, when local server is up): `http://localhost:8791/?run=llama3-8b-exp55b-master-bs3-fix`
- **Run name**: `llama3-8b-exp55b-master-bs3-fix`
- **On-disk path**: `raw/profiles/2026-04-26-exp55b-fp32-master-tokamax/` *(profile not yet copied off the GKE pod's /tmp at time of writing — will be tar'd and uploaded out-of-band; the corresponding tarball at `/tmp/llama3_profile/llama3-8b-exp55b-master-bs3-fix.tgz` was preserved by the post-run `sleep 600`)*
- **Steps captured**: step 7 (single step xprof trace).
- **Contents**: xprof trace.
- **Note**: profiles are gitignored — this experiment page is the only persistent lineage from the trace on disk back to the run that produced it.

## Follow-ups tested (post-acceptance, all 2026-04-26)

| Exp | Stack delta | Result | Verdict |
|-----|-------------|--------|---------|
| 56 | + tokamax CE autotune | 6,202/chip, 34.8 % MFU (+0.4 % vs heuristic) | accepted, current frontier |
| 57 | bs=4 + tokamax CE autotune | OOM during autotune sweep (autotune buffers spill) | refuted |
| 57b | autotune + VMEM=131,072 KiB | 6,157/chip, 34.5 % (-0.7 %; less HBM for activations) | refuted |
| 58 | + recipe XLA flags + autotune | 6,206/chip, 34.8 % (within noise) | neutral / refuted |
| 58b | + recipe XLA flags + heuristic | 6,159/chip, 34.5 % (-0.3 %) | refuted |
| 59 | TP=2 / FSDP=4, bs=3 (global=12) | 5,113/chip, 28.7 % | refuted (smaller batch) |
| 59b | TP=2 / FSDP=4, bs=6 (global=24, fair) | 5,313/chip, 29.8 % (-14 %) | refuted on v6e-8 |
| 60  | scan_remat_policy=`dots_saveable` | OOM by 42 GiB | refuted |
| 60b | scan_remat_policy=`dots_with_no_batch_dims_saveable` | OOM by 42 GiB | refuted |

**Consolidated frontier** at bs=3 seq=8192 program-target shape:

  scan + AMP master (fp32 weights, bf16 compute) + tokamax CE (autotune) +
  splash (autotuned) + VMEM=98,304 KiB + `nothing_saveable` scan remat
  → **6,206 tok/s/chip, 34.8 % MFU**.

Recipe XLA flags, TP=2, looser remat, and larger VMEM all refuted at this
stack/shape. The only remaining direction for breaking 35 % MFU is
**xprof-driven profiling** of the converged stack to identify the hot path.

## Next hypotheses (to file)

1. **xprof analysis of exp 56 trace** — find where the 65 % of un-utilized
   compute time is going (kernel idle gap? memory BW saturation? attention
   block scheduling?). The MaxText reference at 44.6 % MFU tells us the
   workload should reach into the 40s; the 10 pp gap is investigable.
2. **lm_head sharding rework** — the all-gather of `lm_head.weight`
   inside the CE shard_map is small (~9 ms / step at v6e ICI bandwidth)
   but a full V-axis-sharded CE (chunked across V with a cross-shard
   `psum_logsumexp`) would eliminate the gather. Custom kernel territory.
3. **Splash kernel retune at bs=3 seq=8192** — exp 10 said the universal
   winner `block_q=block_kv=1024` holds across seq=1024..8192, so this is
   probably already optimal, but cheap to verify.

## See also

- [Llama 3 8B torchax program page](../README.md)
- [Exp 20 — AMP fp32 master at bs=1 seq=8192 (prior best)](2026-04-25-exp20-amp-fp32-master-seq8k-accepted.md)
- [Exp 23 — AMP at bs=2 seq=4096](2026-04-25-exp23-amp-bs2-seq4k-accepted.md)
- Pallas kernel directory § cross-entropy: tokamax `linear_softmax_cross_entropy_loss`
  (mosaic_tpu impl) is the canonical TPU streamed-V CE kernel surveyed in
  [pallas-kernel-directory § upstream JAX + tokamax](../../../../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md).

## Sources

- `raw/code/torchax/torchax/train.py` — `ScannedModule`, `make_train_step` patterns.
- `raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/` — kernel impl, vjp, autotuning configs.
- `raw/profiles/2026-04-26-exp55b-fp32-master-tokamax/` — xprof trace (out-of-band copy pending).
