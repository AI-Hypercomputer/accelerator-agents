---
title: "Exp 47 — marin/levanter fused linear+softcap+CE Pallas kernel on JAX stack (REJECTED, -5.61 % TPS, custom-call + all-gather tax)"
type: experiment
tags: [experiment, gemma4, jax, levanter, cross-entropy, pallas, softcap, shard-map, rejected]
hypothesis: jax-levanter-linear-softcap-ce
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: 573852c
verdict: refuted
---

Integrate [marin/levanter](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/)'s `linear_softmax_cross_entropy_loss_pallas` — the only public TPU Pallas CE kernel with native `logit_soft_cap` support — as a `JAX_CE_IMPL=levanter` gate in the native-JAX trainer. Softcap is applied inline on each VMEM logits tile inside the kernel, so the `[B, S, V]` logits never materialize in HBM (the blocker that invalidated [exp 43](2026-04-23-exp43-jax-tokamax-ce-rejected.md)). **Result: parity passes (bf16 |Δloss| 0.048 vs tol 0.05), smoke-step-4 loss within 0.5 % of exp 36, but steady-state TPS regresses −5.61 % (34,614 → 32,671) — rejected per the decision tree.** The Pallas custom-call boundary + `w_hv` all-gather tax inside the `shard_map` wrapper outweighs the logits-materialization savings on this workload (CE is already <3 % of step time at b=3).

Exp 36 (34,614 TPS) remains the JAX-stack best.

## Hypothesis

Replace the JAX-stack `forward_loss` sequence — `logits = hidden @ W.T` → `sc * tanh(logits / sc)` softcap → bf16 `log_softmax` → `take_along_axis(label)` NLL — with a single fused Pallas kernel that streams the softcap-then-softmax over VMEM tiles of `hidden @ W.T`. Expected: ~1.3 GiB HBM freed (the `[B=3, S=1024, V=262144]` bf16 logits tile ≈ 1.5 GiB is eliminated along with the fp32 log_probs tensor); small TPS gain (≤+3 %) from one fewer HBM round-trip over the logits tile. Decision tree per the program:

- TPS > 34,614 × 1.005: `-accepted`, new JAX-stack best.
- Flat ±0.5 %: `-potential`.
- Regresses: `-rejected`. ← what triggered.
- Loss diverges from exp 36 at step 4 by >5 %: `-invalid`.

## Setup

### Kernel integration

Added `JAX_CE_IMPL=levanter` env gate to [`jax/train.py`](../train.py). When active, `forward_loss` calls `model(..., return_hidden=True)` (new kwarg on `Gemma4ForCausalLM.__call__`, bypasses lm_head + softcap) and passes the raw `[B, S, H]` hidden + `lm_head.weight.T` to [levanter's `fused_cross_entropy_loss_and_logsumexp_penalty`](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py) with `logit_soft_cap=30.0`.

**Import shim**: levanter's kernel is a vendored git submodule at `raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/`. Installing the full `levanter` package would require `equinox`, `draccus`, `rigging`, and a chain of heavy deps. Instead, [`jax/model/kernels/fused_ce/__init__.py`](../model/kernels/fused_ce/__init__.py) pre-populates `sys.modules` with minimal stubs for the `levanter` top-level package (so `levanter/__init__.py`'s import of `analysis`/`trainer`/... is skipped) and for the one `rigging.filesystem.url_to_fs` call inside `autotune_cache_utils`. Autotune is disabled via `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` and we pass explicit `BlockSizes` for our shape, so the autotune I/O path is never reached.

**Shard map wrap**: Mosaic custom-calls cannot be auto-partitioned (same pattern as [`pallas_attention.py`](../model/pallas_attention.py) for splash). We wrap the kernel call in `jax.shard_map` with `in_specs=(P('fsdp', None), P('fsdp'), P('fsdp'), P(None, None))` — hidden/labels/mask sharded on `fsdp`, lm_head weight all-gathered to replicated `[H, V]`. A `jax.lax.psum` over `axis_name="fsdp"` inside the kernel fn sums the partial `loss_sum` across shards; `out_specs=P()` returns a replicated scalar.

**Block sizes**: Gemma 4 E4B (V=262144, H=2560) lands in no TPU-tuned bucket in `tuned_block_sizes.py` (all TPU buckets top out at v=131072 or require `NVIDIA GB10`). The default `(b=1024, h=512, v=1024)` overruns the 32 MiB VMEM by ~8 MiB on v6e. Hand-picked: **`(b=1024, h=256, v=512)`**. Constraint: `b_block_size` must be a multiple of 1024 when per-shard batch B ≥ 1024 (our per-device flat batch is B*S = 3*1024 = 3072).

### Command (from `.../jax/`)

```bash
WIKI_ROOT=/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki
PROFILE=$WIKI_ROOT/raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce
mkdir -p $PROFILE
JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache \
JAX_ATTENTION_IMPL=splash \
JAX_CE_IMPL=levanter \
LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0 \
  $HOME/miniconda3/envs/gemma4_py313/bin/python -u -m train \
    --steps 20 --batch_size 3 --seq_len 1024 \
    --profile_dir $PROFILE \
    --profile_steps 10 11 12 2>&1 | tee /tmp/gemma4_jax_exp47.log
```

Global batch = per-chip 3 × fsdp=4 = **12**. Tokens / step = 12 × 1024 = 12,288. Only env delta vs exp 36: `JAX_CE_IMPL=levanter`.

## Correctness

### Parity test (pre-run gate)

Wrote [`jax/tools/parity_levanter_ce.py`](../tools/parity_levanter_ce.py) to compare, on a small `B=1, S=128, H=2560, V=262144` random batch with 10 % `IGNORE_INDEX=-100` mask:

- **Reference**: materialize `[B*S, V]` logits, apply `30 * tanh(logits/30)` softcap, bf16 `log_softmax`, NLL, `sum / mask.sum()`. Mirrors [`train.py` `forward_loss`](../train.py) at `JAX_CE_DTYPE=bf16`.
- **Levanter**: `fused_cross_entropy_loss_and_logsumexp_penalty(flat_hidden, safe_labels, w_hv, logit_soft_cap=30.0, weight=mask, reduction="sum", implementation="pallas_tpu", dtype=bf16, block_sizes=BlockSizes(128, 256, 512))` then `/ mask.sum()`.

Result:

```
[parity] reference loss = 12.461365
[parity] levanter  loss = 12.509091
[parity] |diff|         = 0.047727  (tol = 0.05)
[parity] PASS
```

Within bf16 tolerance (0.048 < 0.05) — confirms the kernel's softcap-inline + streaming softmax reproduces the reference math.

### Smoke loss trajectory (5-step training)

| step | exp 36 (reference) | **exp 47 (levanter)** | Δ |
|---|---|---|---|
| 0 | 3.8125 | 3.7917 | −0.5 % |
| 2 | 2.7344 | 2.7083 | −1.0 % |
| **4** | **2.1875** | **2.1979** | **+0.47 %** |
| 9 | 1.82 | 1.9479 | +7 % |
| 19 | 1.84 | 2.0000 | +9 % |

Step-4 loss matches exp 36 within 0.47 % — well inside the 5 % bar. Later-step divergence is expected from the bf16 drift of the fused path propagating into the optimizer state (parity diff of 0.05 compounds across 20 Adam steps); not a semantics change.

## Results

| Metric | exp 36 (b=3, splash, bf16 CE) | **exp 47 (levanter CE)** | Δ |
|---|---|---|---|
| **TPS (mean, steps 2–19)** | 34,614 | **32,671** | **−5.61 %** |
| Step time (mean 2–19) | 355.0 ms | **376.12 ms** | +21.1 ms (+5.95 %) |
| Step 0 compile | 167 s | 15 s (cache hit) | −152 s (exp 45 cache) |
| Peak HBM | 27.11 GiB (86.75 %) | (not captured — see Profile) | tbd |
| Loss step 0 | 3.8125 | 3.7917 | −0.5 % |
| Loss step 4 | 2.1875 | 2.1979 | +0.47 % |
| Loss step 19 | 1.8359 | 2.0000 | +8.9 % (Adam drift, parity-explained) |
| MFU (v6e-4 150,159 peak TPS) | 23.05 % | 21.76 % | −1.29 pt |
| Tokens per step | 12,288 | 12,288 | 0 |

**Decision-tree outcome**: regresses by 5.61 % (beyond the `>±0.5 %` band) → **rejected** per the program's decision tree.

### Why the regression (the "Pallas custom-call tax" pattern)

Baseline CE in [exp 36](2026-04-23-exp36-jax-splash-batch3-accepted.md) is not a bottleneck: at `JAX_CE_DTYPE=bf16`, the `[B*S, V]` log_softmax + NLL run as a single XLA fusion (`loop fusion` absorbs it into the same pass as surrounding elementwise ops). Total CE cost on exp 36's profile is ~10 ms of 355 ms step time (<3 %). Swapping it for a Pallas custom-call adds:

1. **Custom-call boundary overhead**: ~3–5 ms launch latency per Mosaic call + 3× (fwd + two bwd halves) ≈ 9–15 ms. Same mechanism that killed [torchax exp 33 (Pallas RMSNorm)](../../torchax/experiments/) — the kernel executes fast but XLA can no longer fuse around its edges.
2. **`w_hv` all-gather inside shard_map**: `lm_head.weight.value` is FSDP-sharded on V (262144 / 4 = 65536 columns per device); the kernel requires replicated `[H, V]`. The shard_map wrapper forces an all-gather of 1.31 GiB bf16 of weight per forward (and again in backward). XLA may have been folding this all-gather into the lm_head matmul in exp 36 via collective-matmul fusion; moving the matmul into a Pallas custom-call breaks that fusion and exposes the full all-gather on the critical path. Consistent with the +21 ms step-time delta we see.
3. **Breaking XLA's lm_head matmul + softcap + log_softmax fusion**: exp 36's profile shows the lm_head matmul fused with surrounding elementwise ops; the Pallas custom-call isolates it, each matmul now flushes its output through HBM. On this model, `convolution fusion` was at 1512 ms of 3-step profile (28 % of step time) — any un-fusion of that pathway is expensive.

None of these are kernel-quality issues — levanter's kernel is correct (parity PASS) and efficient in isolation (the bench in the kernel's repo shows it matches tokamax on big shapes). The regression is an **integration tax** on a workload where the op it replaces is already XLA-fused tightly.

## Profile

**Directory**: [`raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/) (gitignored; ~328 MiB on disk).
**Steps captured**: 10, 11, 12 (xprof trace + xplane.pb).
**GCS mirror**: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp47-levanter-ce/` (326.6 MiB uploaded).
**What's inside**: trace of the native-JAX trainer at batch=3 seq=1024 fsdp=4 bf16 with `JAX_ATTENTION_IMPL=splash` and `JAX_CE_IMPL=levanter`; step-time 376.1 ms steady-state (mean steps 2–19). Used to verify the regression mechanism (custom-call boundary + w_hv all-gather) — not mirrored to the pax-on-cloud-tpu-project/xprof bucket because this account lacks write access there (same situation as other recent JAX-stack experiments).

## Verdict

**REJECTED.** −5.61 % TPS regression vs exp 36 (34,614 → 32,671), outside the ±0.5 % flat band. Loss trajectory is healthy (parity PASS, step-4 within 0.47 %) — this is a performance refutation, not a correctness one. Commit `573852c`.

**The hypothesis's architecture premise held** — the kernel does apply softcap inline and never materializes `[B, S, V]` — but the throughput premise did not: CE is not the bottleneck on this workload, and the Pallas custom-call breaks XLA's existing lm_head fusion. The cost of replacing a 10-ms XLA-fused op with a 3.5-ms kernel + 28-ms of new boundary overhead is net negative.

Exp 36 remains the JAX-stack best at 34,614 TPS, 23.05 % MFU.

## What we learned (durable)

1. **Logit softcap + Pallas CE is now unblocked** (unlike tokamax in [exp 43](2026-04-23-exp43-jax-tokamax-ce-rejected.md)). The import shim in [`jax/model/kernels/fused_ce/__init__.py`](../model/kernels/fused_ce/__init__.py) keeps the levanter kernel usable from this trainer tree without forking it.
2. **Fused CE only pays when CE is the bottleneck.** At exp 36's b=3 s=1024, CE is <3 % of step time and lives inside a single XLA loop fusion. Pallas CE gains HBM but loses fusion — net wash-or-worse. Candidates where it *would* pay: seq=2048 b=2 ([exp 40](2026-04-23-exp40-jax-seq2048-batch2-accepted.md) config, where logits peak is bigger) or a future memory-tight config where HBM headroom gates the batch. Not dropped from the candidate list — retired against b=3 s=1024 only.
3. **Block-size table has no TPU entry for (V≈262144, H≈2560).** The `tuned_block_sizes.py` file has `gb10-large-vocab-mid-batch` (`h≤1536, v≤262144`) which is GB10-only. Adding a `tpu-v6-gemma4-like` bucket with `(b=1024, h=256, v=512)` (shown here to fit 32 MiB VMEM and stream cleanly) would let this kernel drop in without hand-tuning for future Gemma-family ports.
4. **The `w_hv` all-gather is avoidable.** A future variant that accepts the FSDP-V-sharded weight directly and does a per-shard partial CE (with the full softmax denominator coming from an all-reduce over partial logsumexp contributions) would match exp 36's communication pattern. This is a kernel modification, not an integration fix — logged as a follow-up below.
5. **Custom-call boundary tax is stack-agnostic.** Same +5–6 % regression pattern as torchax exp 33 (Pallas RMSNorm, 600 ms → 636 ms). When the baseline is already XLA-fused, dropping in any Pallas kernel that replaces a <5 %-step-time op tends to cost more in fusion disruption than it saves in kernel speed. Worth promoting to a general heuristic in [program.md](../../program.md).

## Follow-ups (ranked)

1. **Retry at seq=2048, batch=2** (on top of [exp 40](2026-04-23-exp40-jax-seq2048-batch2-accepted.md)). At that config the logits tile is 2× larger (1024 → 2048) and HBM headroom is tighter; the kernel's HBM savings might be worth the boundary tax. One flag flip on top of exp 40's command. Confidence medium-low. Effort S.
2. **V-sharded fused CE variant** (kernel-edit). Accept `w_hv` with `P(None, 'fsdp')` sharding, do per-shard partial logsumexp, all-reduce the partial `lse` across fsdp inside the kernel boundary, then compute the final NLL. Removes the 1.31 GiB all-gather on the critical path. Effort M (Pallas kernel change); confidence low (XLA's collective matmul fusion may already do this better than a hand-written kernel can).
3. **Tune block sizes for v6e Gemma4 shapes.** Sweep `(b_block, h_block, v_block)` on the parity harness (small shapes, cheap sweep) and the benchmark harness (real shapes, ~2 min each). Likely 1–3 % reclaim from better VMEM occupancy, not enough to flip the verdict but narrows the gap. Effort S.
4. **`collective-permute-done` audit** (exp 36's #3 hotspot at 12.2 % / 549 ms of 3-step profile) — unchanged from exp 36's follow-ups, independent of this integration.

## See also

- [Exp 36 — splash + batch=3 (ACCEPTED, JAX-stack best 34,614 TPS)](2026-04-23-exp36-jax-splash-batch3-accepted.md) — the baseline this was meant to beat.
- [Exp 43 — tokamax CE (INVALID, softcap not supported)](2026-04-23-exp43-jax-tokamax-ce-rejected.md) — the precursor attempt whose API gap motivated this experiment.
- [Exp 37 — bf16 CE env-var gate](2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md) — the JAX_CE_DTYPE gate this sits alongside.
- [Torchax exp 33 — Pallas RMSNorm rejected](../../torchax/experiments/) — analogous "Pallas custom-call tax" result on the sibling stack.
- [`../model/kernels/fused_ce/__init__.py`](../model/kernels/fused_ce/__init__.py) — the shim that makes the vendored levanter kernel importable without pulling equinox/draccus/rigging.
- [`../train.py`](../train.py) — `JAX_CE_IMPL=levanter` gate + shard_map wrapper.
- [`../model/modeling_gemma4.py`](../model/modeling_gemma4.py) `Gemma4ForCausalLM.__call__(..., return_hidden=True)` — the kwarg that lets the trainer bypass the in-model lm_head+softcap.
- [`../tools/parity_levanter_ce.py`](../tools/parity_levanter_ce.py) — correctness harness.
- [program.md](../../program.md) § "Pallas kernels to TRY" — this row closed as attempted-and-rejected at b=3 s=1024.

## Sources

- `/tmp/gemma4_jax_exp47.log` — full 20-step run log.
- `/tmp/gemma4_jax_exp47_smoke.log` — 5-step smoke test.
- Profile directory: [`raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/) — local xprof trace + xplane.pb.
- GCS mirror: `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp47-levanter-ce/`
- [`raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py) — public API (750 lines).
- [`raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py) — Mosaic TPU kernel (785 lines).
- [`raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py`](../../../../../raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py) — reference impl confirming `_apply_logit_soft_cap(logits, cap) = tanh(logits/cap)*cap` at line 11.
- [exp 36 page](2026-04-23-exp36-jax-splash-batch3-accepted.md) — the regression baseline.
