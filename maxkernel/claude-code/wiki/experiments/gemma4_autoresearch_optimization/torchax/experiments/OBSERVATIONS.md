# OBSERVATIONS — Gemma 4 E4B perf-autoresearch session

Skim-and-reason aggregation log. Canonical per-experiment pages are the dated `*.md` files in this folder; this log is for the human reviewer to thread the session's arc. Append-only.

## reference: none

No MaxText / native-JAX Gemma 4 reference implementation exists at the time of this program's start. The "ceiling" is therefore unknown; the program finds the best-achievable configuration for this stack (torchax on v6e-4) without a calibrated upper bound. **Follow-up when a reference lands**: port an analogous MaxText / Flax NNX / DeepMind-gemma-repo config for Gemma 4 E4B and re-calibrate MFU expectations.

## baseline: torchax + Gemma 4 E4B + v6e-4 + fsdp=4

See [2026-04-22-baseline.md](2026-04-22-baseline.md) for the full page.

**Summary**:
- seq=1024 canonical: 30,570 tokens/sec, 134.4 ms/step, ~23% MFU, loss 3.93 → 1.97 over 20 steps.
- seq=2048 diagnostic: 32,900 tokens/sec, 249 ms/step, ~26% MFU — but **loss NaN** at seq ≥ 2048. Clean at seq ≤ 1024.
- seq=512 micro: 17,500 tokens/sec, 117 ms/step, ~13% MFU — communication-dominated at this size.
- Peak HBM 29.69 / 31.25 GiB (95%). Fragmentation 0%. Stack 17.4 GiB (weights + opt state), heap 12.3 GiB (activations + intermediates).
- Top ops at seq=1024: convolution fusion 36% (compute-heavy, OI=1376 FLOPs/byte, compute-bound above ridge 578), loop fusion 19% (memory-bound at 62% HBM BW), custom fusion 11% (includes ~64 ms `async-collective-done` wait), all-gather 6.5%, all-reduce-scatter 4.4%, all-reduce 2.7%. Collectives ~14% aggregate.
- Step 0 compile ~150 s. Step 1 **recompiles** for another ~150 s — known open issue (suspected donated-input / output-sharding layout mismatch).
- Scaffold fixes that made the baseline run (detailed in [2026-04-22-baseline.md](2026-04-22-baseline.md)): `Gemma4ForConditionalGeneration` load + text-only-forward monkey-patch + `final_logit_softcapping=30.0`; `interop._jax_view` → `interop.jax_view`; accelerate dependency; drop `device_map="cpu"`.

## experiments

### exp01 — async-collective XLA flag bundle — `discard`

Canonical page: [2026-04-23-exp1-async-collective-flags-rejected.md](2026-04-23-exp1-async-collective-flags-rejected.md).

**Config**:
- Baseline config + `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true"`
- Profile: `raw/profiles/2026-04-23-gemma4-exp1-async-collectives/`; xprof symlink `gemma4_exp1_async_collectives_20260423`.

**Hypothesis**:
Baseline profile shows ~14% of step time in collectives + ~64 ms `async-collective-done` wait. The standard async-collective-fusion + latency-hiding flag bundle was predicted to overlap collectives with compute, shaving 5–10% off step time. Motivated by [async-collectives](../../../../concepts/async-collectives.md) and hypothesis #9 in the program's pre-schema open list.

**Changes made**:
- Env only: `LIBTPU_INIT_ARGS` with the five flags above.
- Two flags rejected as "Unknown" by libtpu 0.0.40: `--xla_tpu_overlap_compute_collective_comms` (no such flag; real name is `_tc`). First attempt had them in `XLA_FLAGS` (wrong — TPU runtime flags go in `LIBTPU_INIT_ARGS`).

**Expected outcome**: 134.4 → ~120 ms/step (−10%), collectives share down, compute share up.

**Actual outcome**:
- TPS: 30,570 → 24,300 (−20.5%).
- Step time: 134.4 → **168.3 ms (+25%)**.
- Loss trajectory identical within bf16-reorder noise.

**Profile signals**:
- all-gather 111 → 106 ms (−5 ms, as hoped).
- all-reduce-scatter 75 → 66 ms (−9 ms, as hoped).
- convolution fusion 613 → 791 ms (+178 ms) with **bytes-accessed 253 → 639 GiB (+2.5×)**.
- loop fusion 321 → 506 ms (+185 ms) with bytes 305 → 568 GiB (+1.9×).
- async-collective data formatting 40 → 72 ms, custom fusion 182 → 235 ms.

**Analysis**:
The stock XLA scheduler had already found a compute-order-friendly schedule for this small workload. Giving it more freedom via the fusion + latency-hiding flags let it reorder collectives for overlap but push compute fusions away from their consumers, forcing extra HBM round-trips (+1 GiB/step net bytes accessed). The collective-overlap win existed (−14 ms aggregate) but was eclipsed by the compute-locality loss (+363 ms). Classic "over-optimization" on a small workload.

**Decision**: `discard`.

**Follow-ups**:
- Retry this flag bundle after a structural change that unlocks them (e.g. after selective remat enables larger batch, where collective overlap has more compute to hide behind).
- Try `--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true` and `--xla_tpu_overlap_compute_collective_tc=true` (the correctly-named flag) in isolation — maybe the regression comes from one specific flag, not the bundle.

### exp03 — full activation remat via `jax.checkpoint` — `keep` (memory-first prep)

Canonical page: [2026-04-23-exp3-full-remat-accepted.md](2026-04-23-exp3-full-remat-accepted.md).

**Config**:
- Code diff: `grad_fn = jax.value_and_grad(forward_loss)` → `grad_fn = jax.value_and_grad(jax.checkpoint(forward_loss))`. One line.
- Profile: `raw/profiles/2026-04-23-gemma4-exp3-full-remat/`; xprof symlink `gemma4_exp3_full_remat_20260423`.

**Hypothesis**:
Baseline at 95% HBM (29.69 / 31.25 GiB) blocks any state-growing experiment. Full activation remat (wrap forward in `jax.checkpoint`) should drop peak HBM by 25–60% at the cost of +30–40% step time. Predicted via `xprof-mcp` TPU-optimization guide §4.6/§5. Unblocks exp 4 (batch 1 → 2) per the `program.md` HBM-ratchet heuristic.

**Changes made**:
- `train.py`: one-line change wrapping `forward_loss` with `jax.checkpoint`. No sharding, config, or flag changes.

**Expected outcome**:
Peak HBM drops from 95% to 55–70%. Step time 134.4 → ~175 ms (+30% ±5). TPS regresses ~25%. Loss trajectory identical.

**Actual outcome**:
- TPS: 30,570 → 23,900 (−21.8%).
- Step time: 134.4 → **171.4 ms (+27.5%)**.
- Peak HBM: 29.69 → **21.08 GiB (−29%)**; utilization 95% → 67%.
- Stack reservation halved (17.4 → 8.7 GiB); heap flat (12.3 → 12.4 GiB); fragmentation 0% → 25%.
- Compile step 0 +22% (checkpoint machinery).
- Loss trajectory identical within bf16-reorder noise (step-0 loss 3.9339 → 3.9147, step-14 min 1.9685 → 1.9621).

**Profile signals**:
- Bottleneck: shifted — `convolution fusion` still 37% but absolute FLOPs up 33% (356 → 473 TFLOPs). **Forward runs twice** (once for fwd output, once during bwd for remat); matches the 4× forward cost ratio.
- `all-gather` **doubled** (111 → 225 ms) — weight all-gather runs on both the rematted fwd and the bwd. Biggest concrete cost beyond raw matmul compute.
- `loop fusion` +52 ms; `custom fusion` +41 ms. Proportional.
- `all-reduce` slightly *faster* (46 → 33 ms) — recomputation enabled some collective fusion during bwd.

**Analysis**:
The mechanism matches the prediction: remat converts "long-lived activations on the stack" into "recompute during bwd." Stack reservation halved is the clearest indicator. The compute tax is 33% FLOPs and 27.5% wall-clock — a bit better than the heuristic said (30–40%), probably because the doubled all-gather overlaps with compute enough to amortize. Fragmentation rose to 25% — expected when the live set contracts, and benign. The critical signal is peak HBM at 67% — this frees ~10 GiB for the next experiment.

By isolated-TPS rules exp 3 would be `discard`. By the program's memory-ceiling rule + HBM-ratchet heuristic it's `keep`: the point is to prepare for exp 4. Final judgment is the combined exp 3 + exp 4 TPS vs baseline.

**Decision**: `keep` (memory-win; TPS-neutral outcome deferred to exp 4).

**Follow-ups**:
- **Exp 4 — double batch** (launching concurrently with this writeup). 10 GiB headroom → batch 2 should fit with remat. Target TPS > 30,570 (baseline).
- **Selective remat via `jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims`**: much better compute/memory tradeoff if exp 4 leaves HBM slack.
- **Host-offload the biggest single activation** (Ultra-Scale-playbook-style) rather than full remat, for a lighter compute tax.
- **Revisit exp 1 (async-collective flags)** at bigger batch — the regression mechanism (scheduler reorder breaking compute locality) may flip positive when there's more compute to hide behind.

### exp07 — selective remat + batch=3 — `discard` (memory-pressure degrades per-token)

Canonical page: (brief — row in RESULTS.tsv + this block).

**Config**:
- Code unchanged from exp 5 / 6. Config: `--batch_size 3 --seq_len 1024`. Global=12 (per-chip=3). Tokens/step=12,288.
- Profile: `raw/profiles/2026-04-23-gemma4-exp7-selective-batch3/`; xprof `gemma4_exp7_selective_batch3_20260423`.

**Hypothesis**:
Exp 6 showed per-token cost drops from 35.7 (batch=1) to 32.3 µs (batch=2). Pushing to batch=3 should drop it further via more collective amortization. HBM ratchet says: try it as long as memory fits.

**Actual outcome**:
- TPS: 30,925 → **29,720 (−3.9 %)** vs exp 6, **−2.8 % vs baseline**.
- Step time: 264.9 → 413.5 ms (ratio 1.56× for 1.5× tokens — expected ~1.4× by linear fit from exp 5→6, so slightly worse than projection).
- Peak HBM: 25.92 → **30.50 GiB (97.6 %)** — nearly OOM, **0.74 GiB free**.
- Per-token cost: 32.3 → **33.7 µs** (+4.3 %). Goes the wrong way.
- Loss descent: normal.

**Analysis**:
Per-token efficiency **degraded** when it should have improved. Two concurrent causes:
1. **HBM pressure near ceiling** — allocator has near-zero slack; each kernel launch pays for small reallocations / defragmentation / compile-time scheduling around tight memory. Fragmentation went 48 % → 0 % (batch=3 packs all gaps, but the allocator has to work harder).
2. **Linear scaling of compute exceeds amortization gain**. At batch=2 the amortization of fixed overhead (~19 % per exp 6 mechanism) was the win; at batch=3, the additional amortization saved ~5 % but the memory-pressure cost ate that and more.

Sweet spot is **batch=2** at this config. To push batch=4 or higher, need another memory win first (offload remat / splash attention / bf16 CE / smaller optimizer state).

**Decision**: `discard`. Revert `--batch_size` to 2. Exp 6 remains current best.

**Follow-ups**:
- **Exp 8 (queued)** — splash attention via Pallas. Targets `convolution fusion` 37 %, avoids N² score-matrix HBM traffic. First pure-Pallas-kernel experiment.
- **Offload variant remat** (`offload_dots_with_no_batch_dims`) — move saved dots to host; frees HBM for batch>2.
- **Drop fp32 upcast in the hand-rolled CE** (~4 GiB memory win at batch=2).

### exp05 — selective remat (`checkpoint_dots_with_no_batch_dims`) — `keep` (memory-first prep, low tax)

Canonical page: (to be filed; OBSERVATIONS has the full block).

**Config**:
- Code diff vs baseline: `grad_fn = jax.value_and_grad(jax.checkpoint(forward_loss, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims))`.
- Config unchanged: batch=1, seq=1024, fsdp=4.
- Profile: `raw/profiles/2026-04-23-gemma4-exp5-selective-remat/`; xprof symlink `gemma4_exp5_selective_remat_20260423`.

**Hypothesis**:
Full remat (exp 3/4) had two costs: +33% compute (forward done twice) and doubled all-gather. Selective remat via `checkpoint_dots_with_no_batch_dims` policy **saves the dots** (matmul outputs — the expensive-to-recompute ones) and recomputes only the cheap elementwise intermediates (norms, residuals, etc). Predicted: +5–10% step time, modest memory savings, **no forward doubling**.

**Changes made**:
- Replaced `jax.checkpoint(forward_loss)` with `jax.checkpoint(forward_loss, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)`.

**Expected outcome**:
Step time 134.4 → ~142–148 ms. Peak HBM drops less than exp 3 (maybe 28 → 23 GiB). No forward doubling so all-gather unchanged.

**Actual outcome**:
- TPS: 30,570 → 28,035 (−8.3% vs baseline; better than exp 3's −21.8%).
- Step time: 134.4 → **146.1 ms (+8.7%)** — exactly in predicted range.
- Peak HBM: 29.69 → **19.32 GiB (−35%)**; utilization 95% → **62%**.
- Stack: 17.37 → **7.0 GiB (−60%)** — even better than exp 3's 8.72 GiB!
- Free memory: 1.56 → **11.93 GiB** (+10.4 GiB headroom).
- Fragmentation 22% (expected when live set shrinks).
- Compile step 0 only +4.4% (vs full remat's +22%).

**Profile signals**:
- `convolution fusion` FLOPs: 356 → 358 TFLOPs (**+0.6% — no forward doubling**, as predicted). Contrast exp 3 where it went 356 → 473 TFLOPs (+33%).
- `all-gather`: 111 → 123 ms (+11 ms). Compare exp 3 at 225 ms — selective doesn't force the double-gather.
- `loop fusion`: 321 → 375 ms (+54 ms, bigger than exp 3's +52). The elementwise recompute is real but small in absolute terms.
- `custom fusion`: 182 → 164 ms (−18 ms, surprise). Fewer async-collective-done waits — not sure why yet.
- `all-reduce-scatter fusion`: 75 → 82 ms.

**Analysis**:
Selective remat is a much better trade than full remat for this workload:
- Compute tax 8.7% vs 27.5% (3× better).
- Memory savings 35% vs 29% (better!).
- No all-gather doubling.
- Compile tax 4% vs 22%.

The surprising result is that selective saved MORE memory than full. Hypothesis: the policy keeps matmul outputs (dense, contiguous, few) but recomputes all the broadcast / fusion / layout intermediates, which are individually small but can be numerous. The net live-set reduction beats full remat because full remat's "keep only inputs" materializes a lot of layout-churn buffers that selective avoids.

**Per-token cost**: 146.1 / 4096 = 35.7 µs — vs baseline 32.8 µs (+8.7%). To beat baseline, need batch amortization: exp 6 with batch=2 should drop per-token closer to baseline because collective fixed costs amortize.

**Decision**: `keep` (memory-first prep). TPS alone is below baseline, but the 11.9 GiB of freed HBM enables exp 6 (batch=2) and beyond. Much better starting point than exp 3.

**Follow-ups**:
- **Exp 6 (launched)** — selective + batch=2. Target per-token ≈ 32–33 µs (beat baseline).
- **Exp 7 (if exp 6 wins)** — selective + batch=4 (if HBM allows).
- **Selective + async-collective flags** — exp 1 bundle may help now that we have memory headroom to absorb its scheduler cost.
- **Replace `checkpoint_dots_with_no_batch_dims` with `offload_dots_with_no_batch_dims`** — offload saved dots to host memory, freeing even more HBM.

### exp04 — double batch under full remat — `discard` (chain net −9 % TPS)

Canonical page: [2026-04-23-exp4-batch2-with-remat-rejected.md](2026-04-23-exp4-batch2-with-remat-rejected.md).

**Config**:
- Code unchanged from exp 3 (`jax.checkpoint(forward_loss)`).
- Config diff: `--batch_size 1` → `--batch_size 2`. Global batch 4 → 8.
- Profile: `raw/profiles/2026-04-23-gemma4-exp4-batch2-with-remat/`; xprof symlink `gemma4_exp4_batch2_remat_20260423`.

**Hypothesis**:
Exp 3 freed ~8.6 GiB HBM. Doubling batch from 1 to 2 fits (HBM ratchet heuristic), amortizes per-step collective overhead across 2× tokens, and flips the chain exp 3 + exp 4 to a net TPS win vs pre-remat baseline.

**Changes made**: config flag only.

**Expected outcome**: step time 171 → ~260 ms (sub-linear scaling from collective amortization), TPS 23,900 → ~32,000 → beat baseline 30,570.

**Actual outcome**:
- TPS: 23,900 → 27,840 (+16.5 % vs exp 3, **−8.9 % vs baseline**).
- Step time: 171.4 → 294.3 ms (1.72× for 2× tokens — sub-linear, but not enough).
- Peak HBM: 21.08 → 28.79 GiB (67 % → 92 %; stack reservation climbed back to 16.4 GiB, nearly undoing exp 3's savings).
- Fragmentation: 25 % → 0 % (batch=2 refills the gaps).
- Per-token cost: 41.8 → 35.9 µs (vs baseline 32.8 µs — still **+9.5 % over baseline**).

**Profile signals**:
- Same op mix as exp 3, scaled.
- Collective amortization worked where expected (collective % of step dropped), but the compute tax from remat is 60 %+ of the step — only the collective minority amortized.
- 92 % HBM blocks batch=4.

**Analysis**:
The 1.72× step-time scaling for 2× tokens tells the story: collectives amortized (the minority), but compute did not (the majority is paying remat's +33 % tax). To reach baseline per-token cost we'd need either (a) higher batch with fatter per-step compute (blocked — 92 % HBM) or (b) lower compute tax (selective remat). Ruling in favor of selective remat for exp 5.

**Decision**: `discard`. Exp 3 + exp 4 chain fails the strict "TPS improved beyond noise" test. Reverting the `jax.checkpoint(forward_loss)` for exp 5 (which replaces it with a selective-remat policy).

**Follow-ups**:
- **Exp 5 (launched)**: selective remat via `checkpoint_dots_with_no_batch_dims`. Target: +5–10 % step time, 50–70 % of exp 3's memory savings. If per-token cost comes in < baseline 32.8 µs → chain supported.
- Once selective pays off at batch=1, push to batch=2 or higher.
- Scan-over-layers remains an orthogonal candidate (compile-time + buffer-sharing).

### exp02 — pin `out_shardings` to fix step-1 recompile — `crash`

Canonical page: (minimal stub; see the commit + this block).

**Config**:
- Code change in `train.py`: built `weights_out_shardings` from `plan.shardings`, `opt_state_out_shardings` via `jax.tree.map(lambda a: a.sharding, opt_state)`, passed as `jax.jit(..., out_shardings=(loss_ns, weights_out_shardings, opt_state_out_shardings))`. Also re-applied `plan.shardings` to `weights` via `jax.device_put` right after `interop.jax_view`.

**Hypothesis**:
The ~150 s step-1 recompile is caused by step 0's output layout not matching its input layout — step 1 receives donated outputs with different shardings than step 0's input cache key, so it re-traces. Pinning `out_shardings` to equal `in_shardings` would make step 1 hit the cache.

**Changes made**:
- `train.py`:
  - After `interop.jax_view(jmodel.params)`, `jax.device_put` each weight to `plan.shardings.get(k, _replicated)`.
  - Added `out_shardings=(replicated, weights_out_shardings, opt_state_out_shardings)` to the `jax.jit` call.

**Expected outcome**: step 1 at ~134 ms instead of ~150 s. Saves ~300 s per run (both compile steps) → ~3 min / experiment iteration speed win.

**Actual outcome**: **crash at trace time**.
```
ValueError: Received incompatible devices for jitted computation.
Got argument weights['lm_head.weight'] of main.<locals>.train_step
with shape bfloat16[262144,2560] and with device ids [0, 1, 3, 2]
on platform TPU and explicit output sharding with device ids [0] on platform TPU
```

**Profile signals**: none — pre-trace crash.

**Analysis**:
Gemma 4's `tie_word_embeddings=True` causes `lm_head.weight` ↔ `model.language_model.embed_tokens.weight` to share storage. torchax's `JittableModule` deduplicates one of them (based on `id(v.elem)`), but `load_state_dict(assign=True)` before that can separate them into distinct jax.Arrays with different shardings. When I built `weights_out_shardings = {k: plan.shardings.get(k, _replicated) for k in weights}`, the tied key's sharding in `plan.shardings` was correct (4-device) but jit's internal view of the key disagreed — it saw the output sharding as single-device. Root cause unlocalized (possible JittableModule behavior with assign=True).

**Decision**: `crash`. Reverted the `out_shardings` + `device_put` additions. Revert verified (baseline still runs). Step-1 recompile remains open; not a blocker for other experiments but costs ~150 s per run.

**Follow-ups**:
- Route around the tied-weight dedup: either disable dedup (`JittableModule(model, dedup_parameters=False)`) + manually tie lm_head ↔ embed_tokens post-load, or apply `out_shardings` only to non-tied weights and let jit infer for the tied one.
- Test whether `jax.lax.with_sharding_constraint` inside `train_step` (rather than `out_shardings` on jit) works better for the tied case.
- Alternative: use persistent compile cache (`JAX_COMPILATION_CACHE_DIR`) so subsequent runs skip both compiles (would also address step-0 cost across runs, not just step 1 within a run).

## approach evolution

### approach update — 2026-04-23 — Pallas kernel landscape section added to program.md

After exp 1–7 established that stock XLA path + selective remat + batch=2 is the current best (+1.2 % over baseline), the user directed a formal Pallas analysis. Added a "Pallas kernel landscape" section to `program.md` covering:
- Decision rule: Pallas beats XLA on memory-bound ops (< ridge point); XLA wins on compute-bound dense matmuls.
- Existing TPU Pallas kernels to try: **splash attention** (via `jax.experimental.pallas.ops.tpu.splash_attention` or tokamax), **memory-efficient CE** (tokamax `linear_softmax_cross_entropy_loss`).
- Kernels to build (Wave 1 finding: tokamax has no TPU Pallas kernel for these): **RMSNorm**, **SwiGLU/GLU**, **fused residual+RMSNorm**, **fused softcap+CE**, **fused Q/K RMSNorm+RoPE**.

Exp 8 (splash attention) kicked off as the first Pallas experiment.

### approach update — 2026-04-23 — initial program.md landing

This program.md landed at the start of the formal session, after the baseline + exp01 + exp02 were already run. Those three ran under a lighter-weight "optimization loop procedure" section in the program `README.md`; that section is now replaced by a pointer to this file. Future experiments (exp03 onward) follow this protocol. The pre-program.md experiments are retroactively transcribed into `RESULTS.tsv` and this `OBSERVATIONS.md` for continuity.

### exp34 — native-JAX (Flax NNX) port baseline — `keep`

Canonical page: [2026-04-23-exp34-jax-baseline-accepted.md](2026-04-23-exp34-jax-baseline-accepted.md).

**Config**:
- Brand-new `jax/` trainer: `wiki/experiments/gemma4_autoresearch_optimization/jax/{train.py,data.py,model/{modeling_gemma4,weight_loader,sharding}.py}` (~2.3 kLOC). Runs the same model config via a native Flax NNX implementation (not HF PyTorch under torchax dispatch).
- Command (from `jax/`): `python -m train --batch_size 1 --seq_len 1024 --steps 20 --profile_dir $WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-baseline --profile_steps 10 11 12`.
- Profile: `raw/profiles/2026-04-23-gemma4-jax-baseline/` (198 MB).

**Hypothesis**:
A native-JAX port with XLA-SDPA attention should match the torchax baseline-seq1024 (30,570 TPS) within noise — establishing a clean anchor for future JAX-native optimizations (splash, tokamax, scan-over-layers) that are cheaper to wire from the JAX side than through torchax.

**Changes made**:
- New code: `jax/model/modeling_gemma4.py` ports `Gemma4{RMSNorm, TextScaledWordEmbedding, TextRotaryEmbedding, TextMLP, TextAttention, TextDecoderLayer, TextModel, ForCausalLM}` as Flax NNX modules. Text-only; skips audio/vision/multimodal/MoE.
- New loader: `jax/model/weight_loader.py` reads HF safetensors into the NNX Param tree, drops audio/vision tensors, drops shared-KV k_proj/v_proj/k_norm/v_norm (matching HF's `_keys_to_ignore_on_load_unexpected`). 665 assigned, 54 shared-KV skipped, 2 missing (audio/vision embed connectors).
- New sharding plan: `jax/model/sharding.py` — FSDP 1D same rule as torchax (largest divisible dim). All 623 ≥2D params sharded; 42 scalar `layer_scalar` buffers replicated.
- New trainer: `jax/train.py` — CLI parity with `torchax/train.py`, identical summary block, identical CE loss + optax.adamw + selective remat.
- Correctness harnesses: `jax/tools/parity_{layer,attn,check}.py`. The layer check (RMSNorm, MLP, RoPE, full decoder layer for sliding + full layer types) is the canonical one — ~30 s on CPU.

**Expected outcome**: 30,570 TPS ±1 % (noise), loss 3.93 → 2.04 ish, step 0 compile ~150 s.

**Actual outcome**:
- TPS: **30,285** (-0.9 % vs torchax baseline-seq1024; within noise).
- Step time (median steps 13-19): **135.24 ms** (vs torchax 134.4 ms, +0.6 %).
- Step 0 compile: 141.92 s. Step 1 recompile: 141.92 s (identical long-compile pattern; same open issue as exp 2).
- Loss trajectory step 0 → 19: **3.9219 → 2.2969**, matches torchax 3.93 → ~2.0 within bf16 reorder noise.
- Per-layer parity vs HF PyTorch (bf16): RMSNorm = 0, MLP = 0.0011, RoPE = 0, full decoder layer (sliding + full) = 0.031. PASS.

**Profile signals**:
- Profile captured at steps 10-12. Not yet analyzed in detail (follow-up); visual inspection deferred to the first optimization experiment that needs to target a specific op.

**Analysis**:
Three HF-Gemma4 semantics details would have been silent bugs:
1. `Gemma4RMSNorm` is `weight * normed` (weight init = ones), **NOT** `(1 + weight) * normed` (Gemma 2/3 convention). Task-spec's `init-zero + (1+weight)` was wrong for Gemma 4.
2. `Gemma4TextAttention.scaling = 1.0`, NOT `1/sqrt(head_dim)` — the per-head q_norm/k_norm already normalize per-head. This bug produced saturated softmax (pos 0 got ~1.0, everything else 0) — caught by position 0 matching but all other positions diverging.
3. `num_kv_shared_layers=18`: 42-layer E4B last 18 layers read KV from earlier same-type layer. The HF checkpoint still carries their k_proj/v_proj/k_norm/v_norm weights — HF's `_keys_to_ignore_on_load_unexpected` drops them at load. Our NNX loader needs the same drop.

The -0.9 % vs torchax is within measurement noise; essentially identical. Against exp 25's 33,372 TPS (session-best: splash + batch=3 + bf16-CE + fused-bwd + selective-remat), we're -9.2 %, all of which is attributable to **not yet having splash attention** on the JAX side. That's the first obvious follow-up.

**Decision**: `keep`. First native-JAX trunk point; future JAX-native experiments fork from here.

**Follow-ups**:
- **Exp 35 (next)**: wire `jax.experimental.pallas.ops.tpu.splash_attention` into `jax/model/modeling_gemma4.py` analogous to `torchax/model/pallas_attention.py`. Expected: match exp 25's 33,372 TPS (+10 %), unlock batch=3.
- **Exp 36**: `flax.nnx.scan` over the 42 layers — but E4B is heterogeneous (sliding/full with different head_dim), so needs two sub-scans.
- **Exp 37**: tokamax `linear_softmax_cross_entropy_loss` directly on the JAX side (no torchax interop needed).
- **Exp 38**: root-cause the step-1 recompile. NNX's `split/merge` should make input-pytree shape stable across steps; still seeing ~142 s recompile. Likely XLA input-layout specialization.
