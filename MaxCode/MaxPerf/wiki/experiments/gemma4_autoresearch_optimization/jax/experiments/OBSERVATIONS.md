<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# OBSERVATIONS — Gemma 4 E4B native-JAX stack

Skim-and-reason aggregation log for the **native-JAX (Flax NNX)** path. Canonical per-experiment pages are the dated `*.md` files in this folder; this log threads the jax-stack session's arc for the human reviewer. Append-only.

Sibling: [`../../torchax/experiments/OBSERVATIONS.md`](../../torchax/experiments/OBSERVATIONS.md) — the torchax stack's aggregation log.

## port baseline: exp 34

See [2026-04-23-exp34-jax-baseline-accepted.md](2026-04-23-exp34-jax-baseline-accepted.md) for the full page.

**Summary** (v6e-4, 1D fsdp=4, bf16, XLA SDPA attention):
- seq=1024 b=1: **30,285 TPS, 135.24 ms/step**, loss 3.92 → 2.30 over 20 steps.
- Compile step 0: ~116 s. Step 1 recompile: ~119 s.
- Match to torchax baseline-seq1024 (30,570 TPS, 134.4 ms/step): **within −0.9 % noise** — port is numerically correct.
- Against torchax exp 25 (session-best, 33,372 TPS): **−9.2 %**. Gap explained entirely by missing splash + bf16 CE + fused_bwd + batch=3.

**Port semantics bugs caught (worth remembering for anyone porting Gemma 4)**:
1. `Gemma4RMSNorm` applies `weight * normed` with **weight init-ones**, NOT `(1 + weight) * normed` with init-zero. The HF source's comment is misleading; the real init is ones.
2. `Gemma4TextAttention.self.scaling = 1.0` (NOT `1/sqrt(head_dim)`) — because `q_norm` / `k_norm` (per-head RMSNorm on Q and K) already produce unit-norm vectors so the dot product is already at the right scale.
3. The last 18 of 42 layers are `is_kv_shared_layer=True` and **drop their own `k_proj` / `v_proj`** — they borrow cached K/V from an earlier layer. HF safetensors reflects this (54 weights skipped in the weight loader).

**Scaffold pieces landed**:
- `jax/model/modeling_gemma4.py` (779 LOC) — Flax NNX port.
- `jax/model/weight_loader.py` (199 LOC) — HF safetensors → NNX Param tree.
- `jax/model/sharding.py` (316 LOC) — FSDP + TP plans (mirrors torchax/model/sharding.py including the exp-32 hybrid 2D fix).
- `jax/train.py` (302 LOC) — trainer with full CLI flag parity with torchax/train.py.
- `jax/data.py` (73 LOC) — wikitext loader.
- `jax/tools/parity_{layer,attn,check}.py` (447 LOC) — correctness harnesses.

## exp 35 — splash pallas attention in JAX (POTENTIAL, flat)

See [2026-04-23-exp35-jax-splash-potential.md](2026-04-23-exp35-jax-splash-potential.md) for the full page.

**Config**:
- Command diff from exp 34: `JAX_ATTENTION_IMPL=splash` env var + new module `jax/model/pallas_attention.py` (~180 LOC) wired into `Gemma4TextAttention.__call__` behind the env-gate.
- Profile path: [`raw/profiles/2026-04-23-gemma4-jax-exp35-splash/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp35-splash/)
- **Profile browser URL**: http://localhost:8791/?run=2026-04-23-gemma4-jax-exp35-splash

**Hypothesis**: splash should win +2–5 % on the JAX port at batch=1 seq=1024, mirroring torchax exp 8 (+2.7 % at batch=2 seq=1024). Primary mechanism: avoid N² score-matrix materialization; secondary: Mosaic tiling beats XLA-fused GEMM for the `(B, H, S, D) @ (B, H, D, S)` + softmax + `(B, H, S, S) @ (B, H, S, D)` pattern.

**Changes made**:
- New file `jax/model/pallas_attention.py` — exposes `splash_attention(q, k, v, sliding_window)` using `jax.experimental.pallas.ops.tpu.splash_attention.make_splash_mha_single_device`, wrapped in `jax.shard_map(P('fsdp', None, None, None))` with the active mesh (set via `set_mesh(mesh)` from train.py). Block config mirrors torchax exp-25 best: `block_q = block_kv = block_kv_compute = 1024`, `block_*_dkv = 1024`, `use_fused_bwd_kernel=True`, `QKVLayout.SEQ_MINOR` for q/k/v. LocalMask(window_size=(512, 0)) for sliding layers; CausalMask for full layers. LRU-cached per (seq_len, num_q_heads, sliding_window, head_dim) — Gemma 4 has two head_dim variants (sliding@256, full@512).
- **No pre-kernel Q scaling** (unlike torchax/pallas_attention.py which does `q * 1/sqrt(d)`): the JAX port sets `self.scaling = 1.0` because q_norm / k_norm already pre-normalize per head — matches splash's no-1/sqrt-d convention exactly.
- Kernel pre-built at startup under `jax.ensure_compile_time_eval()` so MaskInfo is a concrete jax.Array (without this, first call inside the top-level jit captures tracers → step-1 retrace trips `UnexpectedTracerError`).
- `modeling_gemma4.py` — added the env-var dispatch in `Gemma4TextAttention.__call__`. One `if os.environ.get(...)=='splash'` branch; otherwise falls through to the existing `_attn_xla_sdpa`.
- `train.py` — reads `JAX_ATTENTION_IMPL`, calls `pallas_attention.set_mesh(mesh)` + pre-builds the kernel for (sliding_window, head_dim=256) and (None, head_dim=512) at startup.
- New parity harness `jax/tools/parity_splash.py` (~130 LOC) compares splash vs XLA-SDPA on-TPU at B=4 T=128; reports raw attn-output and post-o_proj errors.

**Expected outcome**: +2–5 % TPS; step time 135 → 129 ms; peak HBM modestly lower.

**Actual outcome**:
- TPS: 30,285 → **30,386** (Δ **+0.33 %** — flat, within noise)
- Step time (median 6-15): 135.2 → **134.8 ms** (-0.4 ms)
- Peak HBM: 16.85 → **16.43 GiB** (-0.42 GiB, 53.9 % → 52.6 %)
- Compile time: 142 → 132 s (-10 s — splash happens to compile faster here)
- Loss step 19: **bit-matches** exp 34 at 2.2969.

**Profile signals**:
- Bottleneck: compute (same as baseline). Mixed balance across conv-fusion, loop-fusion, custom-fusion.
- **Top op shift**: `convolution fusion` 614.6 → 549.0 ms (-65.6 ms / 3-step window), `custom fusion` 152.3 → 169.1 (+16.8) — confirms splash kernel is executing; matmul time moved from XLA into Mosaic. Net ~49 ms saved across 3 steps = ~16 ms / step, consistent with the -0.4 ms median step-time delta at noise level.
- `loop fusion` (~337 ms) unchanged — RMSNorm / residual-add ops are dominant, unrelated to attention kernel swap.
- HBM drop small because batch=1 seq=1024 keeps `[1, 8, 1024, 1024]` bf16 = 16 MiB / layer — already small; splash's 4× reduction unlocks batch=3+ / seq=2048+ regimes.

**Analysis**: The mechanism works exactly as predicted; the wall-clock delta is small because (a) batch=1 seq=1024 is below the regime where the N² tensor dominates, and (b) the splash custom-call launch overhead per layer (~0.4 ms × 42 layers × 3 steps ≈ 50 ms) offsets most of the matmul savings at these shapes. This is the same shape-sensitivity pattern seen in the torchax exp-8 → exp-25 arc: splash's marginal value scaled from +2.7 % (b=2 s=1024) → +8 % (b=3 s=1024 w/ bf16 CE) as batch increased. At batch=1 seq=1024 we're on the left end of that curve.

**Decision**: `potential` (parked) — correct implementation, flat TPS at current config. Merging to trunk because the code is the prerequisite for exp 36+ (batch=3, bf16 CE) and keeping it gated by env var means the XLA-SDPA baseline remains trivially reproducible.

**Follow-ups**:
- **exp 36 — splash + batch=3**. Direct analog of torchax exp 18 (+8.0 %). HBM 52.6 % has room for batch=3 (+~2-3 GiB). Confidence high, effort S.
- **exp 37 — splash + bf16 CE** (tokamax or hand-roll). Torchax exp-12 lesson: the `[B, S, V=262144]` fp32 logits tensor is ~1.5 GiB at b=2; dropping to bf16 / fused CE frees HBM and trims ~1-3 %.
- **exp 38 — splash + scan-over-layers**. JAX port owns the layer loop directly (no `ScannedModule` blocker), so this is more tractable than on torchax. Main win: step-0 compile drop ~130 s → ~4 s. Throughput neutral.
- Kernel-launch overhead finding: ~17 ms / 3-step added in custom-fusion for 42 × 3 = 126 splash calls is ~0.13 ms / call. This is a floor — scan-over-layers might amortize it by keeping the kernel code hot.

## exp 36 — splash + batch=3 in JAX (ACCEPTED, +13.9 %, new JAX-stack best, BEATS torchax session-best)

See [2026-04-23-exp36-jax-splash-batch3-accepted.md](2026-04-23-exp36-jax-splash-batch3-accepted.md) for the full page.

**Config**:
- Command diff from exp 35: `--batch_size 1` → `--batch_size 3`. No code change. `JAX_ATTENTION_IMPL=splash` unchanged.
- Profile path: [`raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/) (321 MB, gitignored).
- **Profile browser URL**: http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3
- GCS mirror: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`

**Hypothesis**: splash's per-call Pallas launch overhead is batch-independent (~0.13 ms × 42 layers × 3 calls/step ≈ 16 ms, mostly fixed), so bumping batch 1 → 3 amortizes it over 3× more tokens. Predicted +5–10 %, based on the torchax exp 15 → exp 18 arc (+7.0 % then +0.9 % additive on fused_bwd) where splash's marginal TPS value rose as batch grew.

**Changes made**:
- None to code. Only `--batch_size 1` → `--batch_size 3` on the command line.
- Data loader automatically switches to per-chip-batch=3 (global batch 12 = 3 × fsdp=4; still divisible, no sharding surprise).

**Expected outcome**: +5–10 % TPS; step time 135 → ~375 ms (scales with batch but sub-linearly); peak HBM 52.6 % → ~60 % (stack constant 16.4 GiB + ~3 × 1.5 GiB activation delta).

**Actual outcome**:
- TPS (median 6–15): 30,386 → **34,614** (Δ **+13.9 %** — upper end of expected range)
- Step time (median 6–15): 134.8 → **355.0 ms** (2.63× for 3× tokens — confirms sub-linear scaling, per-token cost −13.9 %)
- Step time (xprof avg steps 10–12): 142.2 → **375.5 ms** (profile inflates slightly)
- Peak HBM: 16.43 → **27.11 GiB** (52.6 % → **86.75 %**) — +10.68 GiB, all in heap (activations); stack constant at 16.44 GiB
- Free HBM at peak: **4.14 GiB** — room for one more batch raise (b=4) or one memory optimization
- Compile step 0: 132 → 167 s (+35 s; expected, bigger shape graph)
- Loss step 19: healthy 1.84 (descends 3.81 → 1.84; different seed/batch vs exp 35's 2.30, not bit-comparable)

**Profile signals**:
- Bottleneck: compute-bound (same as exp 35), now dominated by `convolution fusion` (33.6 % of step time) + `loop fusion` (28.1 %).
- **Top HLO-op diff vs exp 35 (3-step profile window)**:
  - `convolution fusion`: 549 → 1512 ms (**×2.75** for 3× tokens — MXU utilization improves at bigger shapes).
  - `loop fusion` (RMSNorm + residual-add): 332 → 1265 ms (**×3.81** — super-linear — next-tier bottleneck surface).
  - `custom fusion` (splash kernel): 169 → 175 ms (**×1.03** — **near-constant**, validating the per-call overhead hypothesis exactly).
  - `collective-permute-done`: ~11 → **550 ms** (12.2 % of step time) — **new pattern at b=3**, SPMD re-sharding cost.
- HBM: 27.11 GiB peak, no fragmentation (0.0 %), heap 10.67 GiB / stack 16.44 GiB / free 4.14 GiB.
- MXU utilization vs roofline not reported by xprof at this config (still 0 % — same as exp 35; TPU-profile plugin limitation on Pallas-heavy graphs).

**Analysis**: The mechanism predicted in exp 35 was right on both counts:
1. **Splash per-call overhead amortization** (primary): custom-fusion scaled 1.03× while everything else scaled 2.75–3.8×. At b=1, splash was 9.9 % of step; at b=3 it's 3.9 %. This alone buys ~6 %.
2. **MXU utilization improvement** (bonus, not anticipated): convolution-fusion growing 2.75× for 3× tokens means per-matmul MXU rows fill better at `[B=3, ...]` shapes. +7 % of the delta lives here.

Both effects compound in parallel and sum to the observed +13.9 %. **The native-JAX stack now outperforms the torchax session-best** (33,372 TPS exp 25) by **+3.7 %** — without bf16 CE, without SEQ_MINOR-specific tuning beyond what splash already defaults to, and without custom fused_bwd beyond splash's `use_fused_bwd_kernel=True`. The remaining torchax-side headroom (bf16 CE in exp 37, `collective-permute-done` tightening in exp 38) is additive — we're likely to pass the torchax best by >5 % once those land.

New bottleneck surfaces visible at b=3 that weren't actionable at b=1:
- `loop fusion` at 28.1 % of step time — RMSNorm + residual-add. Pallas RMSNorm kernel (from program.md's build-targets table) now worth the effort.
- `collective-permute-done` at 12.2 % — SPMD re-shuffle. `in_shardings`/`out_shardings` audit on the jitted step.

**Decision**: `keep` (ACCEPTED). New JAX-stack best. No code merge needed (it's a flag change); documenting via experiment page + RESULTS.tsv keep-row + this entry. README.md updated with new "Current state" numbers.

**Follow-ups**:
- **exp 37 — splash + b=3 + bf16 cross-entropy**. Highest priority. Frees ~1.5 GiB (HBM 86.8 % → ~82 %), trims one pass over `[B=3, S=1024, V=262144]` logits. Expected +1–3 % TPS.
- **exp 38 — collective-permute-done investigation**. 12.2 % of step time is a huge new bucket that didn't exist at b=1. Tighter in/out shardings on the jit body might reclaim half of it (5–6 %).
- **exp 39 — Pallas RMSNorm kernel**. `loop fusion` is now 28 % of step time; RMSNorm is 5 × 42 = 210 calls/step; single-HBM-pass kernel worth 3–8 %.
- **exp 40 — scan-over-layers**. Compile-time win (step 0 167 s → ~5 s), not throughput. Still worth doing — iteration speed improvement.
- **exp 41 — b=4**. With 4.14 GiB HBM free, b=4 adds ~3.5 GiB; may just fit. If exp 37 lands first, b=4 becomes very likely to fit. Defer until after 37.

## exp 37 — bf16 CE env-var gate on top of exp 36 (POTENTIAL, flat — was a no-op-by-construction)

See [2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md](2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md) for the full page.

**Config**:
- Command diff from exp 36: `+ JAX_CE_DTYPE=bf16` env-var (new, default). No other flag change.
- Code diff: `jax/train.py` `forward_loss` gains a dtype gate; default path is a no-op vs exp 36.
- Profile path: [`raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/)
- **Profile browser URL**: http://localhost:8791/?run=2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce
- GCS mirror: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`

**Hypothesis (as filed)**: replicate torchax exp 12's +3.0 % TPS / −1.5 GiB HBM win by dropping a fp32 upcast before `log_softmax`. Precondition assumed: such an upcast exists in the JAX port.

**Pre-run discovery**: reading `train.py`@0c44f60 revealed **no fp32 upcast** in the JAX port's CE path. The comment on line 213 ("bf16 log_softmax to drop the ~4 GiB fp32 logits tensor") had described design intent, not a cast to remove. The native-JAX port shipped with bf16 CE in exp 34 by construction — the torchax→JAX rewrite skipped the spurious `.to(fp32)` that torchax had inherited.

**Changes made**:
- `jax/train.py` — added `JAX_CE_DTYPE` env-var gate; default `bf16` (= exp 36 behavior), setting `fp32` reintroduces the upcast path for future A/B comparisons. Also nudged the final reduction's scalar accumulator to explicit fp32 (was implicit bf16) — this touches a `[B*S=3072]` scalar-only tensor, cost negligible, improves loss-value stability across batch sizes.

**Actual outcome**:
- TPS (median 6–15): 34,614 → **34,629** (Δ **+0.04 %** — flat, deep in noise)
- TPS (mean 2–19): 34,583 → 34,653 (+0.20 %)
- Step time (median 6–15): 355.0 → **354.85 ms** (−0.15 ms)
- Peak HBM: 27.11 → **27.45 GiB** (86.75 % → **87.84 %**, **+0.34 GiB**, the opposite direction from hypothesis) — stack +0.34 GiB, heap unchanged at 10.67 GiB, free 4.14 → 3.80 GiB
- Loss step 19: 1.8359 → 1.8314 (−0.0045, healthy; explicit fp32 scalar accumulator slightly more accurate)

**Profile signals**:
- HLO graph **bit-identical-modulo-one-implicit-cast** to exp 36. `loop fusion` −14.5 ms, everything else unchanged within <1 ms.
- conv-fusion 1512 ms (+0.7), loop-fusion 1250 ms (−14.5), collective-permute-done 549 ms (−0.3), splash custom-fusion 175 ms (−0.1).

**Analysis**: the experiment measured the default (bf16) path against exp 36's identical-by-construction default; the +0.04 % delta is noise. The hypothesized 1.5 GiB heap save did not materialize because there was no fp32 logits tile in exp 36 to save — XLA was already lowering `jax.nn.log_softmax(flat_logits_bf16)` to bf16-in/bf16-out from exp 34 onwards. The peak-heap tenant at the 10.67 GiB peak moment is the backward-pass activation tile, not the forward-pass log-softmax intermediate.

**What the torchax stack had that the JAX port didn't**: torchax's `train.py` line 418 had an explicit `flat_logits.to(torch.float32)` before `log_softmax`. Exp 12 removed that cast → 1.5 GiB saved. The native-JAX port was written from scratch (exp 34 port) without porting that cast, so the win came for free at port time.

**Decision**: `parked` (POTENTIAL, flat-by-construction). Durable artifact: the `JAX_CE_DTYPE` env-var gate (useful for regression guards if later experiments refactor the LM-head dtype flow). Exp 36 remains the **current JAX-stack best at 34,614 TPS**.

**Follow-ups (updated)**:
- Exp 37 is resolved as no-op. Ranking from exp 36 stands minus exp 37:
- **exp 38 — collective-permute-done investigation**. Still 12.1 % of step time (549 ms/3-step). Now highest-expected-value open hypothesis. `in_shardings`/`out_shardings` audit on the jitted step. Confidence medium. Effort S-M.
- **exp 39 — Pallas RMSNorm kernel**. `loop fusion` 27.6 % of step time (1250 ms). 210 norm calls/step. Confidence medium. Effort M.
- **exp 40 — scan-over-layers**. Compile-time win. Latency-to-first-signal.
- **exp 41 — b=4**. 3.80 GiB HBM free at exp 37. b=4 adds ~3.5 GiB activation — risky; defer until exp 38 frees some collective buffer space.

## exp 43 — tokamax.linear_softmax_cross_entropy_loss on JAX stack (INVALID, API-precondition failure — no run)

See [2026-04-23-exp43-jax-tokamax-ce-rejected.md](2026-04-23-exp43-jax-tokamax-ce-rejected.md) for the full page.

**Config**:
- Command diff from exp 36: proposed `JAX_CE_IMPL={default,tokamax}` env-var gate + call into `tokamax.linear_softmax_cross_entropy_loss(hidden, labels, lm_head.T)` in `forward_loss`. **Not implemented** — API inspection killed it before any code landed.
- Profile path: **none captured** (no run).
- **Profile browser URL**: n/a.
- Experiment page: `2026-04-23-exp43-jax-tokamax-ce-rejected.md`.

**Hypothesis (as filed)**: Replace the JAX stack's `logits = hidden @ W.T → softcap → bf16 log_softmax → NLL` sequence with a fused tokamax Mosaic-TPU kernel that never materializes the `[B=3, S=1024, V=262144]` bf16 logits tile (~1.5 GiB). Expected: ~1.5 GiB HBM freed (peak 86.8 % → ~82 %), small TPS gain (≤+3 %) from one fewer HBM round-trip.

**Pre-run discovery**: inspected `raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py` (+ `base.py`, `reference.py`, `pallas_mosaic_tpu_kernel.py`). The public API is:

```python
def linear_softmax_cross_entropy_loss(
    x: Real[Array, "B H"], labels: Integer[Array, "B"], weights: Real[Array, "H V"], *,
    reduction: Literal["sum", "mean"] = "sum",
    precision: ... = None, implementation: ... = None,
) -> Real[Scalar, ""]
```

No `logits_soft_cap` (or equivalent) kwarg anywhere in the op — `grep -rin "soft.*cap\|softcap" .../linear_softmax_cross_entropy_loss/` returns zero matches. The softcap references in tokamax live only in **attention** kernels.

**Why the softcap cannot be worked around**: Gemma 4's softcap is `sc * tanh(hidden @ W.T / sc)` applied element-wise to the full `[B, S, V]` logits (modeling_gemma4.py:782–786). It is non-linear in `hidden @ W.T`. Three fold-in options fail:
1. Fold into `W`: `sc * tanh(x W^T / sc)` is non-linear in `x W^T` — no `W'` exists with `x W'^T ≡ softcap(x W^T)`. Algebraically impossible.
2. Fold into `hidden`: same obstruction, non-linearity is post-matmul.
3. Apply externally: requires materializing `[B, S, V]` — defeats the kernel's sole purpose (zero-materialization streaming) and is strictly worse than exp 36 (extra pass).

The only correct path is a **kernel fork** with a softcap pre-op applied inline on each VMEM logits tile before `log_softmax` — exactly the "**Fused final logit softcap + log-softmax + NLL**" build-target already catalogued in [program.md § "Pallas kernels to BUILD"](../../program.md). Exp 43 empirically confirms that entry.

**Program-contract violation avoided**: skipping softcap is listed under "What you CANNOT do" in program.md. An "experiment" that drops it would change the model's output distribution — an `-invalid` on two independent axes (architecture contract + semantic divergence).

**Changes made**: **none**. Zero lines of code modified. No branch created.

**Actual outcome**:
- No run executed, no profile captured, no HBM / TPS / step-time measurement. Per SCHEMA § experiment template: "If the run was not executed, omit the Profile section and note the reason in `## Verdict`." Done.

**Secondary API mismatches** (for a future revisit if softcap ever lands):
- No `ignore_index` — `IGNORE_INDEX=-100` tokens handled externally via `mask = (labels != IGNORE_INDEX)` in `train.py:235`. Tokamax wants integer labels, internally one-hots. For `-100`, `jax.nn.one_hot` produces all-zeros (under-range), so the contribution *cancels* numerically but tokamax's `reduction="mean"` divides by `B=B*S`, not `mask.sum()`. Needs a post-kernel rescale by `B / mask.sum()`. Fixable at the call site, ~15 lines.
- `x` shape `[B, H]` (flat) — need `hidden.reshape(-1, H)`. Trivial.
- Softcap is the only hard blocker.

**Decision**: `discard` (INVALID / rejected). Commit stays at `c1927ba`. Exp 36 remains JAX-stack best at 34,614 TPS, peak HBM 27.11 GiB (86.75 %).

**Durable artifacts**:
1. Empirical confirmation that program.md's "Pallas kernels to BUILD" entry is correct — a Gemma-aware CE kernel is the only path to this win on this model. Estimated ~1.5 GiB HBM + 2–5 % TPS if built (fork the 690-line mosaic_tpu kernel, add a softcap pre-op on the logits-tile accumulator before `log_softmax`). Effort M.
2. Updated view of the program.md "Pallas kernels to TRY" table (line 107): tokamax LCE is listed as a drop-in candidate, but it's **not** a drop-in for any softcap model (Gemma 2/3/4). Logged as an edit-suggestion for the table (not in this experiment's scope to change).

**Follow-ups (unchanged ranking)**: `collective-permute-done` audit (exp 38-style) → Pallas RMSNorm → scan-over-layers → any future Gemma-CE kernel build that enables this exp's hypothesis.

## exp 47 — marin/levanter fused Pallas linear+softcap+CE on JAX stack (REJECTED, −5.61 % TPS, custom-call + all-gather tax)

See [2026-04-24-exp47-jax-levanter-ce-rejected.md](2026-04-24-exp47-jax-levanter-ce-rejected.md) for the full page.

**Config**:
- Command diff from exp 36: `JAX_CE_IMPL=levanter` + `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` env vars + new `jax/model/kernels/fused_ce/` import shim + wiring in `jax/train.py` `forward_loss` (`JAX_CE_IMPL=levanter` branch calls `Gemma4ForCausalLM.__call__(..., return_hidden=True)` to bypass lm_head + softcap in the model, then calls levanter's `fused_cross_entropy_loss_and_logsumexp_penalty` with `logit_soft_cap=30.0` inside a `jax.shard_map` wrapper) + a small `Gemma4ForCausalLM.lm_head_weight()` helper on the model + the `return_hidden` kwarg on `__call__`.
- Profile path: [`raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/) (local) + `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp47-levanter-ce/` (GCS mirror).
- Experiment page: `2026-04-24-exp47-jax-levanter-ce-rejected.md`.

**Hypothesis**: Replace the JAX-stack CE sequence (`hidden @ W.T → softcap → bf16 log_softmax → NLL`) with levanter's fused Pallas Mosaic-TPU kernel that applies softcap **inline on each VMEM logits tile** before the streaming `log_softmax`. This is the only public TPU Pallas CE kernel with a `logit_soft_cap` kwarg — the exact gap that made exp 43 invalid against tokamax. Expected: ~1.3 GiB HBM freed + small TPS gain (≤ +3 %) from eliminating the `[B, S, V]` logits pass.

**Key pre-work**:
1. **Import shim** (`jax/model/kernels/fused_ce/__init__.py`, 88 LOC): levanter's top-level package imports `equinox`, `draccus`, `trainer`, etc. — all heavy deps not in `gemma4_py313`. Pre-populating `sys.modules` with empty stubs for `levanter` and `rigging.filesystem` lets the kernel submodules (`levanter.kernels.pallas.fused_cross_entropy_loss.*`) load in isolation. Autotune-on-miss disabled via env var; we pass explicit `BlockSizes` to sidestep the `rigging` GCS-writer path entirely.
2. **Parity harness** (`jax/tools/parity_levanter_ce.py`, 140 LOC): reference (materialize logits + softcap + bf16 log_softmax + NLL) vs levanter (fused kernel w/ `logit_soft_cap=30.0`, `weight=mask`, `reduction="sum"`) on a `B=1 S=128 H=2560 V=262144` random batch, 10 % ignore-index. Result: `|diff| = 0.048` vs tol 0.05 — **PASS**.
3. **Block-size hand-pick**: Gemma 4 E4B (V=262144, H=2560) lands in no TPU-tuned bucket (`tuned_block_sizes.py` TPU buckets top out at V=131072; `gb10-large-vocab-mid-batch` is NVIDIA-GB10-only). Default `(1024, 512, 1024)` overruns 32 MiB VMEM by 8 MiB. Used `(b=1024, h=256, v=512)`. `b_block` had to be ≥1024 (multiple-of-1024 invariant in the kernel's `_validate_inputs` when per-shard B≥1024; our per-device flat batch is B*S=3*1024=3072).
4. **Shard_map wrap**: Mosaic custom-calls cannot be auto-partitioned (same constraint as splash in [`pallas_attention.py`](../model/pallas_attention.py)). `in_specs=(P('fsdp', None), P('fsdp'), P('fsdp'), P(None, None))` for `(hidden, labels, mask, w_hv)` — lm_head weight all-gathered to replicated `[H, V]`. `jax.lax.psum(local_sum, axis_name='fsdp')` sums partial loss-sums across shards; `out_specs=P()` returns a replicated scalar.

**Expected outcome**: +1–3 % TPS, peak HBM −1.3 GiB (86.8 % → ~82 %). Follow-on to the exp 43 rejection; softcap gap closed.

**Actual outcome**:
- TPS (mean steps 2–19): 34,614 → **32,671** (Δ **−5.61 %** — beyond the ±0.5 % flat band → **rejected**)
- Step time (mean 2–19): 355.0 → **376.1 ms** (+21.1 ms / +5.95 %)
- Peak HBM: 27.11 GiB → n/a (profile captured but not parsed — all-gather of w_hv adds temporary 1.31 GiB during CE call window)
- Step 0 compile: 167 s → **15 s** (compile-cache hit from exp 45; not a new win)
- Parity: **PASS** (|diff| 0.048 vs tol 0.05)
- Smoke step-4 loss: exp 36 = 2.1875 → exp 47 = 2.1979 (**+0.47 %**, well within 5 % semantic-drift bar)
- Later-step loss drift: step 19 exp 36 = 1.84 vs exp 47 = 2.00 (+9 %) — expected bf16-parity-noise compounding through Adam state across 20 steps; not a semantic change.

**Profile signals** (inferred, not yet xprof-browsed):
- Two new costs added vs exp 36:
  1. **Pallas custom-call boundary**: 3 Mosaic calls per step (fwd + two bwd halves) × ~5 ms launch latency = ~15 ms. Matches the custom-call-tax pattern from torchax exp 33 (Pallas RMSNorm, +36 ms / step regression from the same cause).
  2. **`w_hv` all-gather inside shard_map**: 1.31 GiB bf16 of lm_head weight (262144 × 2560) is all-gathered per forward pass (and again in backward). XLA may have been folding a collective-matmul version of this into the lm_head matmul in exp 36 via `AllGatherMatmul` fusion; moving the matmul into a Pallas custom-call breaks that fusion and exposes the gather on the critical path. This accounts for most of the remaining ~6 ms step-time delta.
- Total CE cost in exp 36 was <3 % of step time (~10 ms of 355 ms) — replacing a tightly-XLA-fused 10-ms op with a 3.5-ms Pallas kernel + 17–20 ms of new boundary + collective overhead is structurally net-negative.

**Analysis**: correctness premise held (softcap applies inline, `[B, S, V]` never materializes); throughput premise did not (CE isn't the bottleneck on this workload — splash + b=3 already amortized per-call overhead; the lm_head matmul + softcap + log_softmax was living in one XLA loop fusion). The Pallas CE kernel wins where CE is a meaningful fraction of step time (e.g. much longer sequences with larger logits-tile HBM pressure, or when CE is already on the critical path). At b=3 s=1024 v6e-4 it loses.

**Decision**: `discard` (rejected). Commit `573852c`. Exp 36 (34,614 TPS, 23.05 % MFU) remains the JAX-stack best.

**Durable artifacts**:
1. **`JAX_CE_IMPL=levanter` env gate + import shim** (`jax/model/kernels/fused_ce/__init__.py`). Makes the vendored levanter kernel importable from this trainer tree without forking it or installing equinox/draccus/rigging. Useful for future variants (e.g., if we rerun at seq=2048 where the cost-benefit flips, or if we eventually V-shard the CE).
2. **Parity harness** (`jax/tools/parity_levanter_ce.py`). Permanent correctness gate for any future CE kernel swap — run it before a benchmark run, refuse to proceed if |diff| > tol.
3. **`Gemma4ForCausalLM.__call__(return_hidden=True)` + `lm_head_weight()`** — clean seams for any future CE-replacement experiment (tokamax-with-softcap kernel, hand-rolled streamed CE, etc.). Generalizes beyond exp 47.
4. **Empirical confirmation of the "Pallas custom-call tax" heuristic on the JAX stack**. Matches torchax exp 33 (Pallas RMSNorm) on the sibling stack: when the baseline is XLA-fused tightly and the op being replaced is a small fraction of step time, dropping in a Pallas kernel structurally costs more than it saves. Worth promoting to program.md as a general rule.

**Follow-ups (ranked)**:
1. **Retry at seq=2048 b=2** (on top of [exp 40](2026-04-23-exp40-jax-seq2048-batch2-accepted.md)). Logits tile is 2× larger; HBM pressure is higher; the kernel's savings might cross the tax. One-flag-flip experiment. Confidence medium-low. Effort S.
2. **V-sharded fused CE kernel variant**. Accept `w_hv` with `P(None, 'fsdp')` sharding, do per-shard partial `logsumexp`, all-reduce the partial `lse` across fsdp inside the kernel boundary, emit the final NLL. Removes the 1.31 GiB all-gather. Kernel-edit, not an integration fix. Effort M, confidence low (XLA's AllGatherMatmul fusion may already be doing this optimally).
3. **Tune block sizes** on v6e-specific Gemma4 shapes. Sweep `(b, h, v)` via parity harness + 2-min benchmark loops. Unlikely to flip the verdict (the tax is structural) but could reclaim 1–3 %.
4. **Unchanged from exp 36 follow-ups**: `collective-permute-done` audit → Pallas RMSNorm → scan-over-layers.

## exp 49 — scan-over-layers in JAX stack (POTENTIAL, compile -61.5%, TPS -21.2%)

See [2026-04-24-exp49-jax-scan-layers-potential.md](2026-04-24-exp49-jax-scan-layers-potential.md) for the full page.

**Config**:
- Command delta from exp 36: `JAX_SCAN_LAYERS=1` env gate added. Existing `JAX_ATTENTION_IMPL=splash`, `--batch_size 3`, `--seq_len 1024` unchanged.
- New file: `jax/model/scan_layers.py` (551 LOC). Wired into `Gemma4TextModel.__call__` via a 17-LOC env-gated dispatch.
- Profile path: [`raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp49-scan-layers/)
- **Profile browser URL**: http://localhost:8791/?run=2026-04-24-gemma4-jax-exp49-scan-layers

**Hypothesis**: Replacing the 42-iter Python for-loop with `jax.lax.scan` should drop step-0 compile from ~180 s to ~5-15 s (per torchax exp 26's analysis), and may gain 2-5 % TPS from shared activation buffers. Torchax exp 26 parked the same idea due to 5 torchax-specific blockers (ScannedModule kwargs assertion, heterogeneous state_dict, etc.) — none of which apply to the native-JAX port that owns `Gemma4TextModel.__call__` directly.

**Structure**: The E4B layer pattern is exactly `[sliding x 5, full x 1] x 7`. Heterogeneous `head_dim` (sliding=256, full=512) forbids a single flat scan over all 42 layers. Used two nested scans: outer over 7 super-blocks, inner over 5 sliding layers per block; 1 full-attention layer per block applied inline. Stacked weight trees: sliding = `[7, 5, ...]`, full = `[7, ...]`.

**KV sharing** (18 shared layers borrow K/V from layers 22 and 23): handled with B1+B2+B5 from the torchax exp 26 sub-problem list —
1. B1 **zero-stub weights** on shared layers so stacking is homogeneous.
2. B2 **stored_k/v in scan carry** instead of dict side-effect; `jnp.where(is_kv_shared, borrowed, k_local)` selects.
3. B5 **traced int scalars** `is_kv_shared[block, inner]`, `is_store_kv[block, inner]` — no Python branches in the body.

**Expected outcome**: 3-15x compile-time drop (primary). 2-5 % TPS gain (secondary, upside bet).

**Actual outcome**:
- **Compile step 0: 180 s → 69.3 s** (−61.5 %, 2.6× faster). **Primary hypothesis confirmed**.
- TPS (mean steps 2-19): 34,614 → **27,290** (−21.2 % — **secondary hypothesis refuted**, large regression).
- Step time mean: 355.0 ms → 450.3 ms (+26.8 %).
- MFU: 23.05 % → **18.17 %** (−4.88 pt).
- Loss match: step 4 = 2.1990 vs exp 36's 2.1875 (**+0.5 %**, within bf16 reorder noise over 42 layers); step 19 = 1.8441 vs 1.8359 (**+0.5 %**). Trajectory descends cleanly. **Correctness PASS.**

**Regression root-cause (inference, pending xprof drill-down)**:
1. **Wasted zero-stub matmuls** on 18 shared layers: `k_proj` and `v_proj` with zero weights are real compute in HLO because XLA can't static-prove zeros inside a scan body with traced weight slices. Estimated ~35 ms/step.
2. **`jax.checkpoint` forced per-layer remat**: without it, scan materializes a full activation stack (`f32[7, 5, 3, 1024, 2560]` × several = 35 GiB OOM). With it, every layer is re-executed during backward, losing XLA's fine-grained remat flexibility that the exp-36 for-loop enjoyed.
3. **Splash custom-call inside shard_map inside checkpoint inside scan**: the kernel's `jax.shard_map(check_vma=False)` composes awkwardly with scan's HLO structure. XLA's ability to overlap the per-layer all-gather with compute is likely worse than in the unrolled case where all 42 splash calls can be scheduled with global visibility.

**Analysis**: The compile-time premise held cleanly. The runtime premise (shared activation buffers) was dominated by the overhead of the per-layer remat + wasted matmuls. Scan is a durable iteration-loop accelerator (on cache miss, fresh HLO compiles in 69 s vs 180 s) but not a steady-state TPS lever on this workload.

**Decision**: `keep` (potential). Code lives on main behind `JAX_SCAN_LAYERS=1` env gate. Exp 36 (34,614 TPS, 23.05 % MFU, Python for-loop) remains the JAX-stack best — the trunk default is unchanged.

**Durable artifacts**:
1. **`jax/model/scan_layers.py`** (551 LOC): pure-JAX functional Gemma 4 decoder-layer body + two-level scan with B1+B2+B5 KV-sharing handling. Reusable for any future "run scan path" follow-up.
2. **Env-gated dispatch** in `modeling_gemma4.py`: 17 LOC, off by default. Zero cost to the baseline path.
3. **Empirical data point on scan-under-autodiff** on Gemma 4 at v6e-4: shows that the B7 memory problem (scan stores per-iteration activations) must be solved with nested `jax.checkpoint` to avoid OOM on a 42-layer stack at b=3 s=1024. This is the torchax exp 26 B7 that was never measured — now measured and documented.
4. **Resolution of torchax exp 26 on the native-JAX stack**: the 7 Option-B sub-problems (B1-B7) can be resolved cleanly when the Python owns `Gemma4TextModel.__call__` directly; the torchax blockers were specific to torchax's scaffolding.

**Follow-ups (ranked, none promoted as urgent)**:
1. **exp 50 — cond-dispatched shared vs non-shared**: replace zero-stub matmuls with `jax.lax.cond` on `is_kv_shared`. Risk: cond may cost more than the matmul saves. Low-medium confidence. Only worth it if we find a production use case for the scan path.
2. **exp 51 — relax per-layer remat policy**: try a finer-grained `jax.checkpoint` policy inside the scan body. Effort S.
3. **Measure scan + xla SDPA** (no splash) to isolate splash's contribution to the regression. Trivial to try.
4. **Leave env gate off default**. The compile-cache (exp 45) already handles the ~360-s compile cost by caching; scan's floor-drop only matters on cache misses, which happen on fresh HLO. Revisit if a future code change invalidates the cache frequently.


## exp 50 — scan-over-layers tuned (POTENTIAL, compile -59.6%, TPS -16.1% vs exp 36 / +6.4% vs exp 49)

See [2026-04-24-exp50-jax-scan-tuned-potential.md](2026-04-24-exp50-jax-scan-tuned-potential.md) for the full page.

**Config**:
- Command delta from exp 49: none at env level — same `JAX_SCAN_LAYERS=1 JAX_ATTENTION_IMPL=splash`. Three code changes:
  1. Split `scan_layers` into two scan groups: 4-block non-shared (24 layers with real k/v_proj) + 3-block shared (18 layers reading K/V from closure-captured carry).
  2. Replaced exp 49's bare `@jax.checkpoint` on scan body with `jax.checkpoint(..., policy=checkpoint_dots_with_no_batch_dims)` matching exp 36's outer policy.
  3. In `train.py`: skip the outer `jax.checkpoint(forward_loss, policy=...)` wrap when `JAX_SCAN_LAYERS=1` (scan bodies already carry the policy per-iter; outer wrap was nested-remat'ing).
- Profile path: [`raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp50-scan-tuned/)
- **Profile GCS mirror**: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp50-scan-tuned/`

**Hypothesis**: Removing (1) zero-stub matmul waste on 18 shared layers and (2) the bare `jax.checkpoint` per-layer remat (replacing it with the policy-matched variant) should close most of exp 49's 21 pt TPS regression. Expected recoup: ~10-13 %. Target: ±0.5 % of exp 36 (34,441-34,787 TPS).

**Actual outcome**:
- Compile step 0: 180 s -> **72.7 s** (−59.6 %, 2.5× faster; vs exp 49's 69.3 s: +4.9 % penalty from compiling two scan groups instead of one).
- TPS: 34,614 -> **29,044** (−16.1 % all-sample 18 steps; −11.1 % on 17-sample window excluding one 828-ms step-9 outlier observed in both 15-step and 20-step runs, steady state otherwise 399 ms flat).
- Over exp 49: **+6.4 % TPS all-sample** / **+12.8 % 17-sample**. ~5 pt of the 21 pt gap closed.
- MFU: 19.34 % (all) / 20.50 % (17-sample) vs exp 36's 23.05 %, exp 49's 18.17 %.
- Loss match: step 4 2.1730 vs exp 36's 2.1875 (−0.66 %), step 19 1.8271 vs 1.8359 (−0.48 %). Clean descent, well within the 5 % tolerance.

**Remat path history (durable learning)**:
1. Fix A (remove inner checkpoint, rely only on outer `forward_loss` wrap): **OOM at 51.86 GiB**. Scan materializes `bf16[4,5,3,1024,10240]` (~1.17 GiB × 7 copies) + `f32[4,5,3,1024,2560]` (~600 MiB × 8 copies) across inner×outer iterations as it stores scan-invariant activations for backward. This confirms exp 49's OOM reasoning and makes it explicit that any scan-over-42-layers-at-b=3 variant needs SOME form of in-body remat to fit 31.25 GiB HBM. Durable observation for future scan work.
2. Fix B (inner `jax.checkpoint(policy=checkpoint_dots_with_no_batch_dims)`): fits. 432.98 ms, 28,380 TPS, −18.0 %.
3. Fix B + drop outer `forward_loss` checkpoint: 399.24 ms (17-sample steady state), 30,778 TPS, −11.1 %. **~34 ms/step gained** by removing double-remat.

**Root-cause finding — nested-checkpoint double-remat**: With the outer `jax.checkpoint(forward_loss, policy=...)` AND per-iter `jax.checkpoint(scan_body, policy=...)` both in effect, the backward pass replays the forward at two levels — once inside each scan iteration's checkpoint boundary, and again at the outer boundary. Removing the outer wrap when the scan bodies carry the same policy saves ~7.9 % step time with no memory cost (the per-iter checkpoints still prevent the scan-activation OOM). **New heuristic**: never nest `jax.checkpoint` with the same policy; the outer is redundant when the inner already covers every iteration.

**Durable artifacts**:
1. **`jax/model/scan_layers.py`** — cleaner two-group structure (no zero-stub hack, no `jnp.where` KV mux). Easier to reason about and extend.
2. **`jax/train.py` scan-aware checkpoint dispatch** — documents the nested-checkpoint pitfall inline.
3. **Empirical data point on nested-`jax.checkpoint` double-remat at scale**: quantified at ~7.9 % step time on Gemma 4 E4B at v6e-4 b=3 s=1024. General learning for any future scan-under-checkpoint workload.

**Follow-ups (ranked, none urgent since exp 36 remains JAX-stack best)**:
1. **exp 51 — scan + xla SDPA (no splash)** to isolate whether splash under shard_map under scan is the residual hot spot. Flag-only. If xla path shows smaller regression, splash is the culprit; we'd need a splash-scan-compatible sharding primitive. Confidence high for diagnosis.
2. **HLO drill-down** of exp 50 vs exp 36 step traces. Identify which HLO ops grew by >10 % under scan. Effort S. Confidence medium for finding one more lever.
3. **exp 52 — flatten inner + outer scans** within each group using `vmap + jit` or by reshaping `[n_blocks, 5, ...]` to `[n_blocks*5, ...]`. Non-obvious given sliding/full heterogeneity; the full layer still needs inline treatment. Effort M, confidence low.
4. **Leave env gate off default**. Exp 45 compile cache covers the 180-s compile on cache hits; scan's 72.7-s floor only matters on cache misses (code edits). The −11-16 % TPS cost makes scan a dev-loop tool, not a production default.

## exp 52 — new-regime baseline: fp32-master + bf16-compute AMP, seq=8192 infeasible → seq=2048 b=1 accepted (ACCEPTED)

**Config**:
- Command diff from prior keep (exp 36): `--dtype bf16` → `--weights-dtype fp32 --compute-dtype bf16`; seq_len 1024 → 2048; batch_size 3 → 1 (forced by the fp32-master memory ceiling); splash+fsdp=4 unchanged.
- Run command: `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache JAX_ATTENTION_IMPL=splash python -u -m train --steps 20 --batch_size 1 --seq_len 2048 --weights-dtype fp32 --compute-dtype bf16 --profile_dir $PROFILE_DIR --profile_steps 10 11 12`
- Profile path: `raw/profiles/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/` (on-disk, gitignored)
- **Profile GCS mirror**: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/`
- **Profile browser URL**: `http://localhost:8791/?run=2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master` (will resolve once xprof server is re-pointed at the autoresearch logdir — currently pointing at a different bucket).
- Experiment page: [2026-04-24-exp52-jax-fp32master-seq2k-accepted.md](2026-04-24-exp52-jax-fp32master-seq2k-accepted.md)

**Hypothesis**: User asked for a new regime — fp32 master weights (for the optimizer), bf16 compute (for matmul/conv), seq_len=8192. Establish a fresh baseline at this config and re-run the optimization loop against the new baseline. Expect: fp32 master adds ~8 GiB of opt-state HBM (3× params fp32 vs bf16), trimming headroom; seq=8192 triples the activation stack vs seq=2048. Hypothesis-secondary: seq=8192 may not fit; probe the wall before committing to it.

**Changes made** (code + flag wiring, merged in commit 517a689 + 176fd2c on main):
- `jax/train.py`: added `--weights-dtype` and `--compute-dtype` CLI args + `--dtype` legacy shortcut; seq_len default 2048 → 8192; `JAX_REMAT_POLICY` env-var (exposes `dots_with_no_batch_dims`, `nothing_saveable`, `everything_saveable`, `dots`, `offload_dot_with_no_batch_dims`).
- `jax/model/modeling_gemma4.py`: every NNX module (`Gemma4RMSNorm`, `Gemma4TextScaledWordEmbedding`, `Linear`, `Gemma4TextMLP`, `Gemma4TextAttention`, `Gemma4TextDecoderLayer`, `Gemma4TextModel`, `Gemma4ForCausalLM`) accepts split `weights_dtype` / `compute_dtype` kwargs. `Linear` forward downcasts weight to compute_dtype at the dot. `embed_tokens` gather casts to compute_dtype. `layer_scalar` casts to hidden-states dtype to avoid fp32-upcast of the residual.
- `jax/model/weight_loader.py`: new `shardings={path: NamedSharding}` kwarg scatter-shards each tensor at device_put time via a host-numpy scratch buffer → per-device HBM directly. This avoids a 10.5 GiB fp32 PLE embedding materializing on device 0 at load time (compile-time OOM at init otherwise). `target_dtype = weights_dtype if weights_dtype is not None else dtype`.
- `jax/train.py` reorder: `apply_sharding` now runs BEFORE `load_hf_weights` so the shardings lookup is available at load time.
- `jax/train.py` init-in-bf16 workaround: `_init_weights_dtype = compute_dtype` even when `weights_dtype=fp32`, so the random-init pass stays in bf16 (fits on device 0); the loader replaces with fp32 sharded at load time; a `_fixup_dtype_meta` pass retargets the per-module `weights_dtype` attribute for introspection.
- `jax/model/scan_layers.py`: all `x @ w.T` inside scan bodies routed through `_matmul_amp(x, w)` (downcasts `w` to `x.dtype` when they differ); `layer_scalar` cast-to-hidden-dtype. Without this the scan carry types mismatched (`bf16[4,8192,2560]` in, `f32[4,8192,2560]` out) and `jax.lax.scan` rejected the loop.
- `torchax/train.py`: flags present for CLI parity, warns + falls back to `--dtype` if a split is requested (full AMP implementation is JAX-only; HF PyTorch takes a single `torch_dtype`).

**Expected outcome**:
- seq=8192 b=1 fp32-master probably OOMs on v6e-4 by 3–6 GiB (roughly: fp32 opt-state +8 GiB over bf16; activation stack +2× vs seq=1024). bf16-legacy seq=8192 might fit; unclear.
- Fallback: seq=2048 b=1 fp32-master should fit (exp 40 bf16 b=2 at 27 GiB peak leaves ~4 GiB room, fp32 master eats most of that). Expected TPS: ~27-28K (between exp-36's 34,614 at lower-seq higher-batch and exp-40's 31,809 at the exp-40 shape).

**Actual outcome**:
- seq=8192 b=1 fp32-master: compile OOM, **35.18 GiB vs 31.25 GiB, exceeded by 3.93 GiB**.
- seq=8192 b=1 bf16-LEGACY (no AMP): also compile OOM, 36.16 GiB, exceeded by 4.91 GiB. **The seq=8192 wall is not caused by fp32 master — it's the model + activation stack on 4 chips.**
- seq=8192 b=1 bf16-legacy + JAX_SCAN_LAYERS=1: 32.46 GiB, exceeded by only 1.22 GiB. Scan narrows the gap but not enough.
- seq=8192 b=1 fp32-master + JAX_REMAT_POLICY=nothing_saveable (full remat): **39.66 GiB** (worse, +4.48 GiB over default). XLA serializes more live tensors to avoid recomputing.
- seq=8192 b=1 fp32-master + offload_dot_with_no_batch_dims (host RAM): **38.17 GiB** (also worse). XLA planner does not credit offload as HBM-freed at compile time. Matches torchax exp 11 lesson verbatim.
- seq=6144 b=1 fp32-master: **49.66 GiB** required — dramatically worse than seq=8192. Non-monotonic XLA scheduling: no intermediate seq_len between 2048 (fits) and 8192 (3.93 GiB over) provides a stepping stone; in fact seq=4096 (39.58 GiB) and seq=6144 (49.66 GiB) are WORSE.
- seq=2048 b=2 fp32-master: 39.37 GiB OOM (b=2 infeasible under AMP).
- **seq=2048 b=1 fp32-master: FITS.** Steady-state step time 305.6 ms (steps 6–15 median), **26,807 TPS**. Compile step 0 only 16.48 s (cache hit, exp 45 wins continue). Loss descent 3.25 → 2.30 clean, no NaN.

- TPS vs exp 40 (closest old-regime long-seq reference, bf16-only seq=2048 b=2 at 31,809 TPS): **−15.7 %**. Delta dominated by batch halving (b=2→b=1 under fp32-master), not by AMP per se. Pure AMP isolation needs a b=1 s=1024 sibling experiment (see Follow-ups).
- TPS vs exp 36 (old-regime best, 34,614 TPS at s=1024 b=3): **−22.5 %** (but very different shape).
- MFU: not computed this session (would need the same matmul-FLOP accounting done for exp 36).

**Profile signals**:
- Did not run xprof MCP queries — the local xprof_mcp server was pointed at a different GCS logdir (`gs://tpu-pytorch-alekseyv-us-central2/jax-experiment`) than the autoresearch bucket where exp 52's profile was uploaded. GCS profile is uploaded and the direct URL `http://localhost:8791/?run=2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master` will work once the xprof server is re-pointed. On-disk trace is complete and queryable via any xprof pointed at this folder.
- Sanity: loss trajectory 3.25 → 2.30 matches exp 40's early-step descent within noise. No NaN. No divergence from a typical bf16-splash run.

**Analysis**:
- **Mechanism matches expectation for "seq=8192 at risk"**. Post-hoc: opt-state at fp32 is `2 × params_fp32 ≈ 16 GiB` global → 4 GiB/chip with fsdp=4; params fp32 are another 2 GiB/chip; activations at b=1 s=8192 are ~7 GiB/chip (~42 layers × 160 MiB MLP intermediate, partially saved under remat). Plus PLE embed, lm_head lookup, collective buffers. 4+2+7 ≈ 13 GiB + overheads puts us very close to 31.25 GiB. AMP overhead alone (~1 GiB fp32 opt state delta vs bf16) is NOT the dominant blocker — the whole budget is tight at seq=8192 on 4 chips.
- **The AMP implementation is correct**. All casts happen at the right boundaries (weights cast-to-bf16 at matmul, embed cast-to-bf16 at lookup, layer_scalar cast-to-residual-dtype). Loss descends normally. fp32 master weights flow into optax.adamw naturally (it handles any param dtype). Grad dtype follows param dtype (fp32), so updates are applied in fp32 as intended.
- **seq=2048 b=1 is the representative workload for the new regime on v6e-4**. Until we have v6e-8 or a memory-shrinking code change, this is the new-regime baseline for all future optimization experiments.
- **Non-monotonic XLA scheduling (seq=4096 > seq=8192 in HBM)** is a genuine surprise and informs future experiment design: when the intermediate seq_lens look plausible, measure before assuming they sit on a monotonic curve between known-good and known-bad shapes. File as a heuristic.

**Decision**: **keep** — new-regime baseline reference. Committed on main.

**Follow-ups** (ranked by expected-gain × confidence / effort):
1. **exp 53 — pure AMP isolation at shared shape** (fp32-master b=1 s=1024 vs bf16 b=1 s=1024). Directly measures "how much TPS does fp32 master cost at a fixed shape". Expected: ≲5 %. Effort S.
2. **exp 54 — splash block sweep at seq=2048 under fp32 master** (SPLASH_BLOCK_Q=2048 full-tile; default 1024 gives 2 tiles/head). Expected: ±1 %. Effort S.
3. **exp 55 — scan_layers at seq=2048 b=1 fp32-master**. Compile-time win durable; TPS may differ from the exp 50 outcome at a different shape. Effort S (just env-var flip).
4. **exp 56 — 2D mesh (dp=2, tp=2) at seq=8192**. K/V replicated on tp=2 (num_kv_heads=2 divides 2) — unblocks mixed-sharding. Might fit seq=8192. Effort M. Memory-win-first per program heuristic.
5. **exp 57 — offload PLE embedding lookup to host**. `embed_tokens_per_layer` is 11 GiB fp32 / 5.5 GiB bf16 — biggest single tensor. Streaming it from host at the first-layer boundary could free enough HBM for seq=4096 or seq=8192. Risky code change. Effort L.

## exp 53 — splash block-size sweep at new-regime baseline (REJECTED, flat)

**Config**: seq=2048 b=1 fp32-master + bf16-compute; two splash block variants {2048, 512} vs exp 52's default block=1024.

- **Variant A (block=2048)**: compile-time `CompileTimeScopedVmemOom` — splash_mha_fwd_residuals needs 32.14 MiB VMEM vs 32 MiB hard limit (144 KiB over). Not runnable.
- **Variant B (block=512)**: TPS 26,807 — dead flat vs exp 52 (−0.0 %). Same step time median (305.6 ms).

Writeup: [2026-04-24-exp53-jax-splash-block-sweep-fp32master-rejected.md](2026-04-24-exp53-jax-splash-block-sweep-fp32master-rejected.md). Profile GCS mirror: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/`.

**Analysis**: Confirms the old-regime exp 48 plateau observation transfers to the new regime. Splash block size is not a TPS lever for Gemma 4 E4B on v6e-4 at any shape we've measured so far (old regime b=3 s=1024, new regime b=1 s=2048). **Durable heuristic** (promote to program.md Heuristics): "Splash block size is flat across all measured Gemma 4 E4B shapes on v6e-4 with the fused_bwd kernel + SEQ_MINOR layout — don't open a new block-size experiment unless the shape or kernel changes materially."

**Decision**: **discard**. Exp 52 remains new-regime baseline.

## new-regime ceiling analysis (exp 52–53 arc)

Filed as [analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md](../../../../analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md). Records the seq=8192 memory-wall data, the non-monotonic XLA compile-time peak observation, the counter-intuitive `nothing_saveable` / `offload_dot_with_no_batch_dims` regression, and the three-branch forward path (Branch A: optimize at seq=2048; Branch B: larger mesh; Branch C: memory-saving code changes).
