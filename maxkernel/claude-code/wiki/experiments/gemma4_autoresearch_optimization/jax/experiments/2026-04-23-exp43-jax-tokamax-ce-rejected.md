---
title: "Exp 43 — tokamax.linear_softmax_cross_entropy_loss on JAX stack (INVALID / rejected, API-precondition failure — no softcap support)"
type: experiment
tags: [experiment, gemma4, jax, tokamax, cross-entropy, api-mismatch, invalid, no-run, softcap]
hypothesis: jax-tokamax-linear-softmax-ce
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: c1927ba
verdict: invalid
---

Attempt to integrate `tokamax.linear_softmax_cross_entropy_loss` (fused `linear + log_softmax + CE` Pallas kernel that never materializes the full `[B, S, V]` logits tensor) into the JAX-stack trainer to free ~1.5 GiB of HBM and remove one pass over the vocab-dimensioned logits. **Result: INVALID before any run — the tokamax public API has no `logits_soft_cap` (or equivalent) argument, and Gemma 4's `final_logit_softcapping=30.0` (`logits = 30 * tanh(logits / 30)`) is a non-linear transform applied element-wise to the `[B, S, V]` logits. Applying the softcap externally requires materializing that tensor, which defeats the only reason to use the fused kernel. Skipping the softcap violates the program's architecture contract ("You CANNOT: Skip `final_logit_softcapping=30.0`"). No code merged, no experiment run.**

Exp 36 (34,614 TPS) remains the JAX-stack best. This result **empirically confirms the "Pallas kernels to BUILD" entry** already filed in [program.md § Pallas kernels to BUILD](../../program.md) — a **Gemma-aware CE variant** (fused softcap + log-softmax + NLL over a streamed `hidden @ lm_head.T` tile) is the next-tier kernel-build target for this stack.

## Hypothesis (as filed)

Replace the JAX-stack `forward_loss` sequence — `logits = hidden @ W.T` → `sc * tanh(logits / sc)` softcap → bf16 `log_softmax` → `take_along_axis(label)` NLL — with a single tokamax fused kernel. Expected: ~1.5 GiB HBM freed (no `[B=3, S=1024, V=262144]` bf16 materialization ≈ 1.5 GiB); small TPS gain (≤+3 %) from one fewer HBM round-trip over the logits tile. Decision tree:

- TPS > 34,614 × 1.005: `-accepted`, new JAX-stack best.
- Flat ±0.5 %: `-potential`.
- Regresses: `-rejected`.
- Loss diverges from exp 36 at step 4 by >5 %: `-invalid`.
- **Softcap incompatible with tokamax API → can't apply cleanly → `-invalid`.** ← this is what triggered.

## What actually happened — API inspection (no run performed)

### Verified tokamax is installed and imports

```
$ $HOME/miniconda3/envs/gemma4_py313/bin/python -c \
    "import tokamax; print(tokamax.__version__); \
     from tokamax._src.ops.linear_softmax_cross_entropy_loss import api as lce; \
     print([x for x in dir(lce) if not x.startswith('_')])"
0.0.12
['Array', 'Callable', 'IMPLEMENTATIONS', 'Implementation', 'Integer', 'Literal',
 'Real', 'Scalar', 'Sequence', 'TypeAlias', 'base', 'chunked_xla', 'jax',
 'linear_softmax_cross_entropy_loss', 'pallas_mosaic_tpu']
```

### Inspected the public API signature

[`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py):

```python
def linear_softmax_cross_entropy_loss(
    x:       Real[Array, "B H"],
    labels:  Integer[Array, "B"],
    weights: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean"] = "sum",
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | ... = None,     # "mosaic_tpu" | "xla" | "chunked_xla"
) -> Real[Scalar, ""]:
  """loss = -reduction(labels_one_hot * log(softmax(X @ W)))"""
```

### Inspected the reference impl to confirm the math

[`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py):

```python
logits = x @ w
log_probs = jax.nn.log_softmax(logits, axis=-1)
labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
loss = -labels_one_hot * log_probs                      # no softcap, no mask
lse = jax.nn.logsumexp(logits, axis=-1)
return (loss.sum(dtype=jnp.float32), lse)
```

The Pallas TPU kernel ([`pallas_mosaic_tpu_kernel.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py), 690 lines) implements the same math streamed over vocab blocks in VMEM — also **no softcap hook**.

### Searched the entire tokamax LCE op for any softcap surface

```
$ grep -rin "soft.*cap\|logits_soft_cap\|softcap" \
    raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/
(no matches)
```

The only `logits_soft_cap` references in the tokamax codebase live inside **attention** kernels (`dot_product_attention`, flash / splash variants) — not the CE op.

### Why the softcap cannot be folded in externally without defeating the kernel

Gemma 4's softcap in [`jax/model/modeling_gemma4.py:782–786`](../model/modeling_gemma4.py):

```python
logits = hidden @ weight.T           # (B, T, V)
sc = self.config.final_logit_softcapping  # 30.0
if sc is not None and sc > 0:
    sc_f = jnp.float32(sc)
    logits = sc_f * jnp.tanh(logits.astype(jnp.float32) / sc_f)   # non-linear, per-element on the full [B,S,V]
    logits = logits.astype(hidden.dtype)
```

Three fold-in options, each fails:

1. **Fold softcap into `weight`**: `softcap(x W^T) = sc * tanh(x W^T / sc)` is non-linear in `x W^T`. There is no weight transform `W'` such that `x W'^T = sc * tanh(x W^T / sc)` for all `x`. Algebraically impossible.
2. **Fold softcap into `hidden`**: same obstruction. `sc * tanh(x W^T / sc)` is not a function of a transformed `hidden` applied linearly; the non-linearity sits *after* the matmul.
3. **Apply softcap after the fused kernel**: would require the kernel to return `[B, S, V]` logits — but the whole reason to use this kernel is that it **never materializes** `[B, S, V]`. Applying softcap externally means materializing `[B, S, V]`, i.e., going back to the status quo (and worse — now we have two passes: one through the fused kernel and one over the materialized logits). Complete defeat.

The only viable path is a **kernel fork** that applies `sc * tanh(logits / sc)` inline on each VMEM tile of `logits` before the streaming softmax — i.e., a new Mosaic-TPU kernel. This was already catalogued in [program.md § "Pallas kernels to BUILD"](../../program.md) as "**Fused final logit softcap + log-softmax + NLL**" — a build, not a drop-in.

### Violating the softcap is off-limits by program contract

`program.md § "What you CANNOT do"` explicitly: **"Skip `final_logit_softcapping=30.0`."** The baseline loss at step 4 with softcap is 2.1875 (exp 36); without softcap the model trains a different distribution — an `-invalid` outcome on two counts (architecture contract + semantic divergence).

## Additional API mismatches (noted for a future revisit)

Even if softcap were handled, the tokamax API has further gaps that would bite:

- **No `ignore_index`**. The training labels include `IGNORE_INDEX=-100` padding tokens that must not contribute to the loss. The JAX stack handles this with an explicit `mask = (flat_labels != IGNORE_INDEX)` in `train.py:235`. A `-100` index into tokamax's one-hot `jax.nn.one_hot(labels, V)` produces the all-zeros row (by `jax.nn.one_hot`'s under-range semantics) — so the `-100 → 0` contribution happens to cancel numerically, but the sum-divided-by-mask-sum scaling still differs from tokamax's `mean` over all `B=B*S` tokens. Needs a one-hot label remap + post-loss rescale by `mask.sum() / B`. Fixable at the call site, but adds ~15 lines.
- **`labels` signature is `Integer[Array, "B"]` (flat indices)**, with one-hot done internally. Matches what we have after `labels.reshape(-1)`. Fine.
- **`reduction="sum" | "mean"` only**. We want `sum_over_valid / mask.sum()` (ignore-aware mean). Not natively expressible — workaround as above.
- **`x` shape must be `[B, H]`** (flat, not `[B, S, H]`). We can reshape `hidden` to `[B*S, H]` before passing; the kernel batches over the leading dim. Fine.

None of these would cause an `-invalid` on their own — they're S-effort glue. The softcap is the hard blocker.

## Changes made

**None.** No edits to `train.py` or `model/modeling_gemma4.py`. The gated `JAX_CE_IMPL={default,tokamax}` env-var was not introduced because the `tokamax` branch cannot be implemented correctly. Exp 43 is reported on API inspection alone, consistent with the decision-tree exit "**Softcap incompatible → `-invalid`. Don't merge.**"

## Results

| Metric | exp 36 (b=3, splash, bf16 CE) | **exp 43 (tokamax CE attempt)** | Δ |
|---|---|---|---|
| **TPS (median, steps 6–15)** | 34,614 | **not measured** | — |
| Step time (median 6–15) | 355.0 ms | — | — |
| Peak HBM | 27.11 GiB (86.75 %) | — | — |
| Loss (step 4) | 2.1875 | — | — |
| Code churn | — | **0 lines** | — |
| Compile cost | 167 s | — | — |

## Profile

**No profile captured.** Run not executed per decision-tree exit on API-inspection. No `raw/profiles/2026-04-23-gemma4-jax-exp43-tokamax-ce/` directory was created. Per `SCHEMA.md § Behavioral rules 7`: "missing profile ⇒ verdict is `inconclusive` at best" — but the verdict here is `invalid` (API-precondition failure, not an inconclusive run), so a profile is not required. Per `SCHEMA.md § experiment template`: "If the run was not executed (e.g., an infrastructure-only dry check), omit the section and note the reason in `## Verdict`."

## Verdict

**INVALID / rejected.** No experiment was run. The tokamax public API (`tokamax.linear_softmax_cross_entropy_loss`) **does not support Gemma 4's `final_logit_softcapping=30.0`** — there is no `logits_soft_cap` (or equivalent) kwarg in `api.py`, `base.py`, `reference.py`, or the Pallas TPU kernel. Applying softcap externally to the `[B, S, V]` logits defeats the kernel's sole purpose (never materializing `[B, S, V]`). Skipping softcap violates the program's architecture contract and would change the model's output distribution.

Commit stays at `c1927ba`. Exp 36 (34,614 TPS, 86.75 % HBM) remains the JAX-stack best.

## What we learned (durable)

1. **tokamax's current public CE kernel is architecture-agnostic** — no softcap, no logit-transform hook. Gemma-family models (Gemma 2, Gemma 3, Gemma 4 all use softcap on final logits) cannot use it as-is. Grok-1, Llama, Mistral, DeepSeek do not softcap and could.
2. **The gap catalogued in [program.md § "Pallas kernels to BUILD" → "Fused final logit softcap + log-softmax + NLL"](../../program.md) is now empirically confirmed**: a Gemma-aware variant of tokamax CE that applies `sc * tanh(logits/sc)` inline on each VMEM logits-tile before the streaming softmax is the correct build. Estimated effort: M (extend the existing 690-line Mosaic TPU kernel with a softcap pre-op on the logits accumulator before `log_softmax`); estimated gain: ~1.5 GiB HBM freed + small TPS (2–5 %, HBM-pass elimination on a single logits tile).
3. **Always read the kernel API before planning the experiment.** The program.md Pallas-to-TRY table (line 107) lists tokamax LCE as a drop-in candidate with "~4 GiB HBM freed; smaller compute impact". The table is correct about the potential but silently assumes the architecture has no softcap — a footnote "requires softcap support; Gemma 4 does not qualify" should be added (not in this experiment's scope to edit; logged as a follow-up).

## Follow-ups (ranked)

1. **Kernel build — `gemma_linear_softmax_ce` with inline softcap.** Fork tokamax's `pallas_mosaic_tpu_kernel.py`, add a `logits_soft_cap: float | None` kwarg, apply `sc * tanh(logits / sc)` to the logits tile before `log_softmax` inside the VMEM block. Preserve the streaming (no `[B, S, V]` materialization). Open a new experiment under a fresh number when the kernel builds and runs. Confidence medium (the tokamax kernel does not currently expose a hook; straightforward extension but requires running its test suite to catch VMEM-tile boundary regressions). Effort M.
2. **`collective-permute-done` audit (exp 38-style deep dive).** Still the #3 hotspot at b=3 (12.1 % of step time, 549 ms / 3-step profile). Tighten `in_shardings` / `out_shardings` on the jitted step body. Expected 5–6 % TPS.
3. **Pallas RMSNorm kernel** (program.md build-target). `loop fusion` at 28 % of step time; 210 norm calls/step. Expected 3–8 %.
4. **scan-over-layers** — compile-time win (167 s → ~5 s).

## See also

- [Exp 36 — splash + batch=3 (ACCEPTED, JAX-stack best 34,614 TPS)](2026-04-23-exp36-jax-splash-batch3-accepted.md) — baseline this was meant to beat.
- [Exp 37 — bf16 CE env-var gate on JAX stack (potential, flat by construction)](2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md) — the prior CE-path experiment; the `JAX_CE_DTYPE` gate it introduced is what a future `JAX_CE_IMPL=tokamax` gate would have composed with.
- [program.md](../../program.md) § "Pallas kernels to BUILD" → "Fused final logit softcap + log-softmax + NLL" — the build target this result validates.
- [program.md](../../program.md) § "What you CANNOT do" → "Skip `final_logit_softcapping=30.0`" — the hard constraint that forces `-invalid`.
- [`../model/modeling_gemma4.py`](../model/modeling_gemma4.py) L782–L786 — where the softcap is applied (post-lm_head, pre-loss).
- [`../train.py`](../train.py) L224–L244 — the `forward_loss` the hypothesis proposed to replace.
- Torchax [exp 27 attempt — tokamax.dot_product_attention (failed on sliding-window masks)](../../torchax/experiments/) — sibling tokamax integration attempt; different API, different failure mode. This is the first tokamax-CE attempt on either stack.

## Sources

- [`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py) — public API, 129 lines, no softcap kwarg.
- [`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/base.py) — XLA reference-backed Op wrapper, fwd/bwd signatures both softcap-free.
- [`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py) — 70 lines, confirms `loss = -reduction(one_hot(labels) * log_softmax(x @ w))` with no softcap / no mask.
- [`raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py) — 690 lines, streaming mosaic-tpu implementation, also no softcap hook.
- `tokamax==0.0.12` (editable install at `raw/code/tokamax`); `IMPLEMENTATIONS = {xla, chunked_xla, mosaic_tpu}`; default on TPU is `mosaic_tpu`.
