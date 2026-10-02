---
title: "Exp 26 — scan-over-layers (PARKED, Option A blocked, Option B deferred)"
type: experiment
tags: [experiment, gemma4, scan, compile-time, structural, parked]
hypothesis: scan-over-layers-compile-compression
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp26-scan-over-layers"
verdict: parked
---

Investigated whether the 42-layer `for layer in self.layers` loop in `Gemma4TextModel.forward` can be collapsed into a single `jax.lax.scan`. Goal: cut compile-step-0 (~150 s) to ~5–15 s and potentially reduce activation buffers. **Outcome: both Option A and Option B blocked; landed a diagnostic-only scaffold with graceful fallback; no performance change, no merge to trunk.**

## Hypothesis under test

**Statement**: Python-level layer unrolling materializes 42 copies of the decoder block in HLO, inflating compile time and (secondarily) peak HBM during live-range analysis. `jax.lax.scan` compresses this to one body; compile should drop ~30–60× and step-time may gain 2–8% from shared activation buffers.

Origin: wiki audit on the scaling-book + torchax codebases flagged scan-over-layers as the one remaining unexplored structural compile-time lever.

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp26-scan-over-layers` off trunk at exp 25.
- New file `torchax/model/scan_layers.py` (262 lines). Structure mirrors `pallas_attention.py`/`sharding.py`: `register_scan_over_layers(mesh)` is imported and called once at startup in `train.py`.
- Diagnostic path: import `torchax.train.ScannedModule`, print Option A blockers, attempt Option B (stub that always raises), catch and log `[scan_layers] fallback to unscanned: <reason>`, return False. HF's original `forward` stays in place.

## Why Option A (torchax.train.ScannedModule) does not apply to Gemma 4

`ScannedModule` is torchax's canonical scan wrapper. Five independent blockers, each load-bearing for Gemma 4 semantics:

1. **kwargs assertion**. `ScannedModule.forward` does `assert not kwargs`. `Gemma4TextModel.forward` invokes layers with 6 named kwargs (`shared_kv_states`, `position_embeddings`, `attention_mask`, `position_ids`, `past_key_values`, `**kwargs`). Blown the moment the wrapped forward is called.
2. **Heterogeneous state_dict**. `_stack_layer_weights` does `torch.stack([m.state_dict()[k] for m in modules])` per key. Gemma 4's last 18 of 42 layers have `is_kv_shared_layer=True` and therefore drop certain attention params — the stack would `KeyError`.
3. **Per-layer Python ints**. Each layer carries `layer_idx` and possibly `kv_shared_layer_index` as Python ints that branch the forward. Scan requires traced values.
4. **Side-effectful dict carry**. The forward mutates `shared_kv_states: dict` by index — not a pure pytree carry.
5. **Per-layer indexed inputs**. `per_layer_inputs[i]`, `position_embeddings[layer_type]`, `causal_mask_mapping[layer_type]` are all layer-indexed. Scan requires stacked-along-scan-axis inputs.

Each blocker is independently sufficient to reject ScannedModule. Fixing all five means rewriting `Gemma4TextModel.forward` end-to-end, which is effectively Option B.

## Option B (custom scan with stacked weights + explicit kv carry) — sub-problems

Not implemented in this experiment. The scaffold in `scan_layers.py` documents the seven sub-problems for a future pass:

- **B1** — Homogenize `state_dict`: insert zero-param placeholders on kv-shared layers so stacking succeeds, mask out in the kernel.
- **B2** — Replace dict carry with pytree: `shared_kv_states` becomes `(K_shared, V_shared)` tensors, indexed by traced step.
- **B3** — Stack per-step inputs: `per_layer_inputs` → `[42, B, S, D]` lifted to scan axis.
- **B4** — Stack per-layer weights: one pytree where leading axis is layer.
- **B5** — `layer_scalar` carry: `layer_idx` becomes a traced int carried through scan.
- **B6** — Pure scan body: no Python dict mutation, no Python branches on layer_idx.
- **B7** — Backward correctness: verify `jax.checkpoint` composition around the scan is still per-layer-remat and not per-42-layer-remat (otherwise memory blows up).

Estimated cost: 300–500 lines of careful wiring + a correctness run against stock HF layer loop. Belongs in a dedicated design pass, not a single agent iteration.

## Results

No runtime delta measured — the register call falls back to HF's default forward. Startup prints:

```
[scan_layers] Option A (torchax.train.ScannedModule) does not apply to Gemma 4. Reasons:
[scan_layers]   (1) ... (5) ...
[scan_layers] fallback to unscanned: Gemma 4's 42-layer stack is heterogeneous ...
```

Trainer continues with stock forward. TPS / compile-time / HBM unchanged from exp 25 baseline (this branch).

## Profile

No trace — exp 26 landed a diagnostic-only scaffold; the register call falls back to HF's stock forward. Runtime unaffected vs exp 25.

## Verdict

**PARKED.** Not a kept experiment (no perf change). Not a failure (no regression). Option A is permanently refuted for Gemma 4; Option B is a viable future project with a clear sub-problem decomposition but out of scope for a single autoresearch iteration.

**Not merged to trunk** — scaffold lives only on the exp 26 branch as a design doc. Future Option B work should branch from here or from current trunk and carry this file as the starting analysis.

## Next hypotheses (promoted to exp 27 candidate pool)

- Persistent JAX compile cache (`JAX_COMPILATION_CACHE_DIR`) — orthogonal to scan; addresses the ~150 s compile-step-0 cost from the tooling side, not the HLO side. Does not change TPS but speeds the experiment loop ~30× on cache hits.
- tokamax `dot_product_attention(use_base2_exp=True)` — replaces splash's natural-exp softmax with exp2, which has TPU hardware support. Expected 1–3 % on the attention path.
- 2D mesh (fsdp=2, tp=2) — risky at this batch/seq; keeps in the pool.
- Pallas RMSNorm with hand-rolled `custom_vjp` — revives exp 20 after the pallas-forge autograd gap.

## See also

- [program.md § Pallas kernel landscape and compile-time compression](../../program.md)
- `torchax/model/scan_layers.py` (262 lines, on this branch only) — contains the full blocker analysis as a header docstring.
- [torchax codebase](../../../../codebases/torchax.md) — `torchax.train.ScannedModule` is the Option A entry point.
- Gemma 4 source — `modeling_gemma4.Gemma4TextModel.forward` is the rewrite target for Option B. (No `codebases/transformers.md` page — the HF transformers library is not ingested as a codebase; refer directly to upstream `github.com/huggingface/transformers` at `src/transformers/models/gemma4/modeling_gemma4.py`.)

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/scan_layers.py` (new, 262 lines)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (+2 lines, on branch only)
- No profile captured — no runtime change to measure.
