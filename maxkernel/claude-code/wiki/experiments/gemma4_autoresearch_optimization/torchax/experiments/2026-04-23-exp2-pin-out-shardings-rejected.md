---
title: "Exp 2 — pin out_shardings to fix step-1 recompile (CRASH)"
type: experiment
tags: [experiment, gemma4, step-1-recompile, crash, tied-weights]
hypothesis: pin-out-shardings-on-jit
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD (torchax submodule 8f957d1); attempted train.py change; reverted"
verdict: invalid
---

Attempt to fix the ~150 s step-1 recompile by pinning `out_shardings` on `jax.jit` to equal the input shardings. Pre-trace `ValueError` from tied-weight sharding plumbing; reverted and filed. The step-1 recompile remains open — costs ~150 s per run of iteration overhead.

## Hypothesis under test

**Statement**: Step 0 compiles with jit-inferred output shardings that differ from the input shardings. When those outputs are donated into step 1 as its new inputs, their sharding layout doesn't match step 0's input cache key — step 1 misses the cache and retraces. Pinning `out_shardings` to equal `in_shardings` would force step 0's outputs to have the same layout, letting step 1 hit the step-0 compile cache and run at ~134 ms instead of ~150 s. Saves ~2.5 min per experiment run.

Origin: baseline's observed step-1 recompile; [2026-04-22-baseline.md](2026-04-22-baseline.md) "Next hypotheses" #2.

## Setup

- Same config as [baseline-seq1024](OBSERVATIONS.md#baseline-torchax--gemma-4-e4b--v6e-4--fsdp4): `--steps 10 --batch_size 1 --seq_len 1024 --strategy fsdp`.
- Code changes in `train.py`:
  - After `weights = interop.jax_view(jmodel.params)`, re-apply `plan.shardings` via `jax.device_put` to every weight.
  - `weights_out_shardings = {k: plan.shardings.get(k, replicated) for k in weights}`.
  - `opt_state_out_shardings = jax.tree.map(lambda a: a.sharding, opt_state)`.
  - `jitted_step = jax.jit(train_step, donate_argnums=(0, 2), out_shardings=(loss_sharding, weights_out_shardings, opt_state_out_shardings))`.
- Command: identical to baseline.

## Results

**Pre-trace crash** before any step executed:

```
ValueError: Received incompatible devices for jitted computation.
Got argument weights['lm_head.weight'] of main.<locals>.train_step
with shape bfloat16[262144,2560] and with device ids [0, 1, 3, 2]
on platform TPU and explicit output sharding with device ids [0] on platform TPU
```

- Input sharding of `weights['lm_head.weight']`: 4-device (`[0, 1, 3, 2]`).
- `out_shardings` entry as seen by jit: single-device (`[0]`).
- My constructed `weights_out_shardings['lm_head.weight']` came from `plan.shardings['lm_head.weight']` which is `NamedSharding(mesh_4_devices, P('fsdp', None))` — should be 4-device. jit saw something different.

Retried once (identical result). Reverted the `out_shardings` + re-apply-shardings additions. Sanity revert verified the baseline still runs without changes.

## Mechanism

`tie_word_embeddings=True` on Gemma 4 makes `lm_head.weight` ↔ `model.language_model.embed_tokens.weight` share storage. The flow:

1. `model.state_dict()` emits **both** keys (each pointing to the same underlying tensor).
2. The sharding-plan loop applies `jax.device_put(..., plan.shardings[k])` to every state-dict key — so both views get the same full-mesh sharding.
3. `model.load_state_dict(sharded_state, assign=True)` assigns the sharded tensors back to the model's params — but `assign=True` replaces the param's `.data` with the passed tensor, which can **break the tying relationship** between `lm_head` and `embed_tokens`. After this, they may be separate tensors.
4. `JittableModule(model)` extracts params via `extract_all_buffers`. With `dedup_parameters=True`, it removes duplicates keyed on `id(v.elem)`. If (3) left them as distinct jax.Arrays, dedup doesn't fire — both survive.
5. Either way, the key `lm_head.weight` ends up in the `weights` dict.

The specific mechanism by which jit sees the `out_shardings` for `lm_head.weight` as single-device remains unlocalized. The jit internals likely canonicalize the sharding tree and discover an inconsistency specific to the tied key.

## Profile

No trace — run crashed pre-trace at jit-compile due to tied-weight `lm_head`/`embed_tokens` sharding mismatch. No profile directory beyond the crash log.

## Verdict

**CRASH (reverted). Marked `invalid` to signal the attempted change broke tracing; nothing was measured, so there is no number to report.** The hypothesis (step-1 recompile caused by input/output sharding mismatch) is **still viable** — not refuted — but requires a different implementation that routes around the tied-weight dedup. Step-1 recompile remains an open 150 s/run iteration-speed loss.

## Next hypotheses (follow-ups)

1. **Route around tied-weight dedup** by constructing `JittableModule(model, dedup_parameters=False)` and manually tying `lm_head.weight ← embed_tokens.weight` after load. Then `out_shardings` can be built from the single canonical tensor.
2. **`jax.lax.with_sharding_constraint` inside `train_step`** as an alternative to `out_shardings`, applied per-leaf on the returned weights. May bypass jit's tree-pytree validation of tied leaves.
3. **Persistent compile cache** (`JAX_COMPILATION_CACHE_DIR`) — solves step-1 recompile across runs (doesn't help the same-run step-1 case, but amortizes step-0 compile). Much lower effort; worth trying regardless.
4. **Disable donation** (`donate_argnums=()`) — would remove the donated-layout mismatch entirely but costs 2× weight/opt-state memory. At 95% HBM this likely OOMs; recheck after a memory-saving experiment.

## See also

- [Program page](../../README.md).
- [program.md](../../program.md) — the agent protocol (exp02 is the first pre-protocol experiment; this page is retroactively templated).
- [OBSERVATIONS.md § exp02](OBSERVATIONS.md#exp02--pin-out_shardings-to-fix-step-1-recompile--crash) — the reasoning block.
- [2026-04-22-baseline.md](2026-04-22-baseline.md).
- [torchax codebase](../../../../codebases/torchax.md) — `JittableModule`, `extract_all_buffers`.

## Sources

- `/tmp/gemma4_exp2.log` — crash traceback (not persisted).
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` — diff since reverted.
