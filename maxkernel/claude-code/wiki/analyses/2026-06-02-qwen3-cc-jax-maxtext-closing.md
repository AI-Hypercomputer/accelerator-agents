---
title: "Closing the MaxText gap on the Qwen3-8B jax lane — what scan+overlap won, and the documented hard wall"
type: analysis
tags: [qwen3-cc, jax, maxtext, scan, overlap, mfu-gap, closing, analysis]
model: qwen3-cc
created: 2026-06-02
updated: 2026-06-02
---

# Closing the MaxText gap on the Qwen3-8B jax lane (v6e-8)

> [!note] (2026-06-16) The MaxText MFU target referenced here (45.3% / 38.0%) is **non-causal** and overstates the gap — causal-adjusted it's **39.8% / 36.6%**. See the [qwen3-cc-maxtext caveat](../models/qwen3-cc-maxtext.md).

> [!warning] Contradicted by [v034 (maxtext-CE bs2)](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v034-maxtext-ce-s8k-bs2.md) on 2026-06-02
> The "documented hard wall" conclusion below (seq8192 batch anti-amortizes; the gap is MaxText's
> bs3 kernel-pipeline efficiency, not a config lever) is **WRONG**. Porting MaxText's exact CE — the
> T5X `@jax.custom_vjp cross_entropy_with_logits` (`--use_maxtext_ce`) — made batch **amortize**:
> v034 bs2 = **34.4% / 5,992 tok/s/chip > v028 bs1 (5,632)**, reversing the bs1>bs2>bs3 series this
> page treated as a structural wall. The wall was **CE-backward-transient-bound** (the tokamax/autodiff
> CE backward), not collective/attention-bound. The climb toward MaxText (6,942) is **reopened** —
> seq8192 frontier moved v028 → v034, gap 81% → 86.3%, with bs3 (v035) in flight. See the
> [maxtext-CE hypothesis](../hypotheses/qwen3-jax-maxtext-ce.md) and the **[MaxText-CE climb closing analysis](2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md)**
> (the arc's final writeup — seq8192 frontier now v035 34.6% / 6,030 = 86.9% of MaxText). The scan+overlap
> findings below remain valid; only the "hard wall" framing is superseded.

Final synthesis of the autoresearch arc that ran from "implement the missing MaxText optimizations"
(scan-over-layers + named host-offload + collective-overlap flags) through the directive
**"meet MaxText performance or document a hard wall."** This page records what the implementation
achieved, the full experiment arc (v025–v032), and the honest verdict: scan+overlap is a **real
seq8192 win**, but full MaxText parity at seq8192 is blocked by a **documented kernel-pipeline hard
wall** that is not a config lever on the tokamax/JAX stack.

## Headline

| Shape | jax start | jax final (this arc) | MaxText ref | jax-vs-MaxText | verdict |
|-------|-----------|----------------------|-------------|----------------|---------|
| **seq8192** | v009 bs1: 30.4% / 5,305 | **v028 bs1: 32.3% / 5,632** (+6.2%) | bs3: 45.3% / 6,942 | **81.1%** tok/s/chip | win, residual = hard wall |
| **seq2048** | v018 bs4: 35.8% / 6,964 | **v018 bs4: 35.8% / 6,964** (unchanged) | bs4: 38.0% / 7,116 | **97.9%** — parity | scan+overlap refuted here (v032) |

**The implemented package moved the seq8192 frontier +1.9 pp / +6.2%** (30.4% → 32.3%). It did **not**
move seq2048 (already at parity; scan+overlap regresses there). The residual seq8192 gap to MaxText
(81% → 100%) is MaxText's **bs3 per-token kernel-pipeline efficiency** — a documented hard wall, not a
lever we hold.

## What was implemented (the 3-lever MaxText package)

Per the [gap teardown](2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md), MaxText's seq8192 advantage
decomposed into: (1) named host-offload freeing HBM to fit bs3, (2) scan-over-layers keeping the program
small + MXU full, (3) async-collective-fusion overlapping the FSDP reduce-scatter (4.0% async vs our
12.9% synchronous). All three were implemented on the native-JAX lane:

1. **named host-offload remat** — `jax.checkpoint_policies.save_and_offload_only_these_names` offloading
   `checkpoint_name`-tagged proj/mlpwi activations to `pinned_host` (MaxText's `*_proj=offload` recipe).
   Equivalence-passing (CPU fwd+bwd parity vs HF, max|Δ| ~1e-7).
2. **scan-over-layers** — the big NNX refactor: stacked-param storage via `nnx.split_rngs` + `nnx.vmap`
   init (params get a leading `[N=36]` axis), then `nnx.split` → `jax.lax.scan` over **one** compiled
   decoder body. No runtime weight copy (the naive split-at-forward shim OOM'd +16G). HF equivalence
   PASS (scan↔unrolled fwd 9.5e-7, grad 4.3e-8).
3. **collective-overlap flags** — 4 async-collective-fusion XLA flags, stacked on the v018 7-flag
   scheduler bundle.

## Experiment arc (v025–v032)

| v | shape / config | result | what it taught |
|---|----------------|--------|----------------|
| v025 | offload, bs2, seq8192 | fits | revealed the LM-head logit wall (f32[B,L,V]) |
| v026 | offload+CE, bs3, seq8192 | OOM **+2.34G** | narrowed bs3 blocker; CE drops the logit wall |
| v027 | named-offload+CE, bs3 | OOM | real blocker = tokamax CE f32[4096,151936] lm_head-weight all-gather (4.64G), NOT activations |
| v028 | **scan + overlap, bs1, seq8192** | 🏆 **32.3% / 5,632** | scan makes the overlap flags **productive** (they regressed on the unrolled graph, v023). **New frontier.** Compile 35s. |
| v029 | full stack, **bf16 CE weight**, bs3 | 💥 invalid | tokamax mosaic_tpu CE backward **hardcodes f32 scratch** — rejects bf16 lm_head weight (`Invalid dtype for swap`). CE-weight stays f32. |
| v030 | scan+offload+CE(f32), **bs3** | ⚠️ bs3 FITS but **4,595** (−18%) | scan's 1-body program closed v026's +2.34G → bs3 fits; but throughput **regresses** |
| v031 | scan+CE, **bs2**, no offload | ❌ **5,553** (< bs1) | isolates batch from offload: bs2 still < bs1 → the v030 regression is **not** the offload, batch genuinely anti-amortizes |
| v032 | scan+overlap, **seq2048** bs4 | ❌ **6,731** (−3.4% vs v018) | scan+overlap is **seq8192-specific**; regresses at the shorter seq |

## The documented hard wall — seq8192 batch anti-amortizes

The decisive finding is the **monotone batch series at seq8192** (all on the full scan+overlap stack,
918-TFLOPS MFU basis):

```
bs1  5,632 tok/s/chip   (v028)   ← achieved frontier
bs2  5,553              (v031, no offload — isolates batch from offload overhead)
bs3  4,595              (v030, +offload — fits only because scan shrank the program)
```

**Batch does not amortize on our stack at seq8192 — it anti-amortizes.** v031 is the clean control: with
the offload *removed*, bs2 is still below bs1, so the v030 bs3 regression is not an offload artifact — it's
intrinsic. (Consistent with the earlier v016 bs2<bs1 observation, now confirmed to not be a CE artifact.)

MaxText, by contrast, **gains** from bs3 (6,942 > its own bs1). Its advantage is therefore **bs3 per-token
efficiency**, which the [gap teardown](2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) attributes to two
kernel-pipeline-level properties we do not replicate:

- **Better-pipelined offload** — MaxText's profile shows <0.1% host-copy time; its offloaded activations
  stream back overlapped with compute. Our JAX `pinned_host` round-trip is not pipelined that way (the
  3× host traffic at bs3 costs more than the batch occupancy buys).
- **Its own fused cross-entropy** — MaxText's CE avoids the f32[H,V] lm_head-weight all-gather that the
  tokamax mosaic_tpu kernel forces on us (and that kernel hardcodes f32 weight, v029 — we cannot even drop
  it to bf16). The CE-weight gather (4.64G) is what blocks bs3 from fitting without offload in the first
  place.

Both are **kernel-pipeline-level differences, not config levers**. Porting them would mean writing a
hand-fused offload+CE+matmul pipeline to match MaxText's integrated path — a kernel-authoring project, not
a flag or a refactor of the existing stack. **This is the documented hard wall** for the seq8192 batch
dimension on the native-JAX lane.

## Why scan+overlap helped seq8192 but not seq2048

scan+overlap is **not a universal lever** (v028 win, v032 regress). The mechanism is sequence-dependent:
- At **seq8192** the FSDP collective tail is large (12.9% of step, synchronous). scan compiles one body so
  the async-collective-fusion flags can overlap that tail with compute → real win. The scan-body loop-carry
  overhead is amortised against the large collective tail.
- At **seq2048** the collective tail is already a smaller share (shorter matmuls, less to hide), so the
  overlap flags buy little — and the scan-body overhead is now net-negative because there's no large tail to
  amortise it against. Hence −3.4%.

The Knobs matrix records scan+overlap as `seq8192: win / seq2048: regress`.

## Final verdict

- **scan+overlap is a real, banked seq8192 win**: +6.2% tok/s/chip, new frontier v028 (32.3% / 5,632),
  semantics-preserving (HF equivalence PASS), bs1.
- **seq2048 is at parity with MaxText (97.9%)** and unchanged — scan+overlap refuted there; v018 remains
  the frontier.
- **Full MaxText seq8192 parity (45.3% / 6,942) is NOT reachable by config** on this stack. It requires
  MaxText's bs3 per-token efficiency, which is a hand-fused offload+CE kernel pipeline — a documented hard
  wall, surfaced honestly rather than papered over.
- The one remaining *high-prior* lever, if the lane is ever reopened: **a hand-written fused CE** that
  avoids the f32 lm_head-weight all-gather (the bs3-fit blocker) and pipelines like MaxText's. That is a
  kernel-authoring effort, explicitly out of scope for the flag/refactor-level iteration this arc ran.

## See also

- [MaxText-vs-jax MFU gap teardown](2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) — the source analysis that defined the 3-lever package
- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) — model page (frontier + Knobs matrix)
- [Qwen3 8B — maxtext](../models/qwen3-cc-maxtext.md) — reference-ceiling lane
- v028 [scan+overlap seq8192 frontier](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v028-scan-overlap-s8k-bs1.md) · v030 [scan+offload bs3](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v030-scan-offload-ce-s8k-bs3.md) · v031 [bs2 control](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v031-scan-ce-bs2-nooffload-s8k.md) · v032 [seq2048 refuted](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v032-scan-overlap-s2k-bs4.md)
- [jax lane retrospectives](2026-06-02-qwen3_cc-jax-retrospective.md)

## Sources

- xprof traces: `raw/profiles/2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1/`, `.../v030-scan-offload-ce-s8k-bs3/`, `.../v031-scan-ce-bs2-nooffload-s8k/`, `.../v032-scan-overlap-s2k-bs4/`
- MaxText reference: `raw/profiles/` MaxText Qwen3-8B run (per the gap teardown)
