---
title: "Qwen3 jax — scan-over-layers"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "compile ~4 min → seconds per run; HLO shrink; memory-scheduling regularity that may unlock bs2 @ seq 8192 (+3-6 pp path)"
confidence: medium
effort: L
origin: 2026-06-12-qwen3_cc5-jax-retrospective
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — scan-over-layers

*Hypothesis*: Replacing the 36-iteration Python layer loop with
`jax.lax.scan` over stacked layer params (Flax NNX scan pattern) preserves
step time within noise at the current frontier shape while cutting compile
from ~2×130 s to O(1), and its regularized memory scheduling reduces peak
HBM enough to open bs2 @ seq 8192 (the llama3-evidence lever worth +3–6 pp).

*Mechanism*: Top recommendation of the
[2026-06-12 retrospective](../analyses/2026-06-12-qwen3_cc5-jax-retrospective.md)
(score ~0.6; sibling-model precedent: llama3-8b-jax's 43.6%-MFU frontier
stack uses scan on identical hardware). Implementation: stack the 36
`Qwen3DecoderLayer` param trees (they are homogeneous) and scan with the
save_attn remat policy applied to the scanned body
(`jax.checkpoint` inside the scan body; `checkpoint_name` tags preserved).
Watch the known trap: scan complicates 2D sharding propagation in backward —
may need explicit `shard_as` annotations (index: torch.compile + scan topic).

*Falsification criterion*: step time at the v019 frontier shape regresses
> 2% with scan on, or backward-pass sharding breaks (OOM/wrong layouts that
annotations can't fix) → refuted; compile-time win alone doesn't justify a
step-time regression.

## See also

- [Retrospective (top direction)](../analyses/2026-06-12-qwen3_cc5-jax-retrospective.md)
- [v019 (frontier to preserve)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v019-val50.md)

**Supported** by [v020](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v020-scan.md)
(2026-06-12): compile 3.4× faster, step −0.3%, loss parity bit-level, and an
unplanned **−8.0 GiB live-set** (27.62 → 19.63 GiB at bs1/seq8192). Merged to
trunk 11e8f9d as `--use_scan`. The −8 GiB re-opened the batch axis (v021).
