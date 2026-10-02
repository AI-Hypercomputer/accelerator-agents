---
title: "Exp 11 — host-offload remat at seq=2048 batch=2 (REJECTED — OOM same margin)"
type: experiment
tags: [experiment, gemma4, oom, host-offload, rematerialization, hbm-ceiling]
hypothesis: offload-unblocks-seq2048-b2
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: crash
---

> *Backfilled from `RESULTS.tsv` + `OBSERVATIONS.md`.*

Switched remat policy from `checkpoint_dots_with_no_batch_dims` to `offload_dot_with_no_batch_dims` (host-offload flavour) to see if freeing HBM via PCIe-staged activations would close the 1.25 GiB gap from exp 10. **Crashed: 32.48 G vs 31.25 G, 1.24 GiB over — essentially identical margin.** The XLA memory planner does not account for host-offload as freed HBM at compile time.

## Hypothesis

Host-offloaded activations should reduce compile-time peak HBM below 31.25 GiB.

## Result

OOM by 1.24 GiB — unchanged from exp 10. Confirms the `~1.25 GiB compile-overhead pattern` is structural to XLA's compile-time planner, not addressable by changing where activations live at runtime.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp11-splash-seq2048-b2-offload](http://localhost:8791/?run=2026-04-23-gemma4-exp11-splash-seq2048-b2-offload) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp11-splash-seq2048-b2-offload`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp11-splash-seq2048-b2-offload/`](../../../../../raw/profiles/2026-04-23-gemma4-exp11-splash-seq2048-b2-offload/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: none (run did not reach training steps)
- **What's inside**: No runtime trace — compile-time HBM OOM (32.48 G vs 31.25 G, same 1.25 GiB margin as exp 10). HLO dump with `offload_dot_with_no_batch_dims` policy for analysis.

## Verdict

**REJECTED — crash.** Not merged. Host-offload isn't the lever for this bottleneck; would need a different memory model or structural change.

## See also

- [exp 10 — same config without offload](2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md).
- [host-offload concept](../../../../concepts/host-offload.md).

## Sources

- `RESULTS.tsv` row `exp11`.
- Profile directory: `raw/profiles/2026-04-23-gemma4-exp11-splash-seq2048-b2-offload/` — xprof run `2026-04-23-gemma4-exp11-splash-seq2048-b2-offload` at http://localhost:8791/?run=2026-04-23-gemma4-exp11-splash-seq2048-b2-offload

