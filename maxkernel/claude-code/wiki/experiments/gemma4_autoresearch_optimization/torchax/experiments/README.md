# Gemma 4 E4B — torchax stack experiments

Experiment writeups + ledger for the **torchax** (PyTorch-on-JAX) path through the Gemma 4 autoresearch program. Sibling of [`../../jax/experiments/`](../../jax/experiments/README.md) (native-JAX stack).

## Contents

- `2026-04-22-baseline.md` — initial reference run (not a hypothesis test).
- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. Verdict suffixes: `-accepted` / `-rejected` / `-potential` (see [program.md § Experiment verdict suffix](../../program.md)).
- `OBSERVATIONS.md` — skim-and-reason aggregation log threading the torchax session's arc.
- `RESULTS.tsv` — machine-readable ledger (gitignored; tab-separated: `exp_id date tps mfu_pct step_time_ms peak_hbm_gib config status description`).

## Current best (trunk)

**Exp 25** at **33,372 TPS** (seq=1024, batch=3, fsdp=4, bf16) — +9.2 % over baseline-seq1024. Stack: splash Pallas (block=1024, SEQ_MINOR, fused_bwd) + bf16 CE + selective remat. See [2026-04-23-exp25-splash-block1024-accepted.md](2026-04-23-exp25-splash-block1024-accepted.md) and the [ceiling analysis](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).

## See also

- [`../train.py`](../train.py) — the torchax trainer (primary runner).
- [`../model/`](../model/) — sharding + Pallas attention wiring.
- [Shared program protocol](../../program.md).
