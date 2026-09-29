---
title: "Gemma 4 E4B — TPU Autoresearch Optimization"
type: experiment-program
tags: [program, model-under-optimization, gemma, torchax, jax, active]
created: 2026-04-22
updated: 2026-04-22
status: active
hardware: "TPU v6e (primary)"
framework: "torchax (PyTorch-on-JAX); optional native-JAX port"
model_source: "https://huggingface.co/google/gemma-4-E4B"
---

# Gemma 4 E4B — TPU Autoresearch Optimization

A long-running research program that imports Google's **Gemma 4 E4B** via Hugging Face, gets it running on TPU through **torchax** (PyTorch-on-JAX), optionally converts to native JAX, and applies the Karpathy-style **autoresearch loop** to optimize its performance on TPU.

**This is an optimization program, not a quality program.** Per the wiki's scope rule, any change that alters model semantics (output distribution, accuracy, convergence) is invalid — regardless of speedup. We optimize step time, MFU, tokens/sec, and memory only. Outputs of the optimized model must match the reference within bf16 numerical tolerance.

Individual experiments for this program live as dated files in **this folder**: `<YYYY-MM-DD>-<slug>.md`. Working scripts and configs are split by execution path: [`torchax/`](torchax/README.md) (primary — PyTorch-on-JAX) and [`jax/`](jax/README.md) (secondary — native-JAX port). Profiles for each experiment are captured to `raw/profiles/<YYYY-MM-DD>-<slug>/`. See the "Schema note" section at the bottom.

## Target metrics

| Metric | Direction | Notes |
|---|---|---|
| Step time (training) | ↓ | Wall-clock per training step, excluding compile / warmup. |
| Per-token latency (inference) | ↓ | Decode step latency for autoregressive generation. |
| MFU | ↑ | Model FLOPs Utilization against v6e bf16 peak. |
| Tokens/sec | ↑ | Aggregate throughput (training) or generation rate (inference). |
| Tokens/chip/sec | ↑ | Normalizing TPS per chip. |
| Peak HBM | ↓ | Enables larger batch / longer seq / fewer shards. |

## Hardware

- **Primary**: TPU v6e (Trillium) — the target platform.
- **Reference**: TPU v5e / v5p if available (for generation-comparative roofline calibration).
- **Out-of-scope**: GPU (performance claims from GPU sources are calibration-only, not portable).

## Model

- Source: [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B).
- Architecture: Gemma family (GQA attention, SwiGLU FFN, RMSNorm). Confirm exact config on baseline ingest.
- Initial precision: bf16 weights + bf16 activations + fp32 optimizer states; `default_matmul_precision='bfloat16'`.

## How to run

Scripts live in [`torchax/`](torchax/README.md) (primary), [`jax/`](jax/README.md) (secondary, native-JAX port), and [`maxtext/`](maxtext/README.md) (reference baseline — runs upstream MaxText with a wiki-local `gemma4-e4b.yml` config since MaxText doesn't ship one). The runbook below reproduces the [2026-04-22 baseline](torchax/experiments/2026-04-22-baseline.md). For the current MaxText E4B-shape number on v6e-8 see [maxtext/experiments/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md](maxtext/experiments/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md) (**282.9 TFLOP/s/device, 10,003 TPS, 30.8 % MFU; approximation — see caveat**).

### Prerequisites

- A TPU v6e host (v6e-4 or v6e-8). `jax.device_count()` drives the FSDP mesh — whatever's visible is what's used.
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
- A Hugging Face account; accept the Gemma 4 license once at [huggingface.co/google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) (Apache-2.0, not gated, but login is still required for the hub). About 15 GB of free disk for the weights cache.

### One-time setup

```bash
# From the wiki repo root.
cd /mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki

# Create the env and activate it (activate it once per shell from here on).
conda create -n gemma4_py313 python=3.13 -y
conda activate gemma4_py313

# Python deps.
pip install -r wiki/experiments/gemma4_autoresearch_optimization/torchax/requirements.txt
# torchax from the local submodule (matches the wiki's pinned commit).
pip install -e raw/code/torchax
# transformers needs accelerate for device-map handling on recent versions.
pip install accelerate

# Hugging Face auth.
pip install -U "huggingface_hub[cli]"
hf auth login
```

### Smoke run — reproduces the baseline

```bash
# Activate the env once per shell (skip if already active).
conda activate gemma4_py313

# Anchor the profile path at the repo root; run from the torchax folder so
# that `python -m train` finds its sibling `model/` and `data.py` modules.
WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
PROFILE_DIR="$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-smoke"
mkdir -p "$PROFILE_DIR"

cd "$WIKI_ROOT/wiki/experiments/gemma4_autoresearch_optimization/torchax"
python -m train \
  --steps 20 \
  --batch_size 1 \
  --seq_len 1024 \
  --profile_dir "$PROFILE_DIR" \
  --profile_steps 5 6 7
```

**Expected output on v6e-4**: step 0 + step 1 each compile for ~150 s; steady-state from step 2 onward is **~134 ms/step** at seq=1024 (the [2026-04-22 baseline](torchax/experiments/2026-04-22-baseline.md) reports the full seq-length sweep). Loss descends from ~3.9 to ~2.0 over 10 steps. Profile traces for steps 5–7 land under `$PROFILE_DIR`.

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--strategy` | `fsdp` | `fsdp` shards every param's largest divisible dim across all chips (fine-tuning default). `tp` is Megatron-style. |
| `--fsdp` | `0` (auto = `jax.device_count()`) | Only used with `--strategy fsdp`. |
| `--dp` / `--tp` | `1 / 1` | Only used with `--strategy tp`. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size × fsdp` (FSDP) or `× dp` (TP). |
| `--seq_len` | `2048` | **Known issue**: NaN loss at seq≥2048 on the current scaffold; seq≤1024 is clean. See the baseline page. |
| `--dtype` | `bf16` | `bf16` activates `torchax.enable_performance_mode()`. |
| `--profile_dir` / `--profile_steps` | unset | Set both to capture an xprof trace for the listed steps. |

### What lives where

- `torchax/train.py` — the trainer (argparse + train loop, ~500 lines).
- `torchax/model/sharding.py` — FSDP + TP sharding plans.
- `torchax/data.py` — `wikitext-2-raw-v1` loader + fixed-length packer.
- `torchax/run.sh` — convenience wrapper that sets `XLA_FLAGS` / `LIBTPU_INIT_ARGS` and forwards extra args.
- [`torchax/README.md`](torchax/README.md) — per-file detail + known limitations.

### Filing the experiment page

After a run:

1. Pick a slug: `<YYYY-MM-DD>-<short-description>.md` (e.g. `2026-04-24-splash-attention.md`).
2. Create the file next to this README (in `wiki/experiments/gemma4_autoresearch_optimization/`) — follow the `experiment` page template in [`SCHEMA.md`](../../../SCHEMA.md). The [2026-04-22 baseline](torchax/experiments/2026-04-22-baseline.md) is the working example.
3. **Reference the profile**: the profile dir path goes in both `## Profile` and `## Sources`. Profiles are gitignored (`raw/profiles/*` in `.gitignore`), so the experiment page is the only lineage link on disk.
4. Append a `log.md` entry.

## How to start the optimization loop

The agent-facing protocol for this program lives in [`program.md`](program.md) — a self-contained adaptation of Karpathy's autoresearch for Gemma 4 E4B on torchax/v6e-4. **Read `program.md` first.** Every subsequent experiment follows it.

To start a session:

1. Read [`program.md`](program.md) end-to-end (~10 min). Resolve every fixed binding (hardware, conda env, baseline command, libtpu version, xprof_mcp server location).
2. Tail [`OBSERVATIONS.md`](torchax/experiments/OBSERVATIONS.md) for the latest state of the loop (what was kept, what was discarded, open follow-ups).
3. Tail [`RESULTS.tsv`](torchax/experiments/RESULTS.tsv) for the current ledger row and `status: keep` baseline.
4. Pick the next hypothesis per `program.md` §"The experiment loop", priority order (profile-driven → follow-up → wiki-driven → heuristics).
5. Run, profile, symlink, query xprof_mcp, diff vs baseline, write up per the template in `program.md`.
6. Decision + commit or revert per §"Decision". Append to the ledger + log.
7. Repeat.

The methodology itself is mutable — improvements surfaced during experimentation commit to `program.md` and log an entry under `## approach evolution` in `OBSERVATIONS.md`.

### Known issues encountered by the scaffold (baseline run)

See the "Scaffold changes applied during this run" table on [2026-04-22-baseline.md](torchax/experiments/2026-04-22-baseline.md). Short list:
- HF checkpoint is multimodal-only — `Gemma4ForCausalLM.from_pretrained` silently re-initializes weights. Load `Gemma4ForConditionalGeneration` and monkey-patch the forward (already done in `train.py`).
- NaN loss at seq≥2048 even with `final_logit_softcapping`; correctness work pending.
- Step 1 recompiles (~150 s, same as step 0). Low-effort follow-up: explicit `in_shardings` on `jax.jit`.
- OOM at batch=4, seq=2048 (attention N×N materialized, no splash).

## Code

Working scripts, notebooks, and configs are split by execution path:
- [`torchax/`](torchax/README.md) — **primary**: Gemma 4 E4B run via torchax (PyTorch-on-JAX). The baseline and most hypotheses exercise this path.
- [`jax/`](jax/README.md) — **secondary**: a native-JAX port of the model, lit up once the torchax baseline is stable and a port becomes a hypothesis (e.g., to drop dispatch overhead or reach JAX-only tooling more cheaply). A JAX port must reproduce the torchax baseline's outputs within bf16 tolerance before its numbers count.

Every dated experiment in this folder references the exact script + config used by relative path. Code is tracked in the main wiki git repo (not a submodule).

## Baseline

Not yet captured. Baseline experiment is the first hypothesis in the ranked list below.

| Metric | Value | Date | Experiment |
|---|---|---|---|
| Step time | TBD | — | — |
| MFU | TBD | — | — |
| Tokens/sec | TBD | — | — |
| Peak HBM | TBD | — | — |

## Current best

Same as baseline until the first experiment with `verdict: supported` lands.

## Known bottlenecks

*To be identified from the baseline profile. Candidates to look for based on Gemma-family architecture and Wave 1/2 findings:*
- Attention kernel path — whether torchax routes to `tokamax.dot_product_attention` / Splash or falls back to a chunked XLA kernel.
- SwiGLU path — tokamax has no TPU Pallas kernel for [gated-linear-unit](../../concepts/gated-linear-unit.md); almost certainly XLA on TPU.
- RMSNorm path — similarly no TPU Pallas kernel for [layer-norm](../../concepts/layer-norm.md) in tokamax; likely XLA.
- Activation memory pressure — Gemma-family E4B has depth + long-context activations that push HBM; candidate for [rematerialization](../../concepts/rematerialization.md) and [host-offload](../../concepts/host-offload.md).
- Captured-constants bloat — torchax jit'd functions can capture Python-held tensors as constants (see [jax-huggingface part 3](../../sources/2026-jax-huggingface-part-3.md) for a 13.48 GB instance).

## Open hypotheses

Ranked candidates consolidated from Wave 1/2 findings, the [xprof-mcp TPU optimization guide](../../sources/2026-xprof-mcp-tpu-optimization.md), and the [Ultra-Scale Playbook](../../sources/2025-ultrascale-playbook.md). Ranks are preliminary — will be refined once the baseline profile identifies which apply to this model.

| # | Hypothesis | Expected | Conf. | Effort | Origin |
|---|---|---|---|---|---|
| 0 | **Capture baseline** — run 10–20 steps, dump profile, characterize (roofline position, duty cycle, top HLO ops, memory peak). | n/a | high | S | Program bootstrap — first experiment must be this. |
| 1 | Enable `tokamax.dot_product_attention` (Splash Attention) in place of the default torchax attention path. | 15–40% on attention time | high | S | [tokamax splash-attention](../../sources/2026-tokamax-splash-attention.md), [xprof-mcp TPU_OPTIMIZATION §6](../../sources/2026-xprof-mcp-tpu-optimization.md) |
| 2 | Autotune attention block sizes (`block_q`, `block_kv`, `block_kv_compute`) for Gemma 4's head dim + GQA grouping. | 5–15% on attention | med | S | [tokamax autotuning](../../sources/2026-tokamax-autotuning.md), [attention-block-sizes](../../concepts/attention-block-sizes.md) |
| 3 | Expose Splash Attention backward-pass block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) to the autotuner — currently hard-wired to 128. | 5–15% on backward attention | med | M | Wave 1 finding |
| 4 | Write/enable a TPU Pallas kernel for SwiGLU (tokamax currently falls back to XLA on TPU). | 10–20% on FFN | med | L | Wave 1 finding, [gated-linear-unit](../../concepts/gated-linear-unit.md) |
| 5 | Write/enable a TPU Pallas kernel for RMSNorm (tokamax currently falls back to XLA on TPU). | 3–8% on norm layers | med | M | Wave 1 finding, [layer-norm](../../concepts/layer-norm.md) |
| 6 | Selective rematerialization via `jax.checkpoint_policies` instead of full activation checkpointing. | ~+2.7% compute for ~70% activation-memory savings — enables larger batch. | high | S | [xprof-mcp TPU_OPTIMIZATION §4.6, §5](../../sources/2026-xprof-mcp-tpu-optimization.md), [rematerialization](../../concepts/rematerialization.md) |
| 7 | Scan-over-layers (`jax.lax.scan` / `ScannedModule`) — compile time O(N)→O(1). | Compile time, not step time. Enables faster iteration on other hypotheses. | high | M | [scan-over-layers](../../concepts/scan-over-layers.md) |
| 8 | Dimension alignment — ensure all batch/hidden dims honor v6e MXU tile (256×256). | 5–15% if any dim is misaligned | med | S | [dimension-alignment](../../concepts/dimension-alignment.md) |
| 9 | Enable XLA async-collective and latency-hiding scheduler flags. | 5–15% if ICI collectives are exposed | high | S | [async-collectives](../../concepts/async-collectives.md), [latency-hiding-scheduler](../../concepts/latency-hiding-scheduler.md) |
| 10 | FSDP mesh design on v6e — tune shard dims to maximize [ici-roofline](../../concepts/ici-roofline.md) headroom. | workload-specific | med | M | [fsdp](../../concepts/fsdp.md), [sharding](../../concepts/sharding.md) |
| 11 | Ring attention for long-context runs (kernel exists in tokamax but not wired to public API). | only if context length is a goal | low | L | Wave 1 finding, [ring-attention](../../concepts/ring-attention.md) |
| 12 | Tile-scaled FP8 (DeepSeek-V3 style: 1×128 activations, 128×128 weights+scales). | 10–30%, requires v6e MXU FP8 support confirmation. | low | L | [Ultra-Scale Playbook](../../sources/2025-ultrascale-playbook.md), [dtype-strategy](../../concepts/dtype-strategy.md) |
| 13 | Host-offload of optimizer state and/or activations. | HBM savings → larger batch or longer context | med | M | [host-offload](../../concepts/host-offload.md) |
| 14 | Eliminate captured-constants bloat in torchax jit'd functions (use `functional_call` pattern). | memory only, but can unblock larger batch | med | S | [jax-huggingface part 3](../../sources/2026-jax-huggingface-part-3.md) |
| 15 | Static-cache / fixed-shape KV cache for decode (if inference is in scope for this program). | ~8× on decode per jax-huggingface part 3 | high (decode only) | M | [static-cache](../../concepts/static-cache.md), [kv-cache](../../concepts/kv-cache.md) |

## Retired hypotheses

*None yet.*

## History

- **2026-04-22**: Program filed. First baseline captured — see [2026-04-22-baseline.md](torchax/experiments/2026-04-22-baseline.md). Hardware was v6e-4 (not v6e-8 as the scaffold assumed). Steady-state at seq=2048, batch=1, FSDP=4: **249 ms/step, ~33k tokens/sec, ~26% MFU** (corrected from earlier `6PT` overestimate that claimed 44%). Loss is NaN at seq=2048 but clean at seq≤1024 — a correctness bug to fix before the first perf hypothesis lands.
- **2026-04-23**: 20-step confirmation run at seq=1024. Loss **3.93 → ~2.0** (noisy at global batch=4, but trending). Steady-state at seq=1024: **134.4 ± 0.5 ms/step**, ~30.5k tokens/sec.
- **2026-04-23**: [Exp 1](torchax/experiments/2026-04-23-exp1-async-collective-flags-rejected.md) — async-collective XLA flag bundle (`--xla_tpu_enable_latency_hiding_scheduler`, `--xla_tpu_enable_async_collective_fusion`, `+fuse_all_gather`, `+multiple_steps`) via `LIBTPU_INIT_ARGS`. **REFUTED**: 134.4 → 168.3 ms (+25 % regression). Scheduler reordered collectives but broke compute-fusion memory locality (convolution fusion +2.5× bytes, loop fusion +1.9× bytes). Correctness preserved. Flags parked, will revisit once larger effective batch is reachable.
- **2026-04-23**: [Exp 2](torchax/experiments/2026-04-23-exp2-pin-out-shardings-rejected.md) — attempted to pin `out_shardings` on `jax.jit` to fix the step-1 recompile. **CRASH** at trace time: tied-weight (`lm_head` ↔ `embed_tokens`) sharding plumbing collapsed to single-device for the duplicate key; jit rejected the mismatch. Reverted. Step-1 recompile remains open (~150 s/run iteration overhead).
- **2026-04-23**: Formalized the loop into [program.md](program.md) + [RESULTS.tsv](torchax/experiments/RESULTS.tsv) + [OBSERVATIONS.md](torchax/experiments/OBSERVATIONS.md). README trimmed to a runbook + pointer; the agent-facing protocol now lives in `program.md`. Experiments 1 and 2 retroactively transcribed to the ledger + observation log.

---

## See also

- [autoresearch codebase](../../codebases/autoresearch.md) — methodology reference this program adapts (loop shape, experiment discipline, fair-comparison metric).
- [torchax codebase](../../codebases/torchax.md) — the framework bringing Gemma 4 (PyTorch) to TPU via JAX.
- [tokamax codebase](../../codebases/tokamax.md) — kernel library backing several open hypotheses.
- [jax-huggingface codebase](../../codebases/jax-huggingface.md) — Hugging Face model on TPU via torchax prior art.
- [xprof-mcp — TPU optimization guide](../../sources/2026-xprof-mcp-tpu-optimization.md) — primary source for hypothesis rationales.
- [scaling-book](../../codebases/scaling-book.md) — TPU scaling reference (chapter sources pending Wave 3).

## Sources

*Accumulates as experiments are filed.*

---

## Schema note

This folder is an **intentional deviation** from `SCHEMA.md` as originally written. The schema specifies `wiki/experiments/<YYYY-MM-DD>-<slug>.md` (flat, one file per dated experiment). Here the experiments for one long-running optimization program are **nested under a program folder** so the program's plan (this `README.md`), its dated experiments, and optionally local code live together. See the log entry dated 2026-04-22 for the rationale.

Experiment files within this folder still follow the schema's `experiment` page template and naming convention: `<YYYY-MM-DD>-<slug>.md`, with the required frontmatter (`hypothesis:`, `model:`, `commit:`, `verdict:`).
