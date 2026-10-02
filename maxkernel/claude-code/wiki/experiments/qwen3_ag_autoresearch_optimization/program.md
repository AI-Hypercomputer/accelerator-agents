---
title: "autoresearch — Qwen3 8B (model-family overrides)"
type: program
model: qwen3_cc
created: 2026-06-02
updated: 2026-06-02
---

# autoresearch — Qwen3 8B (model-family overrides)

Model-family specialization for the **Qwen3 8B** family. Inherits from `../program.md`; overrides the sections below.

See `../program.md` for the inheritance rules. Sections defined here fully replace the parent's; sections not defined here inherit unchanged.

## Setup

The repo contains 2 execution lanes for the same model architecture:

| Lane | Stack | Trainer entry point |
|------|-------|---------------------|
| torchax | HuggingFace PyTorch wrapped via torchax (PyTorch-on-JAX) | `python -m train` from `qwen3_ag_autoresearch_optimization/torchax/` |
| jax | from-scratch Flax NNX port (not yet implemented) | TODO — `python -m train` from `.../jax/` once written |

Model code lives directly in the lane folder (tracked in the wiki repo, not a submodule):
`wiki/experiments/qwen3_ag_autoresearch_optimization/<lane>/`.

**Model sizes**: 8B (`Qwen/Qwen3-8B`). The trainer's `--model_id` can point at the
smaller tied-embedding sizes for fast smoke tests.

**Architecture invariants** (do NOT change without approval — these define the
trained-model identity):

| Field | Value (Qwen3-8B) |
|-------|------------------|
| n_layers | 36 |
| n_heads (q) | 32 |
| n_kv_heads | 8 (GQA 4:1) |
| head_dim | 128 |
| hidden_dim | 4096 |
| intermediate (FFN) | 12288 |
| vocab_size | 151936 |
| rope_theta | 1000000.0 |
| QK-norm | **yes** — per-head RMSNorm on Q and K before RoPE (Qwen3-specific) |
| QKV bias | none (dropped vs Qwen2) |
| tie_word_embeddings | false |

> TODO: confirm the exact config values against `Qwen3Config.from_pretrained("Qwen/Qwen3-8B")`
> on the box (transformers is not importable in the wiki base env).

Before starting:
- Read this file (model-level) and the root `../program.md`.
- Read the lane `<lane>/README.md` (hardware/operational notes).
- Read the model page: `wiki/models/qwen3-cc-<lane>.md`.
- Read the last 2–3 experiment pages in your lane.

## Qwen3-specific CAN additions

(Additive on top of root `program.md`'s "What you CAN do". Root section still applies.)

- **Preferred edit target**: `qwen3_ag_autoresearch_optimization/<lane>/` —
  `train.py`, `model/sharding.py`, `data.py`, `helper.py`, `config.yaml`.
- The minimal baseline deliberately ships WITHOUT splash attention, scan, tokamax
  CE, AMP master weights, or per-layer remat — each is an open optimization to add
  as its own experiment (see the model page's hypothesis list).
- **Qwen3-specific kernel target**: fused **QK-norm + RoPE** (the `q_norm`/`k_norm`
  RMSNorm sits right before RoPE on the same `[B, S, H, D]` tensor — a single-pass
  Pallas kernel candidate that Llama3 doesn't have).

## Qwen3-specific CANNOT additions

(Additive on top of root `program.md`'s "What you CANNOT do".)

- Change `--model_id` away from a `Qwen/Qwen3-*` base checkpoint (instruction-tuned
  `-Instruct` variants are different models).
- Remove or approximate **QK-norm** — it is part of Qwen3's identity.
- Quantize weights below bf16; skip/approximate the optimizer step, norms, or
  residuals.

## The goal

Maximize MFU at Qwen3-8B on TPU v6e-8. **Target sequence length: 8192.**

The torchax lane is the live frontier; the jax lane (once written) serves as the
reference ceiling for cross-lane comparison, mirroring the llama3 program's
torchax→jax→MaxText progression.

## Branching model — branch naming

Inherits the parent's fork-per-experiment pattern. Project-level branch naming:

```
v6e8-qwen3-8b-${LANE}-$(date +%Y%m%d)-exp${NN}-${slug}
```

Example: `v6e8-qwen3-8b-torchax-20260602-exp00-baseline`.

**Trunk branch**: TODO (set when the code fork's trunk is established).

**Docker image base name**: TODO (e.g. `qwen3-container`). Full image tag = `<base>:<branch-name>`.

## Profile + HLO storage

Both profile and HLO dump for an experiment live under the same GCS slug folder.
The path includes the model name to avoid collisions with other model families
sharing the bucket:

```
gs://<your-bucket>/autoresearch/qwen3_cc/<run-name>/
├── plugins/profile/<ts>/<host>.xplane.pb    ← written by jax.profiler (xprof-mcp reads this)
└── hlo/module_NNNN.*.{txt,pb}               ← from XLA_FLAGS=--xla_dump_to=...
```

`<run-name>` convention: `<YYYY-MM-DD>-qwen3-<lane>-exp<NNN>-<slug>`. The xprof
server is launched with `--logdir=gs://<your-bucket>/autoresearch/qwen3_cc`
so each `<run-name>` shows up as a run. On-disk HLO dumps (gitignored) still go to
`raw/profiles/<YYYY-MM-DD>-<exp-slug>/`.

**Bucket confirmed writable** (2026-06-02): smoke profile streamed direct to
`…/qwen3_cc/2026-06-02-qwen3-torchax-smoke` and loaded via xprof-mcp.

## Per-lane operational details

All selected lanes are covered in this file (consolidated; no per-lane `program.md`).

### torchax — HuggingFace PyTorch via torchax (PRIMARY)

**Conda env**: `py312` (Python 3.12). Activate before any run.

**Trainer entry**: `python -m train` from
`wiki/experiments/qwen3_ag_autoresearch_optimization/torchax/`.

**Baseline command**:
```bash
conda activate py312
cd wiki/experiments/qwen3_ag_autoresearch_optimization/torchax
python -m train --steps 20 --batch_size 1 --seqlen 8192 \
    --profile_dir $PROFILE_DIR --profile_step 10
```
`$PROFILE_DIR` = repo-root-anchored `raw/profiles/<YYYY-MM-DD>-<slug>/`.

**Mandatory env on every launch**:
```
TORCH_DEVICE_BACKEND_AUTOLOAD=0   # REQUIRED in py312 — suppresses torch_tpu's
                                  # privateuse1='tpu' autoload so torchax can claim
                                  # privateuse1='jax'. Without it, import torchax raises
                                  # "register_privateuse1_backend() has already been set!".
HF_TOKEN=<token>            # Qwen3-8B weight download (config is public/no-token; see
                            # project memory: ~/.cache/huggingface/token)
JAX_COMPILATION_CACHE_DIR=<persistent path>   # amortize cold compile (~84 s cold on v6e-4)
XLA_FLAGS=--xla_dump_to=$HLO_DIR --xla_dump_hlo_as_text   # recommended for HLO inspection
# --xla_tpu_* flags go in LIBTPU_INIT_ARGS, NOT XLA_FLAGS.
```

**Conda env note**: py312 ships both `torch_tpu` and `torchax` (they conflict over
the PyTorch `privateuse1` backend) — the autoload flag above resolves it. `fire`
was pip-installed into py312 (it was missing). gsutil on this box needs
`unset CLOUDSDK_PYTHON` (broken pyOpenSSL via `/etc/environment`); profiling sidesteps
it by writing direct to `gs://`.

**Quirks**:
- `axis_types=(Auto, Auto)` mesh is required — newer JAX defaults to Explicit,
  which breaks gather ops in HF Qwen3's forward.
- HF Qwen3 RoPE `inv_freq` is a non-persistent meta buffer; `train.py`'s
  `_materialize_buffers_replicated` regenerates it from `rope_theta`. Don't remove.

**Profiling**: `train.py` captures `[--profile_start_step, +--profile_steps)` via
the JAX profiler (`profiling.py`). Pass `--profile_dir gs://…` to stream the trace
**directly to GCS** (verified; ADC creds present), or a local path with
`--profile_gcs_dir gs://…` to upload after. xprof-mcp reads it once the run dir is
under the xprof server's `--logdir`.

**Known hang surfaces**: TODO (fill from first real runs).

**Kernels available**: baseline uses XLA-through-HF for everything. splash
attention / tokamax CE / fused QK-norm+RoPE are open hypotheses, not wired yet.

### jax — Flax NNX (SECONDARY, not yet implemented)

**Conda env**: `py312`.

**Trainer entry**: TODO — port not written. Lit up once the torchax baseline is
stable and a native-JAX port becomes a ranked hypothesis.

## Sources

- Model pages: `wiki/models/qwen3-cc-torchax.md`, `wiki/models/qwen3-cc-jax.md`
- Lane READMEs: `torchax/README.md`, `jax/README.md`
- Cluster docs: `.env/` directory at repo root.
- Wiki schema: `SCHEMA.md`.
- Parent program: `../program.md`.
