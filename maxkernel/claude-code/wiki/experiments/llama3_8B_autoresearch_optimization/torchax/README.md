# torchax code — Llama 3 8B autoresearch

Working scripts, notebooks, and configs for running **Llama 3 8B via torchax** (PyTorch-on-JAX) on TPU. This is the **primary** execution path for the program — Llama 3 8B ships as a PyTorch model, and torchax carries it to TPU with minimal code changes.

Companion folder: [`../jax/`](../jax/README.md) — native-JAX port (secondary, lit-up once the torchax baseline is stable and a native port becomes a hypothesis).

## Convention

- Code is tracked directly in the main wiki git repo (not a submodule).
- File types permitted: `.py`, `.ipynb`, `.sh`, `.toml`, `.yaml`, `.json`. Binary artifacts (profiles, HLO dumps, checkpoints) go under `raw/profiles/<experiment>/`.
- Every dated experiment (`../<YYYY-MM-DD>-<slug>.md`) references the exact script + config used by relative path into this folder.
- When a script diverges materially between experiments, save a dated copy (`run_v2026-04-25.py`) rather than overwriting — experiment pages must stay reproducible.

## Expected layout (fill as we go)

```
torchax/
  README.md              this file
  train.py               Llama 3 8B fine-tune trainer (UNTESTED scaffold)
  config.yaml            default args for train.py (CLI overrides)
  requirements.txt       pip install targets (jax[tpu], transformers @ main, ...)
  data.py                wikitext loader + fixed-length packer
  model/                 re-exports HF `Llama4*` classes + the sharding plan
    __init__.py          thin re-exports
    README.md            config + sharding assumptions; canonical source of
                         truth about "where the model code lives"
    sharding.py          get_mesh / plan_shardings — NeMo-Megatron recipe
                         adapted for Llama 3 8B's GQA (kv=2) and TP=8
  <YYYY-MM-DD>-<slug>/   per-experiment script+config when materially divergent
```

## Final status (2026-04-27)

**Frontier**: 6,559 tok/s/chip / 36.8 % MFU at `bs=3 seq=8192 fsdp=8` on v6e-8 — see [exp 72a/74b](experiments/2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md). Stack: scan-over-layers + AMP master fp32 weights / bf16 compute + tokamax CE chunked_xla + tokamax-splash with `use_base2_exp + fuse_reciprocal + max_logit_const=30` + recipe XLA flags. **Cumulative climb 4,591 → 6,559 = +42.9 % per chip** vs the morning baseline.

**vs JAX sibling**: native-JAX [exp 28b](../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) on the same hardware/model reaches ~7,700/chip 43.3 % MFU (peak 7,768/43.6 %) at bs=4 seq=8192 — **+17.4 % per-chip over the torchax frontier**. The remaining gap is largely from torchax dispatch overhead at the framework level + the JAX path being able to use MaxText's full XLA flag stack (HOST_OFFLOAD_FLAGS + SC offload of all 3 FSDP collectives) more effectively. After exp 74b the torchax frontier saturated; the program-target throughput work moved to the [`../jax/`](../jax/README.md) sibling.

**vs MaxText reference**: [MaxText baseline](../maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) at 7,069/chip 44.6 % MFU (bs=3 seq=8192) sits **+7.8 % above the torchax frontier**. The torchax path can't close this gap without either (a) closing the dispatch-overhead gap to native JAX, or (b) writing TPU Pallas kernels that XLA isn't already doing — and 2026-04-27 HLO inspection showed XLA already fuses RMSNorm+matmul and SwiGLU+down_proj into `kind=kOutput` Mosaic kernels (see [refuted hypothesis pages](../../../hypotheses/)), so the latter avenue is closed.

## Running the trainer

### First-time setup

```bash
# Accept the Llama license on HF (the model card is Apache-2.0 / not gated
# as of 2026-04-25, but you still need to log in).
pip install -U huggingface_hub
huggingface-cli login

# Install deps (from this folder).
pip install -r requirements.txt
```

The trainer does **not** download weights itself — `transformers.from_pretrained`
handles that on first run, into the usual `~/.cache/huggingface/hub/`.

### Baseline smoke test

`train.py` expects its sibling `model/` and `data.py` modules on `sys.path`, so run it with `python -m train` from **this folder** (`cd` first). Anchor the profile output at the repo root so all experiment artifacts land under `raw/profiles/`.

```bash
conda activate llama3_py313

WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
PROFILE_DIR="$WIKI_ROOT/raw/profiles/2026-04-25-llama3-baseline"
mkdir -p "$PROFILE_DIR"

export HF_HOME="/mnt/disks/persist/torch-tpu/.cache/huggingface"
mkdir -p "$HF_HOME"

cd "$WIKI_ROOT/wiki/experiments/llama3_8B_autoresearch_optimization/torchax"
python -u -m train \
  --steps 10 \
  --batch_size 1 \
  --seq_len 1024 \
  --profile_dir "$PROFILE_DIR" \
  --profile_steps 5 6 7
```

The `-u` flag is important — it disables Python output buffering, so the `[load]` / `[shard]` / `[step]` log lines stream live instead of arriving in chunks. Reproduces the [2026-04-25 baseline](../2026-04-25-baseline.md) at seq=1024 (clean loss; seq=2048 currently hits NaN — see the baseline page).

### Flags

Full list: `python -m train --help` (from this folder).

| Flag | Default | Notes |
|---|---|---|
| `--model_id` | `meta-llama/Meta-Llama-3-8B` | HF repo. |
| `--dataset` | `wikitext-2-raw-v1` | Pass `wikitext-103-raw-v1` for a real dataset. |
| `--seq_len` | `2048` | Known issue: NaN loss at seq≥2048; seq≤1024 is clean. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size × fsdp` (FSDP) or `× dp` (TP). |
| `--steps` | `20` | Step 0 compiles; step 1 currently recompiles (known). Step-time stats drop step 0. |
| `--strategy` | `fsdp` | `fsdp` (default) shards every param's largest divisible dim across all chips. `tp` is Megatron-style. |
| `--fsdp` | `0` (auto = `jax.device_count()`) | FSDP mesh size; `0` means use every visible chip. Only relevant with `--strategy fsdp`. |
| `--dp` / `--tp` | `1` / `1` | Only used with `--strategy tp`. Require `dp × tp == jax.device_count()`. |
| `--dtype` | `bf16` | `bf16` flips `torchax.enable_performance_mode()`; `fp32` flips `enable_accuracy_mode()`. |
| `--profile_dir` + `--profile_steps` | unset | Capture an xprof trace for the listed step indices. |

Target seq_len: 8k
Target dtype: fp32

### Reporting output

Exit prints a summary table:

```
================ summary ================
compile time (step 0) : 45.32s
steps measured        : 19
mean step time        : 182.4 ms
tokens per step       : 8192
tokens / sec          : 44912
wall clock            : 48.8s
==========================================
```

The baseline experiment page fills these in and references the profile
directory. Numbers above are placeholders — until the scaffold actually
runs, all targets are TBD.

### Known limitations of the trainer (2026-04-27 final state)

- **`num_kv_heads = 2` < `tp = 8`** → K/V projections replicated (correct, suboptimal). See `model/sharding.py`. Mitigated at the program level by using FSDP, not TP, as the primary parallelism axis.
- **Step-1 recompile** (~150 s overhead per run iteration). Open follow-up; not blocking but adds friction to fast iteration.
- **NaN loss at seq≥2048** with the original HF transformers Llama 3 8B path; resolved at seq=8192 once the program migrated to AMP master + tokamax CE + scan (exp 20+).
- **Captured constants** addressed via `JittableModule.functional_call`; no multi-GB warnings observed at frontier shape.

## See also

- [torchax codebase page](../../../codebases/torchax.md) — framework architecture, op-lowering, compile boundary.
- [jax-huggingface codebase page](../../../codebases/jax-huggingface.md) — prior-art patterns for running HuggingFace PyTorch models on TPU via torchax.
- [jax-huggingface part 3 — StaticCache + jax.jit decode](../../../sources/2026-jax-huggingface-part-3.md) — captured-constants and `functional_call` patterns relevant to Llama 3 8B.
