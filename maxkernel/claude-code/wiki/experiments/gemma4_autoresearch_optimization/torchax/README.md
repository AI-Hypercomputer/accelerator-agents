# torchax code — Gemma 4 E4B autoresearch

Working scripts, notebooks, and configs for running **Gemma 4 E4B via torchax** (PyTorch-on-JAX) on TPU. This is the **primary** execution path for the program — Gemma 4 ships as a PyTorch model, and torchax carries it to TPU with minimal code changes.

Companion folder: [`../jax/`](../jax/README.md) — native-JAX port (secondary, lit-up once the torchax baseline is stable and a native port becomes a hypothesis).

## Convention

- Code is tracked directly in the main wiki git repo (not a submodule).
- File types permitted: `.py`, `.ipynb`, `.sh`, `.toml`, `.yaml`, `.json`. Binary artifacts (profiles, HLO dumps, checkpoints) go under `raw/profiles/<experiment>/`.
- Every dated experiment (`../<YYYY-MM-DD>-<slug>.md`) references the exact script + config used by relative path into this folder.
- When a script diverges materially between experiments, save a dated copy (`run_v2026-04-22.py`) rather than overwriting — experiment pages must stay reproducible.

## Expected layout (fill as we go)

```
torchax/
  README.md              this file
  train.py               Gemma 4 E4B fine-tune trainer (UNTESTED scaffold)
  run.sh                 wrapper that sets XLA_FLAGS + LIBTPU_INIT_ARGS and
                         hands `--profile_dir` / `--profile_steps` to train.py
  config.yaml            default args for train.py (CLI overrides)
  requirements.txt       pip install targets (jax[tpu], transformers @ main, ...)
  data.py                wikitext loader + fixed-length packer
  model/                 re-exports HF `Gemma4*` classes + the sharding plan
    __init__.py          thin re-exports
    README.md            config + sharding assumptions; canonical source of
                         truth about "where the model code lives"
    sharding.py          get_mesh / plan_shardings — NeMo-Megatron recipe
                         adapted for Gemma 4's GQA (kv=2) and TP=8
  <YYYY-MM-DD>-<slug>/   per-experiment script+config when materially divergent
```

## Running the trainer

**Status:** scaffold only. Written but not executed. See the header of
`train.py` and `UNTESTED` markers for the concrete risks.

### First-time setup

```bash
# Accept the Gemma license on HF (the model card is Apache-2.0 / not gated
# as of 2026-04-22, but you still need to log in).
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
conda activate gemma4_py313

WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
PROFILE_DIR="$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-baseline"
mkdir -p "$PROFILE_DIR"

cd "$WIKI_ROOT/wiki/experiments/gemma4_autoresearch_optimization/torchax"
python -m train \
  --steps 10 \
  --batch_size 1 \
  --seq_len 1024 \
  --profile_dir "$PROFILE_DIR" \
  --profile_steps 5 6 7
```

Reproduces the [2026-04-22 baseline](../2026-04-22-baseline.md) at seq=1024 (clean loss; seq=2048 currently hits NaN — see the baseline page).

The `run.sh` wrapper sets `XLA_FLAGS` + `LIBTPU_INIT_ARGS` and forwards extra args, but currently assumes a v6e-8 layout; prefer the explicit invocation above until `run.sh` is refreshed.

### Flags

Full list: `python -m train --help` (from this folder).

| Flag | Default | Notes |
|---|---|---|
| `--model_id` | `google/gemma-4-E4B` | HF repo. |
| `--dataset` | `wikitext-2-raw-v1` | Pass `wikitext-103-raw-v1` for a real dataset. |
| `--seq_len` | `2048` | Known issue: NaN loss at seq≥2048; seq≤1024 is clean. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size × fsdp` (FSDP) or `× dp` (TP). |
| `--steps` | `20` | Step 0 compiles; step 1 currently recompiles (known). Step-time stats drop step 0. |
| `--strategy` | `fsdp` | `fsdp` (default) shards every param's largest divisible dim across all chips. `tp` is Megatron-style. |
| `--fsdp` | `0` (auto = `jax.device_count()`) | FSDP mesh size; `0` means use every visible chip. Only relevant with `--strategy fsdp`. |
| `--dp` / `--tp` | `1` / `1` | Only used with `--strategy tp`. Require `dp × tp == jax.device_count()`. |
| `--dtype` | `bf16` | `bf16` flips `torchax.enable_performance_mode()`; `fp32` flips `enable_accuracy_mode()`. |
| `--profile_dir` + `--profile_steps` | unset | Capture an xprof trace for the listed step indices. |

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

### Known limitations of the scaffold

See the header of `train.py` and `model/README.md`. Most consequential:

- **Untested end-to-end.** Written from the codebase + source page cookbook;
  no step has run.
- **`num_kv_heads = 2` < `tp = 8`** → K/V projections replicated (correct,
  suboptimal). See `model/sharding.py`.
- **Pytree registration** may need patching to the shipped HF version
  (`CausalLMOutputWithPast`, `DynamicCache`, `StaticCache`).
- **Captured constants**: if JAX logs a multi-GB captured-constants warning,
  follow the `functional_call` pattern from
  [jax-huggingface part 3](../../../sources/2026-jax-huggingface-part-3.md)
  — the scaffold already threads weights through `JittableModule.functional_call`,
  so this should be fine, but verify.

## See also

- [torchax codebase page](../../../codebases/torchax.md) — framework architecture, op-lowering, compile boundary.
- [jax-huggingface codebase page](../../../codebases/jax-huggingface.md) — prior-art patterns for running HuggingFace PyTorch models on TPU via torchax.
- [jax-huggingface part 3 — StaticCache + jax.jit decode](../../../sources/2026-jax-huggingface-part-3.md) — captured-constants and `functional_call` patterns relevant to Gemma 4.
