# Qwen3 — torchax lane

Minimal torchax (PyTorch-on-JAX) baseline trainer for **Qwen3 8B** (`Qwen/Qwen3-8B`)
on **TPU v6e-8**, fine-tuning `wikitext-2-raw-v1` packed at a fixed seq_len.
This is the **primary** lane for the qwen3_cc program; the [`../jax/`](../jax/README.md)
lane is scaffolded but not yet lit up.

The trainer is intentionally minimal — it establishes the baseline number.
Optimization knobs (splash attention, scan-over-layers, tokamax CE, AMP master
weights, per-layer remat) are NOT included yet; each lands later as its own
experiment so the win is attributable.

## Convention

- Code is tracked directly in the wiki git repo (not a submodule).
- Every dated experiment under `experiments/<YYYY-MM-DD>-<slug>.md` references the
  exact script + config used by relative path into this folder.
- When a script diverges materially between experiments, save a dated copy rather
  than overwriting — experiment pages must stay reproducible.

## Layout

```
torchax/
  README.md            this file
  train.py             Qwen3 8B fine-tune trainer (minimal baseline)
  config.yaml          default args for train.py (CLI overrides)
  requirements.txt     pip install targets
  data.py              wikitext loader + fixed-length packer + fake loader
  helper.py            compile_step_func (precompile + cost analysis)
  model/
    __init__.py        re-exports HF Qwen3Config/Qwen3ForCausalLM/AutoTokenizer
    sharding.py        FSDP/TP sharding plan (includes Qwen3 QK-norm entries)
  experiments/         dated experiment writeups (one per run)
```

## How to run (smoke test)

```bash
# conda env: py312  (Python 3.12)
conda activate py312
cd wiki/experiments/qwen3_cc_autoresearch_optimization/torchax

# perf-only smoke (random tokens, no HF weight download) — VERIFIED on v6e-4:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python train.py \
    --model_id Qwen/Qwen3-8B --use_real_data False \
    --seqlen 1024 --batch_size 1 --tp_parallelism 1 --train_steps 8

# baseline measurement with profile capture DIRECT TO GCS (xprof-mcp readable):
RUN=$(date +%Y-%m-%d)-qwen3-torchax-baseline
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
JAX_COMPILATION_CACHE_DIR=/tmp/qwen3_jax_cache \
python train.py --steps 20 --batch_size 1 --seqlen 8192 \
    --profile_dir gs://<your-bucket>/autoresearch/qwen3_cc/$RUN \
    --profile_start_step 8 --profile_steps 3
```

**MANDATORY env: `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.** py312 ships `torch_tpu`,
which auto-registers the PyTorch `privateuse1` backend as `'tpu'` on `import torch`.
torchax then can't rename `privateuse1` to `'jax'` and import fails with
`RuntimeError: torch.register_privateuse1_backend() has already been set!`.
Disabling torch's backend-autoload suppresses `torch_tpu` so torchax wins the slot.

`Qwen/Qwen3-8B` weights are a large HF download — export `HF_TOKEN` (token at
`~/.cache/huggingface/token`) before `--use_real_data True`. The model *config*
(needed even for fake data) is public and downloads without a token.

## Profiling → GCS → xprof-mcp

`train.py` captures steps `[--profile_start_step, +--profile_steps)` via the JAX
profiler (see `profiling.py`). `--profile_dir` may be:
- a **`gs://` path** → the profiler streams the trace straight to GCS (verified;
  needs ADC creds, present on this box). This is the path to use.
- a **local path** → captured locally; add `--profile_gcs_dir gs://…/<run>` to
  rsync it up afterward (falls back to `gsutil`, which needs `unset CLOUDSDK_PYTHON`
  on this box — see project memory).

Canonical GCS layout: `gs://<your-bucket>/autoresearch/qwen3_cc/<run-name>/plugins/profile/<ts>/<host>.xplane.pb`.

To browse/analyze a captured run:
```bash
# xprof server (has the binary in py312), logdir = the qwen3_cc parent:
~/miniconda3/envs/py312/bin/xprof \
    --logdir=gs://<your-bucket>/autoresearch/qwen3_cc --port=8791 &

# xprof-mcp HTTP server (env xprof_mcp_py312) so the MCP tools can query it:
PYTHONPATH=/mnt/disks/persist/torch-tpu \
XPROF_URL=http://localhost:8791 \
XPROF_LOGDIR=gs://<your-bucket>/autoresearch/qwen3_cc \
MCP_PORT=8792 \
~/miniconda3/envs/xprof_mcp_py312/bin/python \
    -m xprof_mcp.server.xprof_mcp_server --transport http &
```
Then `list_runs` / `get_overview` / `get_op_profile` (run name includes the
timestamp subdir, e.g. `2026-06-02-qwen3-torchax-smoke/2026_06_02_03_48_35`).

## Qwen3 vs Llama3 notes (for the sharding plan)

- **QK-norm** (`self_attn.q_norm` / `self_attn.k_norm`): per-head RMSNorm over
  `head_dim`, applied to Q/K after projection and before RoPE. NEW in Qwen3;
  replicated in `model/sharding.py`. (A fused QK-norm+RoPE Pallas kernel is a
  future hypothesis, not in the baseline.)
- **No QKV bias** (Qwen2 had it; Qwen3 dropped it).
- **`tie_word_embeddings=False`** for Qwen3-8B → `lm_head.weight` is a distinct
  sharded parameter. (The smaller Qwen3 sizes tie it; if you switch `model_id`
  to a tied size, `lm_head.weight` simply won't appear in the state_dict and the
  sharding map skips it.)

## Equivalence test (CPU, no TPU)

`test_equivalence.py` checks the autoresearch invariant — the torchax+JAX path
the trainer uses produces the same **forward logits and backward gradients** as
plain HF PyTorch, given identical weights. It uses a tiny Qwen3 config (GQA +
QK-norm + SwiGLU + untied lm_head) and the trainer's exact mechanisms
(`JittableModule.functional_call` for forward, `jax_value_and_grad` for backward).

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu \
    python test_equivalence.py    # exit 0 = all within tolerance
```

VERIFIED 2026-06-02: forward logits max|Δ|≈2.7e-7, all 25 parameter grads
≤6e-8 (float32 reduction-order noise) → **PASS**. Run it after any change to the
model wrapper, sharding, or trainer that could perturb numerics.

## Recent issues / debugging notes

- **2026-06-02 — first smoke VERIFIED on v6e-4 (py312).** `Qwen3-8B`, fake data,
  seq=1024, bs=1, fsdp=4, 8 steps. Cold compile ~84 s; ~312 ms/step; loss stable
  at 11.94 (= `ln(151936)`, correct for uniform-random tokens → pipeline sound).
  Profile (steps 4–6) streamed straight to
  `gs://…/autoresearch/qwen3_cc/2026-06-02-qwen3-torchax-smoke` (xplane.pb 106 MB)
  and loaded/analyzed via xprof-mcp. Op profile: `all-reduce-scatter fusion` =
  31 % of step (FSDP collective dominates at this tiny untuned shape; tc_idle
  212 ms/312 ms — communication-bound; expected, not the real baseline).
- **torch_tpu vs torchax privateuse1 conflict** — must launch with
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (see "How to run"). Without it `import torchax`
  raises `register_privateuse1_backend() has already been set! Current backend: tpu`.
- **gsutil broken on this box** — `CLOUDSDK_PYTHON` (set in `/etc/environment`)
  forces gsutil onto a system Python with a version-skewed pyOpenSSL
  (`X509_V_FLAG_NOTIFY_POLICY` AttributeError). Use `unset CLOUDSDK_PYTHON`, or
  `gcloud storage`, or the Python `google-cloud-storage` client. Profiling avoids
  the issue entirely by writing direct to `gs://` from the JAX profiler.
