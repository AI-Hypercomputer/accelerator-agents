# jax code — Gemma 4 E4B autoresearch (native JAX / Flax NNX)

Native-JAX port of Gemma 4 E4B with a Flax NNX model, a HuggingFace
safetensors weight loader, a JAX sharding plan mirroring
`../torchax/model/sharding.py`, and a trainer that mirrors the torchax
trainer's CLI, flags, and summary block.

Companion folder: [`../torchax/`](../torchax/README.md) — the primary
PyTorch-on-JAX execution path against which the native-JAX port's
correctness + performance are measured.

## Layout

```
jax/
  README.md                 this file
  train.py                  native-JAX trainer (CLI parity with torchax/train.py)
  data.py                   wikitext loader + packer (mirror of ../torchax/data.py)
  requirements.txt
  run.sh                    wrapper (sets XLA_FLAGS / LIBTPU_INIT_ARGS)
  model/
    __init__.py             re-exports
    modeling_gemma4.py      Gemma 4 text tower — Flax NNX
    weight_loader.py        HF safetensors -> NNX Param tree
    sharding.py             FSDP / TP mesh + param-sharding plan
  tools/
    parity_layer.py         per-layer numerical parity vs HF PyTorch
    parity_attn.py          attention-only parity vs HF
    parity_check.py         whole-model parity (slow — CPU only)
```

## Scope

Text-only tower. Skips:
- audio (Gemma4Audio*)
- vision (Gemma4Vision*)
- multimodal orchestrators (Gemma4Model, Gemma4ForConditionalGeneration, Gemma4MultimodalEmbedder)
- MoE (Gemma4TextExperts, Gemma4TextRouter) — E4B is dense per-layer

Includes the PLE (per-layer embeddings) residual path since it's in the
E4B checkpoint and affects the forward.

## Architecture port notes

Confirmed against `transformers 5.x` Gemma 4 source:

- `Gemma4RMSNorm`: weight init = **ones**, forward = `normed * weight`
  (NOT `(1 + weight)` — that was the Gemma 2/3 convention). Compute in
  fp32, cast back at the end.
- `Gemma4TextAttention.scaling = 1.0` — HF deliberately disables the
  usual `1/sqrt(head_dim)` scaling because q_norm / k_norm already
  normalize per-head. Baked into `Gemma4TextAttention` in this port.
- Hybrid layer pattern: 5 sliding (head_dim=256) + 1 full (head_dim=512,
  partial-RoPE 0.25) repeating. Last layer is full. E4B has 42 layers
  (7 full + 35 sliding).
- KV sharing: last 18 of 42 layers read (k, v) from an earlier same-type
  layer. Their own `k_proj` / `v_proj` / `k_norm` / `v_norm` weights in
  the HF checkpoint are dead and are dropped by the weight loader.
- Tied `lm_head`: `tie_word_embeddings=True`. This port does **not**
  instantiate a separate `lm_head` Param — the forward does
  `hidden @ embed_tokens.weight.T` so there's one source of truth and
  one sharding to manage.

## Running

From this folder, with `$HOME/miniconda3/envs/gemma4_py313`
active:

```bash
# default: batch=4 seq=2048 steps=20 on v6e-4 FSDP.
python -m train

# smoke test first:
python -m train --batch_size 1 --seq_len 1024 --steps 20

# capture a profile around steps 10..12:
WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
python -m train --batch_size 1 --seq_len 1024 --steps 20 \
  --profile_dir "$WIKI_ROOT/raw/profiles/2026-04-23-gemma4-jax-baseline" \
  --profile_steps 10 11 12
```

CLI flags mirror `../torchax/train.py`:
`--model_id --dataset --seq_len --batch_size --steps --learning_rate
 --warmup_steps --strategy fsdp|tp --fsdp / --dp / --tp --dtype bf16|fp32
 --profile_dir --profile_steps --log_every --grad_accum --seed --config`.

Summary block is identical to the torchax trainer's so the shared
`awk '/step  2/,/step 19/ ...'` tooling keeps working.

## Parity checks

Before trusting performance numbers, verify semantic parity against HF
PyTorch. The layer-level check takes ~30s on CPU and catches nearly all
bugs:

```bash
python tools/parity_layer.py
# expected output:
#   [rms] max=0
#   [mlp] max ~ 1e-3
#   [rope ...] max=0
#   [layer 0 type=sliding_attention] max ~ 3e-2
#   [parity] PASS
```

A full-model CPU parity check (`tools/parity_check.py`) exists but takes
many minutes (a 7B CPU forward is not fast). Prefer the layer check.

## Known limitations of this pass

- **Attention = XLA SDPA**, not splash Pallas. A follow-up hypothesis
  (see `../2026-04-23-exp34-*`) wires splash in analogously to
  `../torchax/model/pallas_attention.py`.
- **No scan-over-layers**. 42 separate `Gemma4TextDecoderLayer`
  instances unroll at compile time. Follow-up: use `jax.lax.scan` or
  `flax.nnx.scan` over layers with identical shapes — but E4B has
  **heterogeneous** layers (sliding vs full) with **different head_dim
  per layer type**, so scan needs two sub-scans.
- **No fused bwd / explicit opt-state bf16 mix**. `optax.adamw` default.
- **No captured-constants concern**: NNX `split/merge` threads Params
  as pytree leaves through `jit` — no constant-capture risk like the
  torchax `JittableModule` path.

## See also

- [torchax companion folder](../torchax/README.md) — reference
  implementation; correctness anchor.
- [scaling-book](../../../codebases/scaling-book.md) — native-JAX idioms.
- [tokamax codebase page](../../../codebases/tokamax.md) — native-JAX
  kernel library; integration cheaper here than through torchax.
