# Qwen3 — jax lane

Native-JAX (**Flax NNX**) port of **Qwen3** on **TPU v6e-8**. The **model is
implemented and equivalence-verified** against HF PyTorch; the **trainer is not
yet written**. The torchax lane is the reference the JAX lane must match-or-beat
(mirroring the llama3 program's torchax→jax progression).

## Layout

```
jax/
  model/
    modeling_qwen3.py   Flax NNX Qwen3 (RMSNorm, GQA + QK-norm, SwiGLU, RoPE,
                        untied lm_head). XLA-SDPA attention (CPU-runnable);
                        splash + scan-over-layers are later optimizations.
    weight_loader.py    load an HF state_dict into the Flax model (1:1 by name);
                        get_param() navigates a dotted HF key to an nnx.Param.
    __init__.py
  test_equivalence.py   CPU forward+backward equivalence vs HF (see below)
  test_tpu_equivalence.py
                        PyTorch CPU reference vs sharded+jitted JAX TPU path
  README.md             this file
  experiments/          dated experiment writeups (once the trainer lands)
```

Qwen3 vs Llama3 (the port basis): adds **QK-norm** — a per-head `RMSNorm` over
`head_dim` on Q and K, applied after the q/k projection+reshape and before RoPE
(`self_attn.q_norm` / `k_norm`). Drops the Qwen2 QKV bias. Param names match HF
dot-for-dot so `weight_loader` maps keys with no transpose.

## Equivalence test (CPU, no TPU)

`test_equivalence.py` loads an HF Qwen3 `state_dict` into the Flax model and
checks forward logits + per-parameter backward gradients match HF PyTorch (eager
attention). Backward uses `nnx.split` + `jax.value_and_grad` (the functional grad
a trainer takes).

```bash
JAX_PLATFORMS=cpu python test_equivalence.py    # exit 0 = within tolerance
```

VERIFIED 2026-06-02: forward logits max|Δ|≈2.7e-7, loss bit-identical, all 25
parameter grads ≤8e-8 (incl. QK-norm) → **PASS**. Run after any change to the
model or weight loader.

## TPU equivalence test

`test_tpu_equivalence.py` keeps PyTorch on host CPU as the reference, but runs
the native-JAX side through the TPU path: mesh construction, parameter sharding,
input sharding, and a jitted loss+gradient step. It uses a tiny Qwen3 config so
the CPU reference is cheap. By default it refuses to run if JAX selected CPU.

```bash
python test_tpu_equivalence.py              # requires TPU backend
python test_tpu_equivalence.py --dtype=fp32 # stricter debug mode
```

For local syntax/debug only, pass `--allow_cpu`; do not treat that as TPU
coverage.

VERIFIED 2026-06-12 on local v6e-4: backend `tpu`, 4 devices, bf16
sharded+jitted path passed against PyTorch CPU with logits max|Δ| `4.489e-03`,
loss max|Δ| `3.910e-05`, and all 25 parameter gradients within tolerance.

## How to run (trainer — TODO)

```bash
# TODO: native-JAX trainer not yet written. Planned entry:
#   python -m train --steps 20 --batch_size 1 --seqlen 8192
# It will reuse model/ + a sharding plan + a data pipeline (mirror torchax lane).
```

## Recent issues / debugging notes

(append as encountered)
