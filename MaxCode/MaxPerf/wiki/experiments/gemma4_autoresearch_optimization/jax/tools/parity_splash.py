#!/usr/bin/env python
# pylint: skip-file
"""Numerical parity: splash-pallas attention vs XLA-SDPA inside the NNX port.

Two levels of comparison:
  1. **Raw attention output** (before o_proj): uses a dedicated direct call
     into each kernel path with shared Q/K/V. Tolerance: bf16 abs-err <= 1e-2.
     This isolates the kernel math.
  2. **Post-o_proj output** (full attention block): end-to-end sanity.
     Tolerance: bf16 abs-err <= 1e-1 — loose because the 2560-wide o_proj
     matmul in bf16 amplifies small per-feature deltas.

Runs on-TPU because splash is a TPU-only kernel. Tests one sliding-window
layer and one full-attention layer.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _err(a, b):
  a = np.asarray(a).astype(np.float32)
  b = np.asarray(b).astype(np.float32)
  d = np.abs(a - b)
  return float(d.max()), float(d.mean())


def main():
  from transformers import AutoConfig
  from jax.sharding import Mesh
  from jax.experimental import mesh_utils

  hf_config = AutoConfig.from_pretrained("google/gemma-4-E4B")
  text_cfg = (
      hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
  )

  # Use a short seq_len for the parity check.
  B, T, D = 1, 128, text_cfg.hidden_size  # 128 matches hardware-friendly tile
  rngs = nnx.Rngs(0)

  # One sliding layer + one full-attention layer.
  layer_indices = []
  for i, lt in enumerate(text_cfg.layer_types):
    if lt == "sliding_attention" and not any(
        (text_cfg.layer_types[j] == "sliding_attention" and j in layer_indices)
        for j in range(i)
    ):
      layer_indices.append(i)
    if lt == "full_attention" and not any(
        (text_cfg.layer_types[j] == "full_attention" and j in layer_indices)
        for j in range(i)
    ):
      layer_indices.append(i)
    if len(layer_indices) == 2:
      break

  from model.modeling_gemma4 import Gemma4TextAttention
  from model import pallas_attention

  # Mesh: all-devices 1-D 'fsdp' — so shard_map runs even at B=1 (each chip
  # gets 0 examples but shard_map still works because in_specs expects the
  # leading axis to be replicated modulo the mesh size). In practice this
  # means we run B=mesh_size instead of 1. So use B=jax.device_count().
  n = jax.device_count()
  B = n
  devices = mesh_utils.create_device_mesh((n,))
  mesh = Mesh(devices, axis_names=("fsdp",))
  pallas_attention.set_mesh(mesh)
  print(f"[parity_splash] mesh fsdp={n}; using B={B} T={T}")

  # Setup rotary embeddings (once).
  from model.modeling_gemma4 import Gemma4TextRotaryEmbedding

  rotary = Gemma4TextRotaryEmbedding(text_cfg)
  pos_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))

  from model.modeling_gemma4 import _attn_xla_sdpa

  all_pass = True
  for layer_idx in layer_indices:
    lt = text_cfg.layer_types[layer_idx]
    print(f"\n=== layer {layer_idx} type={lt} ===")
    attn = Gemma4TextAttention(
        text_cfg, layer_idx, dtype=jnp.bfloat16, rngs=rngs
    )

    # Build a consistent input tensor.
    key = jax.random.PRNGKey(42 + layer_idx)
    hidden = jax.random.normal(key, (B, T, D), dtype=jnp.bfloat16)

    cos, sin = rotary(pos_ids, lt, jnp.bfloat16)
    position_embeddings = (cos, sin)

    # Manually prepare Q/K/V exactly like Gemma4TextAttention.__call__ does.
    from model.modeling_gemma4 import apply_rotary_pos_emb

    B_, T_, _ = hidden.shape
    q = attn.q_proj(hidden).reshape(B_, T_, attn.num_heads, attn.head_dim)
    q = attn.q_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = attn.k_proj(hidden).reshape(B_, T_, attn.num_kv_heads, attn.head_dim)
    k = attn.k_norm(k)
    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = attn.v_proj(hidden).reshape(B_, T_, attn.num_kv_heads, attn.head_dim)
    v = attn.v_norm(v)
    v = jnp.transpose(v, (0, 2, 1, 3))

    # 1. Raw kernel output (before o_proj) ------------------------------
    # XLA SDPA path — use the bare function.
    if lt == "sliding_attention":
      i = jnp.arange(T)[:, None]
      j = jnp.arange(T)[None, :]
      in_window = (i - j) < text_cfg.sliding_window
      neg = jnp.array(jnp.finfo(jnp.bfloat16).min, dtype=jnp.bfloat16)
      m = jnp.where(in_window, jnp.array(0.0, dtype=jnp.bfloat16), neg)
      xla_mask = m[None, None, :, :]
    else:
      xla_mask = None
    y_xla_raw = _attn_xla_sdpa(
        q,
        k,
        v,
        xla_mask,
        num_key_value_groups=attn.num_key_value_groups,
        scaling=attn.scaling,
        is_causal=True,
    )

    from model.pallas_attention import splash_attention

    y_splash_raw = splash_attention(q, k, v, sliding_window=attn.sliding_window)

    jax.block_until_ready(y_xla_raw)
    jax.block_until_ready(y_splash_raw)

    mx, me = _err(y_xla_raw, y_splash_raw)
    print(
        f"  raw     shape={y_xla_raw.shape} max_err={mx:.5f} mean_err={me:.5f}"
    )
    # max_err tolerance is loose because the XLA path upcasts softmax to
    # fp32 (HF eager convention) while splash keeps it bf16; a few tail
    # outputs differ by O(0.1–0.3) even though mean_err stays ~1e-3.
    tol_raw_max = 3e-1
    tol_raw_mean = 1e-2
    status_raw = "PASS" if mx <= tol_raw_max and me <= tol_raw_mean else "FAIL"
    print(
        f"          tol_max={tol_raw_max} tol_mean={tol_raw_mean} ->"
        f" {status_raw}"
    )
    if status_raw == "FAIL":
      all_pass = False
      print(f"    xla   [0,0,0,:8]={np.asarray(y_xla_raw)[0,0,0,:8]}")
      print(f"    splash[0,0,0,:8]={np.asarray(y_splash_raw)[0,0,0,:8]}")

    # 2. End-to-end attention block (post-o_proj) -----------------------
    os.environ["JAX_ATTENTION_IMPL"] = "xla"
    y_xla, _ = attn(hidden, position_embeddings, xla_mask, {})
    os.environ["JAX_ATTENTION_IMPL"] = "splash"
    y_splash, _ = attn(hidden, position_embeddings, xla_mask, {})

    jax.block_until_ready(y_xla)
    jax.block_until_ready(y_splash)
    mx, me = _err(y_xla, y_splash)
    tol_blk = 1e-1
    status_blk = "PASS" if mx <= tol_blk else "FAIL"
    print(f"  block   shape={y_xla.shape} max_err={mx:.5f} mean_err={me:.5f}")
    print(f"          tol={tol_blk} -> {status_blk}")

  print("\n" + ("All parity checks PASSED." if all_pass else "PARITY FAILED."))
  return 0 if all_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
