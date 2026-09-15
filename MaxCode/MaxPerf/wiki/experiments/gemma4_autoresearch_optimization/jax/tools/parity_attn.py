#!/usr/bin/env python
# pylint: skip-file
"""Attention-only parity check between HF Gemma4TextAttention (eager) and

our NNX port.
"""

from __future__ import annotations

from pathlib import Path
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _to_jnp(t):
  t = t.detach()
  if t.dtype == torch.bfloat16:
    return jnp.asarray(t.to(torch.float32).numpy()).astype(jnp.bfloat16)
  return jnp.asarray(t.numpy())


def _err(a, b):
  d = np.abs(a.astype(np.float32) - b.astype(np.float32))
  return float(d.max()), float(d.mean())


def main():
  jax.config.update("jax_platform_name", "cpu")
  torch.set_grad_enabled(False)

  from transformers.models.gemma4 import modeling_gemma4 as hfm
  from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

  cfg = Gemma4TextConfig(
      vocab_size=512,
      hidden_size=128,
      intermediate_size=256,
      num_hidden_layers=4,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=32,
      global_head_dim=64,
      sliding_window=16,
      rms_norm_eps=1e-6,
      hidden_size_per_layer_input=16,
      vocab_size_per_layer_input=512,
      num_kv_shared_layers=0,  # simpler: no KV sharing
      final_logit_softcapping=30.0,
      tie_word_embeddings=True,
  )
  cfg._attn_implementation = "eager"

  from model.modeling_gemma4 import Gemma4TextAttention as NNXAttn

  for layer_idx in (0, 3):  # sliding then full
    lt = cfg.layer_types[layer_idx]
    print(f"=== layer {layer_idx} type={lt} ===")
    hf_attn = hfm.Gemma4TextAttention(cfg, layer_idx=layer_idx).to(
        torch.bfloat16
    )
    nnx_attn = NNXAttn(
        cfg, layer_idx, dtype=jnp.bfloat16, rngs=nnx.Rngs(layer_idx)
    )
    # Copy weights.
    nnx_attn.q_proj.weight.value = _to_jnp(hf_attn.q_proj.weight)
    nnx_attn.q_norm.weight.value = _to_jnp(hf_attn.q_norm.weight).astype(
        jnp.float32
    )
    nnx_attn.o_proj.weight.value = _to_jnp(hf_attn.o_proj.weight)
    nnx_attn.k_proj.weight.value = _to_jnp(hf_attn.k_proj.weight)
    nnx_attn.v_proj.weight.value = _to_jnp(hf_attn.v_proj.weight)
    nnx_attn.k_norm.weight.value = _to_jnp(hf_attn.k_norm.weight).astype(
        jnp.float32
    )

    # Build input + rope + mask.
    B, T = 1, 8
    hidden = torch.randn(B, T, cfg.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(T).unsqueeze(0)
    hf_rope = hfm.Gemma4TextRotaryEmbedding(cfg)
    cos_t, sin_t = hf_rope(hidden, pos, layer_type=lt)

    mask_val = torch.finfo(torch.bfloat16).min
    causal = torch.triu(
        torch.full((T, T), mask_val, dtype=torch.bfloat16), diagonal=1
    )
    if lt == "sliding_attention":
      idx = torch.arange(T)
      older = (idx[:, None] - idx[None, :]) >= cfg.sliding_window
      causal = torch.where(older, torch.full_like(causal, mask_val), causal)
    hf_mask = causal[None, None, :, :]

    # Run HF.
    shared_kv: dict = {}
    y_hf, _ = hf_attn(
        hidden_states=hidden.clone(),
        position_embeddings=(cos_t, sin_t),
        attention_mask=hf_mask,
        shared_kv_states=shared_kv,
    )
    y_hf_np = y_hf.detach().to(torch.float32).numpy()

    # Run NNX.
    cos_j = _to_jnp(cos_t)
    sin_j = _to_jnp(sin_t)
    if lt == "sliding_attention":
      i = jnp.arange(T)[:, None]
      j = jnp.arange(T)[None, :]
      in_window = (i - j) < cfg.sliding_window
      m_val = jnp.array(float(np.finfo(np.float32).min), dtype=jnp.bfloat16)
      nnx_mask = jnp.where(in_window, jnp.array(0.0, dtype=jnp.bfloat16), m_val)
      nnx_mask = nnx_mask[None, None, :, :]
    else:
      nnx_mask = None
    shared_kv_n: dict = {}
    y_nnx, _ = nnx_attn(
        _to_jnp(hidden),
        (cos_j, sin_j),
        nnx_mask,
        shared_kv_n,
    )
    y_nnx_np = np.asarray(y_nnx.astype(jnp.float32))
    mx, me = _err(y_hf_np, y_nnx_np)
    print(f"  max={mx:.4f} mean={me:.5f}")
    if mx > 5e-2:
      print("  HF[0,0,:8]=", y_hf_np[0, 0, :8])
      print("  NX[0,0,:8]=", y_nnx_np[0, 0, :8])


if __name__ == "__main__":
  main()
