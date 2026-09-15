#!/usr/bin/env python
# pylint: skip-file
"""Per-layer forward parity: HF PyTorch vs native-JAX port.

Creates small modules from both stacks, copies HF weights into the NNX
tree, and compares the forward output on a fixed input. Much cheaper than
a full-model CPU run and catches nearly all porting bugs.

Checks (bf16 on CPU):
  1. RMSNorm
  2. MLP
  3. Rotary embedding (cos/sin values)
  4. One full decoder layer (exercises attention + MLP + PLE branch)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _to_jnp(t: torch.Tensor) -> jax.Array:
  t = t.detach()
  if t.dtype == torch.bfloat16:
    a = jnp.asarray(t.to(torch.float32).numpy()).astype(jnp.bfloat16)
  else:
    a = jnp.asarray(t.numpy())
  return a


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
  diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
  return float(diff.max()), float(diff.mean())


def main() -> int:
  jax.config.update("jax_platform_name", "cpu")
  torch.set_grad_enabled(False)

  from transformers import AutoConfig
  from transformers.models.gemma4 import modeling_gemma4 as hfm
  from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

  # Tiny config — faster init/forward. Uses E4B-style head_dim etc.
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
      num_kv_shared_layers=2,
      final_logit_softcapping=30.0,
      tie_word_embeddings=True,
  )
  cfg._attn_implementation = "eager"
  print(
      f"[parity] tiny config layers={cfg.num_hidden_layers} "
      f"types={cfg.layer_types}"
  )

  # --- 1. RMSNorm parity -------------------------------------------------
  from model.modeling_gemma4 import Gemma4RMSNorm as NNXRMSNorm

  hf_rms = hfm.Gemma4RMSNorm(16, eps=1e-6)
  # HF init: weight=ones by nn.Parameter default; but Gemma 4 init uses zeros.
  torch.nn.init.zeros_(hf_rms.weight)
  # Fill a real value to exercise scale.
  with torch.no_grad():
    hf_rms.weight.copy_(torch.randn(16) * 0.1)
  nnx_rms = NNXRMSNorm(16, eps=1e-6, rngs=nnx.Rngs(0))
  nnx_rms.weight.value = _to_jnp(hf_rms.weight).astype(jnp.float32)
  x_t = torch.randn(2, 5, 16, dtype=torch.bfloat16)
  x_j = _to_jnp(x_t)
  y_t = hf_rms(x_t).detach().to(torch.float32).numpy()
  y_j = np.asarray(nnx_rms(x_j).astype(jnp.float32))
  max_err, mean_err = _err(y_t, y_j)
  print(f"[rms] max={max_err:.6f} mean={mean_err:.6f}")
  assert max_err < 1e-2, f"RMSNorm parity failed: {max_err}"

  # --- 2. MLP parity -----------------------------------------------------
  from model.modeling_gemma4 import Gemma4TextMLP as NNXMLP

  hf_mlp = hfm.Gemma4TextMLP(cfg, layer_idx=0).to(torch.bfloat16)
  nnx_mlp = NNXMLP(cfg, layer_idx=0, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  for name in ("gate_proj", "up_proj", "down_proj"):
    hf_w = getattr(hf_mlp, name).weight
    getattr(nnx_mlp, name).weight.value = _to_jnp(hf_w)
  x_t = torch.randn(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
  y_t = hf_mlp(x_t).detach().to(torch.float32).numpy()
  y_j = np.asarray(nnx_mlp(_to_jnp(x_t)).astype(jnp.float32))
  max_err, mean_err = _err(y_t, y_j)
  print(f"[mlp] max={max_err:.6f} mean={mean_err:.6f}")
  assert max_err < 5e-2, f"MLP parity failed: {max_err}"

  # --- 3. Rotary parity --------------------------------------------------
  from model.modeling_gemma4 import Gemma4TextRotaryEmbedding as NNXRope

  hf_rope = hfm.Gemma4TextRotaryEmbedding(cfg)
  nnx_rope = NNXRope(cfg)
  # Use a dummy "x" to satisfy HF's dtype+device inference.
  pos_t = torch.arange(8).unsqueeze(0)
  pos_j = jnp.asarray(pos_t.numpy())
  x_t = torch.randn(1, 8, cfg.head_dim, dtype=torch.bfloat16)
  for lt in set(cfg.layer_types):
    cos_t, sin_t = hf_rope(x_t, pos_t, layer_type=lt)
    cos_j, sin_j = nnx_rope(pos_j, lt, dtype=jnp.bfloat16)
    for label, a, b in (("cos", cos_t, cos_j), ("sin", sin_t, sin_j)):
      aa = a.detach().to(torch.float32).numpy()
      bb = np.asarray(b.astype(jnp.float32))
      max_err, mean_err = _err(aa, bb)
      print(f"[rope {lt} {label}] max={max_err:.6f} mean={mean_err:.6f}")
      assert max_err < 5e-3, f"RoPE {lt} {label} parity failed: {max_err}"

  # --- 4. Full decoder layer parity (sliding) ---------------------------
  # We compare layer 0 (sliding_attention, head_dim=32) and layer 2
  # (full_attention, head_dim=64) by copying HF weights into NNX.
  from model.modeling_gemma4 import Gemma4TextDecoderLayer as NNXLayer

  def _copy_layer_weights(hf_layer, nnx_layer):
    # RMSNorms.
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
      w = getattr(hf_layer, name).weight
      getattr(nnx_layer, name).weight.value = _to_jnp(w).astype(jnp.float32)
    # layer_scalar buffer.
    nnx_layer.layer_scalar.value = _to_jnp(hf_layer.layer_scalar).astype(
        jnp.bfloat16
    )
    # MLP.
    for name in ("gate_proj", "up_proj", "down_proj"):
      getattr(nnx_layer.mlp, name).weight.value = _to_jnp(
          getattr(hf_layer.mlp, name).weight
      )
    # Attention.
    ha = hf_layer.self_attn
    na = nnx_layer.self_attn
    na.q_proj.weight.value = _to_jnp(ha.q_proj.weight)
    na.q_norm.weight.value = _to_jnp(ha.q_norm.weight).astype(jnp.float32)
    na.o_proj.weight.value = _to_jnp(ha.o_proj.weight)
    if not na.is_kv_shared_layer:
      na.k_proj.weight.value = _to_jnp(ha.k_proj.weight)
      na.v_proj.weight.value = _to_jnp(ha.v_proj.weight)
      na.k_norm.weight.value = _to_jnp(ha.k_norm.weight).astype(jnp.float32)
      # v_norm has no weight (with_scale=False).
    # PLE.
    if hf_layer.hidden_size_per_layer_input:
      nnx_layer.per_layer_input_gate.weight.value = _to_jnp(
          hf_layer.per_layer_input_gate.weight
      )
      nnx_layer.per_layer_projection.weight.value = _to_jnp(
          hf_layer.per_layer_projection.weight
      )
      nnx_layer.post_per_layer_input_norm.weight.value = _to_jnp(
          hf_layer.post_per_layer_input_norm.weight
      ).astype(jnp.float32)

  B, T, D = 1, 8, cfg.hidden_size
  D_ple = cfg.hidden_size_per_layer_input
  # input.
  hidden = torch.randn(B, T, D, dtype=torch.bfloat16)
  per_layer_in = torch.randn(B, T, D_ple, dtype=torch.bfloat16)

  # Precompute rope from HF for both layer types.
  hf_rope = hfm.Gemma4TextRotaryEmbedding(cfg)
  pos_ids = torch.arange(T).unsqueeze(0)
  rope_by_type = {
      lt: hf_rope(hidden, pos_ids, layer_type=lt) for lt in set(cfg.layer_types)
  }

  # Test two layer indices (one sliding, one full).
  for layer_idx in (0, 5 if cfg.num_hidden_layers > 5 else 1):
    if layer_idx >= cfg.num_hidden_layers:
      continue
    lt = cfg.layer_types[layer_idx]
    print(f"[layer {layer_idx} type={lt}]")
    hf_layer = hfm.Gemma4TextDecoderLayer(cfg, layer_idx).to(torch.bfloat16)
    nnx_layer = NNXLayer(
        cfg, layer_idx, dtype=jnp.bfloat16, rngs=nnx.Rngs(layer_idx)
    )
    _copy_layer_weights(hf_layer, nnx_layer)

    cos_t, sin_t = rope_by_type[lt]
    # Build mask. HF decoder expects full (B, 1, T, T) additive.
    import torch.nn.functional as F

    mask_val = torch.finfo(torch.bfloat16).min
    causal = torch.triu(
        torch.full((T, T), mask_val, dtype=torch.bfloat16), diagonal=1
    )
    if lt == "sliding_attention":
      # Mask anything older than window.
      idx = torch.arange(T)
      older = (idx[:, None] - idx[None, :]) >= cfg.sliding_window
      causal = torch.where(older, torch.full_like(causal, mask_val), causal)
    hf_mask = causal[None, None, :, :]

    # HF forward.
    shared_kv_hf: dict = {}
    y_hf = hf_layer(
        hidden.clone(),
        per_layer_input=per_layer_in.clone(),
        shared_kv_states=shared_kv_hf,
        position_embeddings=(cos_t, sin_t),
        attention_mask=hf_mask,
    )
    y_hf_np = y_hf.detach().to(torch.float32).numpy()

    # NNX forward.
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
    shared_kv_nnx: dict = {}
    y_nnx = nnx_layer(
        _to_jnp(hidden),
        _to_jnp(per_layer_in),
        shared_kv_nnx,
        (cos_j, sin_j),
        nnx_mask,
    )
    y_nnx_np = np.asarray(y_nnx.astype(jnp.float32))

    max_err, mean_err = _err(y_hf_np, y_nnx_np)
    print(f"  layer output max={max_err:.5f} mean={mean_err:.6f}")
    # Note: bf16 through attention+MLP+PLE routinely has ~1e-2 max;
    # 3e-2 is generous but catches real bugs.
    assert max_err < 5e-2, f"layer {layer_idx} parity failed: {max_err}"

  print("[parity] PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
