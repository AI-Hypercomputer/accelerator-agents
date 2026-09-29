"""CPU equivalence test — native Flax NNX Qwen3 vs reference HF PyTorch.

Validates that the from-scratch JAX/Flax Qwen3 port produces the SAME forward
logits and backward gradients as HF `Qwen3ForCausalLM`, given identical weights
(the HF state_dict is loaded into the Flax model). Runs entirely on CPU.

  FORWARD  — Flax `model(input_ids)` logits vs HF `.logits`.
  BACKWARD — Flax grads via `nnx.split` + `jax.value_and_grad` vs HF
             `loss.backward()` torch-autograd grads, compared per HF param name.

HF ref uses eager attention so its math matches the Flax einsum SDPA exactly.

Run:
    JAX_PLATFORMS=cpu python test_equivalence.py
Exit 0 = all within tolerance.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM as HFQwen3ForCausalLM

# Local model package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Qwen3ForCausalLM as FlaxQwen3ForCausalLM, load_hf_state_dict, get_param

FWD_ATOL, FWD_RTOL = 2e-4, 2e-4
BWD_ATOL, BWD_RTOL = 2e-3, 2e-3


def tiny_config() -> Qwen3Config:
  return Qwen3Config(
      vocab_size=256,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      max_position_embeddings=64,
      rope_theta=1_000_000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
      attention_dropout=0.0,
      attn_implementation="eager",
  )


def _ce_jax(logits, labels):
  logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  nll = -jnp.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]
  return nll.mean()


def _summarize(name, a, b, atol, rtol) -> bool:
  a = np.asarray(a, dtype=np.float32)
  b = np.asarray(b, dtype=np.float32)
  max_abs = float(np.abs(a - b).max())
  denom = float(np.abs(b).max()) or 1.0
  ok = np.allclose(a, b, atol=atol, rtol=rtol)
  print(
      f"  {'OK ' if ok else 'XX '}{name:<40} max|Δ|={max_abs:.3e}  max"
      f" relΔ={max_abs/denom:.3e}"
  )
  return ok


def main() -> int:
  torch.manual_seed(0)
  cfg = tiny_config()

  # ---- Reference: HF PyTorch (eager attn) on CPU, float32 ----
  ref = HFQwen3ForCausalLM(cfg).to(torch.float32).eval()

  B, L = 2, 16
  g = torch.Generator().manual_seed(1)
  input_ids = torch.randint(0, cfg.vocab_size, (B, L), generator=g)
  labels = torch.randint(0, cfg.vocab_size, (B, L), generator=g)

  ref.zero_grad(set_to_none=True)
  logits_ref = ref(input_ids).logits
  loss_ref = F.cross_entropy(
      logits_ref.reshape(-1, cfg.vocab_size), labels.reshape(-1)
  )
  loss_ref.backward()
  grads_ref = {
      n: p.grad.detach().float().cpu().numpy()
      for n, p in ref.named_parameters()
      if p.grad is not None
  }
  print(f"[ref ] loss={loss_ref.item():.6f}  params_with_grad={len(grads_ref)}")

  # ---- Flax NNX: identical weights (load HF state_dict), float32 ----
  model = FlaxQwen3ForCausalLM(
      cfg,
      weights_dtype=jnp.float32,
      compute_dtype=jnp.float32,
      rngs=nnx.Rngs(0),
  )
  n_loaded, _, unexpected = load_hf_state_dict(
      model, ref.state_dict(), dtype=jnp.float32
  )
  print(f"[flax] loaded {n_loaded} HF tensors; unexpected={unexpected}")

  input_j = jnp.asarray(input_ids.numpy(), dtype=jnp.int32)
  labels_j = jnp.asarray(labels.numpy(), dtype=jnp.int32)

  logits_flax = model(input_j)

  # Gradients via the functional nnx split (mirrors how a trainer takes grads).
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def loss_fn(params):
    m = nnx.merge(graphdef, params, rest)
    logits = m(input_j).reshape(-1, cfg.vocab_size)
    return _ce_jax(logits, labels_j.reshape(-1))

  loss_flax, grads = jax.value_and_grad(loss_fn)(params)
  grad_model = nnx.merge(graphdef, grads, rest)
  print(f"[flax] loss={float(loss_flax):.6f}")

  # ---- Compare ----
  print("\nFORWARD (logits):")
  ok = _summarize(
      "logits", logits_flax, logits_ref.detach().numpy(), FWD_ATOL, FWD_RTOL
  )
  ok &= _summarize(
      "loss",
      np.array(float(loss_flax)),
      np.array(loss_ref.item()),
      FWD_ATOL,
      FWD_RTOL,
  )

  print("\nBACKWARD (gradients, per HF parameter):")
  for name in grads_ref:
    g_flax = get_param(grad_model, name).value
    ok &= _summarize(name, g_flax, grads_ref[name], BWD_ATOL, BWD_RTOL)

  print(
      "\n"
      + ("==== EQUIVALENCE PASS ====" if ok else "==== EQUIVALENCE FAIL ====")
  )
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
