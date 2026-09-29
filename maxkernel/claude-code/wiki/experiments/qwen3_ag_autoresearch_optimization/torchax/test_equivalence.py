"""CPU equivalence test — torchax (JAX) path vs reference HF PyTorch.

Validates the autoresearch invariant: the torchax execution path the trainer
actually uses must produce the SAME logits and gradients as plain HF PyTorch,
given identical weights. Runs entirely on CPU — no TPU needed.

What it checks, on a TINY Qwen3 config (fast):
  1. FORWARD  — `JittableModule.functional_call('forward', ...)` logits
                vs `Qwen3ForCausalLM(input_ids).logits` (plain torch).
  2. BACKWARD — gradients from `torchax.interop.jax_value_and_grad` (the exact
                mechanism `torchax.train.make_train_step` uses) vs
                `loss.backward()` torch-autograd grads.

Both models are seeded identically AND the torchax model is loaded from the
reference's `state_dict`, so any difference is purely the execution path
(JAX-lowered ops vs torch CPU ops), i.e. float reduction-order noise.

Run:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu \
        python test_equivalence.py
Exit code 0 = all checks within tolerance; 1 = a check exceeded tolerance.
"""

from __future__ import annotations

import os

# Force JAX onto CPU and keep float32 (tight tolerance) before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import sys

import torch
import torch.nn.functional as F

import torchax
from torchax.interop import JittableModule
from transformers import Qwen3Config, Qwen3ForCausalLM

# Tolerances — float32, JAX vs torch differ only by reduction order.
FWD_ATOL, FWD_RTOL = 2e-4, 2e-4
BWD_ATOL, BWD_RTOL = 2e-3, 2e-3


def tiny_config() -> Qwen3Config:
  """Small Qwen3 that exercises every structural feature (GQA, QK-norm,

  SwiGLU, RMSNorm, untied lm_head) but runs in well under a second on CPU.
  """
  return Qwen3Config(
      vocab_size=256,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,  # GQA 2:1
      head_dim=16,
      max_position_embeddings=64,
      rope_theta=1_000_000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
      attention_dropout=0.0,
  )


def _summarize(name: str, a: torch.Tensor, b: torch.Tensor, atol, rtol) -> bool:
  a = a.detach().to(torch.float32)
  b = b.detach().to(torch.float32)
  max_abs = (a - b).abs().max().item()
  denom = b.abs().max().item() or 1.0
  max_rel = max_abs / denom
  ok = torch.allclose(a, b, atol=atol, rtol=rtol)
  flag = "OK " if ok else "XX "
  print(f"  {flag}{name:<34} max|Δ|={max_abs:.3e}  max relΔ={max_rel:.3e}")
  return ok


def main() -> int:
  torch.manual_seed(0)
  cfg = tiny_config()
  dtype = torch.float32

  # ---- Reference: plain HF PyTorch on CPU (eval → no dropout) ----
  ref = Qwen3ForCausalLM(cfg).to(dtype).eval()

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
      n: p.grad.detach().clone()
      for n, p in ref.named_parameters()
      if p.grad is not None
  }
  print(f"[ref] loss={loss_ref.item():.6f}  params_with_grad={len(grads_ref)}")

  # ---- torchax path: identical weights, JAX execution ----
  env = torchax.default_env()
  with env:
    tx = Qwen3ForCausalLM(cfg).to(dtype).eval()
    # Identical weights: load the reference state_dict, then move every
    # tensor onto the 'jax' device so ops lower to JAX (CPU here).
    sd_jax = {k: v.to(dtype).to("jax") for k, v in ref.state_dict().items()}
    tx.load_state_dict(sd_jax, assign=True)

    # Non-persistent buffers (Qwen3 rotary `inv_freq`/`original_inv_freq`)
    # aren't in state_dict — move them onto the 'jax' device too, else
    # JittableModule hands plain torch tensors to jax_view. (inv_freq is
    # deterministic from rope_theta, so ref and tx values are identical.)
    for m in tx.modules():
      for bname, b in list(m.named_buffers(recurse=False)):
        if b is not None:
          setattr(m, bname, b.to("jax"))

    jmod = JittableModule(tx)
    input_j = input_ids.to("jax")
    labels_j = labels.to("jax")

    def loss_fn(weights, buffers, args, lbl):
      # functional_call takes the forward positionals via *args — `args`
      # here is the single input_ids tensor (matches train.py's model_fn).
      out = jmod.functional_call("forward", weights, buffers, args)
      logits = out.logits if hasattr(out, "logits") else out
      return F.cross_entropy(
          logits.reshape(-1, cfg.vocab_size), lbl.reshape(-1)
      )

    # Forward logits (same call shape the trainer's model_fn uses).
    out_tx = jmod.functional_call("forward", jmod.params, jmod.buffers, input_j)
    logits_tx = out_tx.logits if hasattr(out_tx, "logits") else out_tx

    # Backward: gradients via the trainer's exact mechanism.
    grad_fn = torchax.interop.jax_value_and_grad(loss_fn)
    loss_tx, grads_tx = grad_fn(jmod.params, jmod.buffers, input_j, labels_j)

    # Move results back to CPU torch for comparison.
    logits_tx_cpu = logits_tx.to("cpu")
    loss_tx_cpu = loss_tx.to("cpu")
    grads_tx_cpu = {k: v.to("cpu") for k, v in grads_tx.items()}

  print(f"[tx ] loss={loss_tx_cpu.item():.6f}")

  # ---- Compare ----
  print("\nFORWARD (logits):")
  ok_fwd = _summarize("logits", logits_tx_cpu, logits_ref, FWD_ATOL, FWD_RTOL)
  ok_loss = _summarize(
      "loss", loss_tx_cpu.reshape(()), loss_ref.reshape(()), FWD_ATOL, FWD_RTOL
  )

  print("\nBACKWARD (gradients, per parameter):")
  ok_bwd = True
  missing = []
  for name in grads_ref:
    if name not in grads_tx_cpu:
      missing.append(name)
      continue
    ok_bwd &= _summarize(
        name, grads_tx_cpu[name], grads_ref[name], BWD_ATOL, BWD_RTOL
    )
  if missing:
    print(f"  XX missing grads in torchax path: {missing}")
    ok_bwd = False
  extra = [k for k in grads_tx_cpu if k not in grads_ref]
  if extra:
    print(f"  (note: extra torchax grads not in ref: {extra})")

  all_ok = ok_fwd and ok_loss and ok_bwd
  print(
      "\n"
      + (
          "==== EQUIVALENCE PASS ===="
          if all_ok
          else "==== EQUIVALENCE FAIL ===="
      )
  )
  return 0 if all_ok else 1


if __name__ == "__main__":
  sys.exit(main())
