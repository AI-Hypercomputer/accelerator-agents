# pylint: skip-file
"""Parity check: levanter fused CE kernel vs the JAX-stack reference CE path.

Reference path mirrors ``train.py:forward_loss`` when ``JAX_CE_DTYPE=bf16``:
materialize ``[B*S, V]`` logits, apply Gemma 4 softcap, bf16 log_softmax, NLL
with ``ignore_index=-100``, ``sum/mask.sum()`` reduction.

Fused path calls ``fused_cross_entropy_loss_and_logsumexp_penalty`` with
``logit_soft_cap=30.0`` and the per-example ``weight=mask`` to zero out
ignored labels, then divides by ``mask.sum()`` to match the ignore-aware mean.

Tol: 5e-2 absolute in bf16 (kernel+softcap chain accumulates more drift than
a plain log_softmax). Prints max/mean abs diff and PASS/FAIL.

Usage::

    /home/alekseyv_google_com/miniconda3/envs/gemma4_py313/bin/python \
        -m tools.parity_levanter_ce

from inside ``wiki/experiments/gemma4_autoresearch_optimization/jax/``.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Keep autotune-on-miss disabled so the bootstrap stays offline.
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

HERE = Path(__file__).resolve().parent
JAX_ROOT = HERE.parent
if str(JAX_ROOT) not in sys.path:
  sys.path.insert(0, str(JAX_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from model.kernels.fused_ce import load_kernel  # noqa: E402

IGNORE_INDEX = -100
SOFTCAP = 30.0
TOL = 5e-2

# Explicit block sizes: Gemma 4 E4B (V=262144, H=2560) lands in no TPU-tuned
# bucket, so the default (1024, 512, 1024) runs the VMEM budget over at
# our shapes. Smaller v-tiles fit in 32 MiB VMEM and keep the kernel
# streaming. Autotune-on-miss is disabled by the caller to keep offline.
DEFAULT_BLOCK_V = 512
DEFAULT_BLOCK_H = 256


def reference_ce(
    hidden: jax.Array, weight: jax.Array, labels: jax.Array
) -> jax.Array:
  """Reference path: materialize logits, softcap, bf16 log_softmax, NLL, ignore-aware mean.

  Matches the JAX-stack ``forward_loss`` at ``JAX_CE_DTYPE=bf16``.
  """
  # logits = hidden @ weight.T (weight is [V, H] per HF embed layout).
  logits = hidden @ weight.T
  sc_f = jnp.float32(SOFTCAP)
  logits = sc_f * jnp.tanh(logits.astype(jnp.float32) / sc_f)
  logits = logits.astype(hidden.dtype)

  vocab = logits.shape[-1]
  flat_logits = logits.reshape(-1, vocab)  # already in hidden dtype
  flat_labels = labels.reshape(-1)
  mask = (flat_labels != IGNORE_INDEX).astype(flat_logits.dtype)
  log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
  safe_labels = jnp.where(
      flat_labels == IGNORE_INDEX, jnp.zeros_like(flat_labels), flat_labels
  )
  picked = jnp.take_along_axis(
      log_probs, safe_labels[:, None], axis=-1
  ).squeeze(-1)
  loss = -(
      picked.astype(jnp.float32) * mask.astype(jnp.float32)
  ).sum() / jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
  return loss.astype(jnp.float32)


def levanter_ce(
    hidden: jax.Array, weight: jax.Array, labels: jax.Array, fn
) -> jax.Array:
  """Fused path: no [B*S, V] materialization, softcap inline in the kernel.

  ``weight`` is the HF-orientation embedding table ``[V, H]``; levanter's
  kernel expects ``[H, V]`` so we transpose once.
  """
  from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes

  B, S, H = hidden.shape
  V = weight.shape[0]
  flat_hidden = hidden.reshape(B * S, H)
  flat_labels = labels.reshape(B * S)
  mask = (flat_labels != IGNORE_INDEX).astype(jnp.float32)
  # Replace -100 → 0 so the kernel's one_hot is well-defined; ``weight=mask``
  # zeros out the contribution of those rows anyway.
  safe_labels = jnp.where(
      flat_labels == IGNORE_INDEX, jnp.zeros_like(flat_labels), flat_labels
  )
  w_hv = weight.T  # [V, H] -> [H, V]; the transpose is free under XLA layout.
  b_block = min(B * S, 128)
  block_sizes = BlockSizes(
      b_block_size=b_block,
      h_block_size=min(H, DEFAULT_BLOCK_H),
      v_block_size=DEFAULT_BLOCK_V,
  )
  loss_sum = fn(
      flat_hidden,
      safe_labels,
      w_hv,
      reduction="sum",
      weight=mask,
      logit_soft_cap=SOFTCAP,
      implementation="pallas_tpu",
      dtype=jnp.bfloat16,
      block_sizes=block_sizes,
  )
  return (loss_sum / jnp.maximum(mask.sum(), 1.0)).astype(jnp.float32)


def main() -> int:
  B, S, H, V = 1, 128, 2560, 262144
  rng = np.random.default_rng(0)
  hidden_np = rng.standard_normal((B, S, H), dtype=np.float32) * 0.02
  # Weight standardized to 1/sqrt(H) for numerical stability (Gemma 4's
  # tie_word_embeddings means this is the embed table).
  weight_np = rng.standard_normal((V, H), dtype=np.float32) * (1.0 / np.sqrt(H))
  labels_np = rng.integers(0, V, size=(B, S)).astype(np.int32)
  # Force ~10 % ignore-index to exercise the mask path.
  ignore_mask = rng.random((B, S)) < 0.1
  labels_np = np.where(ignore_mask, IGNORE_INDEX, labels_np).astype(np.int32)

  hidden = jnp.asarray(hidden_np, dtype=jnp.bfloat16)
  weight = jnp.asarray(weight_np, dtype=jnp.bfloat16)
  labels = jnp.asarray(labels_np, dtype=jnp.int32)

  fn = load_kernel()

  print(
      f"[parity] shapes: hidden={hidden.shape} weight={weight.shape}"
      f" labels={labels.shape}"
  )
  print(f"[parity] ignore-index fraction: {float(ignore_mask.mean()):.3f}")

  loss_ref = jax.jit(reference_ce)(hidden, weight, labels)
  jax.block_until_ready(loss_ref)
  loss_lev = jax.jit(lambda h, w, l: levanter_ce(h, w, l, fn))(
      hidden, weight, labels
  )
  jax.block_until_ready(loss_lev)

  ref_f = float(loss_ref)
  lev_f = float(loss_lev)
  diff = abs(ref_f - lev_f)
  print(f"[parity] reference loss = {ref_f:.6f}")
  print(f"[parity] levanter  loss = {lev_f:.6f}")
  print(f"[parity] |diff|         = {diff:.6f}  (tol = {TOL})")

  if diff <= TOL:
    print("[parity] PASS")
    return 0
  print("[parity] FAIL")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
