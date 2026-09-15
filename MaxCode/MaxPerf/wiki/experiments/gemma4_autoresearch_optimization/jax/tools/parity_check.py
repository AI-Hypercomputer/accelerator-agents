#!/usr/bin/env python
# pylint: skip-file
"""Forward numerical parity: native-JAX port vs HF PyTorch reference.

Loads Gemma 4 E4B in both stacks on CPU (bf16), feeds an all-zero input of
shape (1, 128), and compares the resulting logits elementwise. Reports max
abs error and agreement of argmax.

Note: this is a CPU-only reference check — the HF PyTorch path is the
ground truth. The JAX port should match within bf16 tolerance (abs_err <=
1e-2 is the bar given to us; realistically bf16 softmax/logits are noisy
at ~5e-3 on a 42-layer model with softcap).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: Optional[list] = None) -> int:
  model_id = os.environ.get("PARITY_MODEL_ID", "google/gemma-4-E4B")
  seq_len = int(os.environ.get("PARITY_SEQ", "128"))
  print(f"[parity] model={model_id} seq_len={seq_len}")
  # Force the JAX side onto CPU so HF-torch can live on CPU too.
  jax.config.update("jax_platform_name", "cpu")
  torch.set_grad_enabled(False)

  # HF reference --------------------------------------------------------
  from transformers import AutoConfig, Gemma4ForConditionalGeneration
  from transformers.models.gemma4 import modeling_gemma4 as hf_mod

  cfg = AutoConfig.from_pretrained(model_id)
  text_cfg = cfg.text_config
  print(f"[parity] HF loading {model_id} on CPU (bf16)")
  hf_model = Gemma4ForConditionalGeneration.from_pretrained(
      model_id,
      dtype=torch.bfloat16,
      attn_implementation="eager",
  )
  hf_model.eval()

  input_ids = torch.zeros((1, seq_len), dtype=torch.long)
  # Direct text path: language_model + lm_head, then softcap.
  lm_out = hf_model.model.language_model(
      input_ids=input_ids,
      use_cache=False,
      return_dict=True,
  )
  hidden_hf = lm_out.last_hidden_state
  logits_hf = hf_model.lm_head(hidden_hf)
  sc = text_cfg.final_logit_softcapping
  if sc is not None and sc > 0:
    logits_hf = logits_hf / sc
    logits_hf = torch.tanh(logits_hf)
    logits_hf = logits_hf * sc
  logits_hf_np = logits_hf.detach().to(torch.float32).numpy()
  print(
      f"[parity] HF logits shape={logits_hf_np.shape}"
      f" dtype={logits_hf_np.dtype}"
  )

  # Free HF model memory before loading the JAX side (both weights live
  # simultaneously otherwise — ~15 GB RAM).
  del hf_model, lm_out, hidden_hf, logits_hf
  import gc

  gc.collect()

  # JAX port ------------------------------------------------------------
  from model.modeling_gemma4 import Gemma4ForCausalLM
  from model.weight_loader import load_hf_weights

  print(f"[parity] NNX building / loading {model_id} on CPU (bf16)")
  model = Gemma4ForCausalLM(text_cfg, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  load_hf_weights(model, model_id, dtype=jnp.bfloat16, verbose=False)
  ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
  logits_jx = np.asarray(model(ids).astype(jnp.float32))
  print(f"[parity] NNX logits shape={logits_jx.shape} dtype={logits_jx.dtype}")

  if logits_hf_np.shape != logits_jx.shape:
    raise SystemExit(
        f"SHAPE MISMATCH: HF {logits_hf_np.shape} vs NNX {logits_jx.shape}"
    )
  abs_err = np.abs(logits_hf_np - logits_jx)
  max_err = float(abs_err.max())
  mean_err = float(abs_err.mean())
  print(f"[parity] max abs err: {max_err:.5f}  mean abs err: {mean_err:.6f}")
  # argmax agreement on the last position
  hf_am = logits_hf_np[0, -1].argmax()
  jx_am = logits_jx[0, -1].argmax()
  print(
      f"[parity] argmax(last pos) HF={hf_am} NNX={jx_am} "
      f" match={hf_am == jx_am}"
  )
  # Also print the logit magnitudes (sanity: softcap 30 bounds them).
  print(
      f"[parity] HF  min/max: {logits_hf_np.min():.3f} /"
      f" {logits_hf_np.max():.3f}"
  )
  print(f"[parity] NNX min/max: {logits_jx.min():.3f} / {logits_jx.max():.3f}")

  tol = float(os.environ.get("PARITY_TOL", "1e-2"))
  ok = max_err <= tol
  print(f"[parity] tolerance={tol} -> {'PASS' if ok else 'FAIL'}")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
