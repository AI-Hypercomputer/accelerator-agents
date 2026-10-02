"""Load HF Qwen3 weights into the Flax NNX Qwen3ForCausalLM.

The Flax module hierarchy mirrors HF dot-for-dot (ForCausalLM.model.* and
ForCausalLM.lm_head.*), so an HF state_dict key navigates straight to the
matching `nnx.Param` — `model.layers.5.self_attn.q_proj.weight` →
`fcm.model.layers[5].self_attn.q_proj.weight`. The same navigation works on a
grads-as-a-model (from `nnx.merge(graphdef, grads, rest)`), which the
equivalence test uses to compare gradients by HF name.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_param(model, dotted_key: str):
  """Navigate a dotted HF key to the leaf nnx.Param (integer parts index into

  the `layers` list).
  """
  node = model
  for part in dotted_key.split("."):
    node = node[int(part)] if part.isdigit() else getattr(node, part)
  return node


def load_hf_state_dict(
    flax_model, hf_state_dict, *, dtype=jnp.float32, strict: bool = True
):
  """Assign every HF tensor into the matching Flax nnx.Param.

  `hf_state_dict` maps name -> torch.Tensor (CPU). Returns (n_loaded, missing,
  unexpected). With strict=True, raises on any shape mismatch or missing key.
  """
  import numpy as np

  loaded = 0
  unexpected = []
  for key, tensor in hf_state_dict.items():
    try:
      param = get_param(flax_model, key)
    except (AttributeError, IndexError, KeyError, TypeError):
      unexpected.append(key)
      continue
    arr = jnp.asarray(
        np.asarray(tensor.detach().to("cpu").float()), dtype=dtype
    )
    if tuple(param.value.shape) != tuple(arr.shape):
      msg = (
          f"shape mismatch for {key}: flax {tuple(param.value.shape)} "
          f"vs hf {tuple(arr.shape)}"
      )
      if strict:
        raise ValueError(msg)
      print("[weight_loader] " + msg)
      continue
    param.value = arr
    loaded += 1

  if strict and unexpected:
    raise KeyError(
        f"HF keys with no Flax target: {unexpected[:8]}"
        f"{'...' if len(unexpected) > 8 else ''}"
    )
  return loaded, [], unexpected
