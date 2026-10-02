"""Load HF Qwen3 weights into the Flax NNX Qwen3ForCausalLM.

Non-layer params (embed_tokens, norm, lm_head) mirror HF dot-for-dot. The
decoder
layers are stored STACKED — one Qwen3DecoderLayer whose params carry a leading
`[num_layers]` axis (the scan-native layout) — so HF `model.layers.{i}.<rest>`
maps to the stacked param `model.layers.<rest>` at slice `[i]`. The loader
gathers
the N per-layer HF tensors for each leaf and assigns them as one stacked array.
"""

from __future__ import annotations

import jax.numpy as jnp


class _Slice:
  """Read-only Param-like shim exposing `.value` for one layer-slice of a stacked

  param (used by the equivalence test to compare per-HF-layer grads vs the
  [N,...] stack).
  """

  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value


def _nav(node, parts):
  for part in parts:
    node = node[int(part)] if part.isdigit() else getattr(node, part)
  return node


def _is_layer_key(parts):
  return (
      len(parts) >= 4
      and parts[0] == "model"
      and parts[1] == "layers"
      and parts[2].isdigit()
  )


def get_param(model, dotted_key: str):
  """Navigate a dotted HF key to the leaf Param (or a read-only `_Slice` for a

  stacked-layer key `model.layers.{i}.<rest>` → `model.layers.<rest>`[i]).
  """
  parts = dotted_key.split(".")
  if _is_layer_key(parts):
    i = int(parts[2])
    stacked = _nav(getattr(getattr(model, "model"), "layers"), parts[3:])
    return _Slice(stacked.value[i])
  return _nav(model, parts)


def load_hf_state_dict(
    flax_model, hf_state_dict, *, dtype=jnp.float32, strict: bool = True
):
  """Assign every HF tensor into the matching Flax nnx.Param (stacking decoder

  layers along the leading axis). Returns (n_loaded, [], unexpected).
  """
  import numpy as np

  loaded = 0
  unexpected = []
  # leaf-path (e.g. "self_attn.q_proj.weight") -> {layer_idx: tensor}
  layer_groups: dict[str, dict[int, object]] = {}

  # Non-layer params: assign directly. Layer params: bucket by leaf for stacking.
  for key, tensor in hf_state_dict.items():
    parts = key.split(".")
    if _is_layer_key(parts):
      layer_groups.setdefault(".".join(parts[3:]), {})[int(parts[2])] = tensor
      continue
    try:
      param = _nav(flax_model, parts)
    except (AttributeError, IndexError, KeyError, TypeError):
      unexpected.append(key)
      continue
    arr = jnp.asarray(
        np.asarray(tensor.detach().to("cpu").float()), dtype=dtype
    )
    if tuple(param.value.shape) != tuple(arr.shape):
      msg = (
          f"shape mismatch for {key}: flax {tuple(param.value.shape)} vs hf"
          f" {tuple(arr.shape)}"
      )
      if strict:
        raise ValueError(msg)
      print("[weight_loader] " + msg)
      continue
    param.value = arr
    loaded += 1

  # Stacked layers: gather layers 0..N-1 for each leaf, stack, assign once.
  stacked_layer = getattr(getattr(flax_model, "model"), "layers")
  for leaf, by_idx in layer_groups.items():
    n = len(by_idx)
    try:
      param = _nav(stacked_layer, leaf.split("."))
    except (AttributeError, IndexError, KeyError, TypeError):
      unexpected.extend(f"model.layers.{i}.{leaf}" for i in by_idx)
      continue
    stack = np.stack(
        [np.asarray(by_idx[i].detach().to("cpu").float()) for i in range(n)],
        axis=0,
    )
    arr = jnp.asarray(stack, dtype=dtype)
    if tuple(param.value.shape) != tuple(arr.shape):
      msg = (
          f"stacked shape mismatch for layers.*.{leaf}: flax"
          f" {tuple(param.value.shape)} vs hf {tuple(arr.shape)}"
      )
      if strict:
        raise ValueError(msg)
      print("[weight_loader] " + msg)
      continue
    param.value = arr
    loaded += n

  if strict and unexpected:
    raise KeyError(
        f"HF keys with no Flax target: {unexpected[:8]}"
        f"{'...' if len(unexpected) > 8 else ''}"
    )
  return loaded, [], unexpected
