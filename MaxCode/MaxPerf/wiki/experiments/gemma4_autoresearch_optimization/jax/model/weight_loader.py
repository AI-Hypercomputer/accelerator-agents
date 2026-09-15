# pylint: skip-file
"""HuggingFace safetensors -> Flax NNX weight loader for Gemma 4.

Loads `google/gemma-4-E4B` checkpoint files and copies tensor values into
the NNX parameter tree of a `Gemma4ForCausalLM` instance (built with
random init). Drops all non-text-tower keys (audio, vision, multimodal
connectors, expert-MoE, clipped-linear buffers, etc.). Also drops the
``k_proj`` / ``v_proj`` / ``k_norm`` / ``v_norm`` weights for shared-KV
layers (HF runtime ignores them).

Assumes the HF param-name convention:
    model.language_model.<sub>                -> model.<sub>
    lm_head.weight                            -> (tied, ignored if tie=True)

Usage:
    from transformers import Gemma4TextConfig
    from flax import nnx
    from model.modeling_gemma4 import Gemma4ForCausalLM
    from model.weight_loader import load_hf_weights

    cfg = Gemma4TextConfig.from_pretrained("google/gemma-4-E4B")
    model = Gemma4ForCausalLM(cfg, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    load_hf_weights(model, "/path/to/model.safetensors")
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterator

from flax import nnx
import jax
import jax.numpy as jnp

_LANGUAGE_PREFIX = "model.language_model."


def _iter_safetensors_paths(model_id_or_path: str) -> list[str]:
  """Resolve a HF model id or path to a list of safetensors shard paths."""
  if os.path.isdir(model_id_or_path):
    # explicit directory
    paths = sorted(glob.glob(os.path.join(model_id_or_path, "*.safetensors")))
    if paths:
      return paths
  if os.path.isfile(model_id_or_path):
    return [model_id_or_path]
  # Resolve via HF cache.
  from huggingface_hub import snapshot_download

  snapshot = snapshot_download(
      repo_id=model_id_or_path,
      allow_patterns=["*.safetensors", "*.json"],
  )
  paths = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
  if not paths:
    raise FileNotFoundError(
        f"No safetensors found for {model_id_or_path!r} at {snapshot!r}"
    )
  return paths


def _iter_hf_tensors(paths: list[str]) -> Iterator[tuple[str, "torch.Tensor"]]:
  """Yield (name, tensor) pairs across all safetensors shards."""
  from safetensors import safe_open

  for p in paths:
    with safe_open(p, framework="pt") as f:
      for k in f.keys():
        yield k, f.get_tensor(k)


def _strip_prefix(name: str) -> str:
  """model.language_model.X -> model.X  (train loop's NNX tree root is

  a Gemma4ForCausalLM whose `.model` is the text tower).
  """
  if name.startswith(_LANGUAGE_PREFIX):
    return "model." + name[len(_LANGUAGE_PREFIX) :]
  return name


# Param names we never want to copy into the text-only NNX tree.
_DROP_SUBSTRINGS = (
    "audio_tower.",
    "vision_tower.",
    "multi_modal_projector.",
    "embed_audio_tokens.",
    "embed_vision_tokens.",
)


def _should_skip(name: str) -> bool:
  return any(s in name for s in _DROP_SUBSTRINGS)


def _resolve_nnx_param(model: nnx.Module, dotted_path: str) -> nnx.Param | None:
  """Look up `model.a.b.c.weight` on an NNX module tree.

  Returns None if any segment is missing (we use that to silently drop shared-KV
  projections and similar).

  Accepts list indices: ``model.layers.0.foo`` resolves via Python list
  indexing on the ``layers`` list.
  """
  node = model
  parts = dotted_path.split(".")
  for p in parts:
    if p.isdigit() and isinstance(node, list):
      idx = int(p)
      if idx >= len(node):
        return None
      node = node[idx]
      continue
    if not hasattr(node, p):
      return None
    node = getattr(node, p)
    if node is None:
      return None
  if isinstance(node, nnx.Param):
    return node
  return None


def load_hf_weights(
    model: nnx.Module,
    model_id_or_path: str,
    *,
    dtype: jnp.dtype | None = None,
    weights_dtype: jnp.dtype | None = None,
    shardings: dict | None = None,
    verbose: bool = False,
) -> dict[str, int]:
  """Copy HF weights into the NNX param tree in place.

  ``weights_dtype`` (exp 52+) is the storage dtype on the NNX side —
  e.g. fp32 for master-weights AMP, bf16 for legacy. If both
  ``weights_dtype`` and ``dtype`` are given, ``weights_dtype`` wins.
  If neither is given, the tensor's native HF dtype is preserved.

  ``shardings`` (exp 52+) is an optional ``{nnx_path: NamedSharding}``
  lookup. When provided, each loaded tensor is ``jax.device_put``'d
  with the matching sharding so that fp32-master tensors never
  materialize fully on a single device (PLE embed is 10.5 GiB in fp32
  — exceeds per-chip HBM headroom on v6e-4 if landed replicated).

  Returns a small report dict with counts of assigned / skipped /
  missing params.
  """
  target_dtype = weights_dtype if weights_dtype is not None else dtype
  paths = _iter_safetensors_paths(model_id_or_path)
  # PORT: shared-KV layers don't own k_proj/v_proj/k_norm/v_norm in the
  # NNX tree (we set them to None during __init__). HF checkpoint still
  # carries these weights; drop them here.
  num_layers = model.config.num_hidden_layers
  num_shared = getattr(model.config, "num_kv_shared_layers", 0)
  first_shared = num_layers - num_shared if num_shared else num_layers
  shared_drop_pat = re.compile(
      rf"model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\."
  )

  stats = {
      "assigned": 0,
      "skipped_modality": 0,
      "skipped_shared_kv": 0,
      "skipped_tied_lm_head": 0,
      "missing": 0,
  }
  missing_names: list[str] = []

  for hf_name, tensor in _iter_hf_tensors(paths):
    if _should_skip(hf_name):
      stats["skipped_modality"] += 1
      continue
    name = _strip_prefix(hf_name)
    # Skip lm_head if tied (we don't instantiate lm_head in that case).
    if name == "lm_head.weight" and getattr(model, "_tied", False):
      stats["skipped_tied_lm_head"] += 1
      continue
    # Skip shared-KV-layer projections.
    m = shared_drop_pat.match(name)
    if m:
      layer_idx = int(m.group(1))
      if num_shared and layer_idx >= first_shared:
        stats["skipped_shared_kv"] += 1
        continue
    # Try to resolve.
    param = _resolve_nnx_param(model, name)
    if param is None:
      stats["missing"] += 1
      if len(missing_names) < 20:
        missing_names.append(name)
      continue
    # Convert torch -> jnp. torch bf16 doesn't round-trip through numpy
    # (no native bf16 dtype in numpy), so cast to fp32 first on host
    # (CPU RAM), then either (a) scatter-shard directly into per-device
    # HBM via jax.device_put(np_array, sharding) — never materializing
    # the full fp32 tensor on any single device — or (b) fall back to
    # the old un-sharded jnp.asarray path for callers that don't pass
    # a sharding plan.
    t = tensor.detach()
    import torch as _torch
    import numpy as _np

    # Build a host-side numpy array in either fp32 (for HF bf16 shards,
    # which numpy can't express natively) or the tensor's native dtype.
    if t.dtype == _torch.bfloat16:
      np_arr = t.to(_torch.float32).numpy()  # host fp32
      host_dtype = "fp32_from_bf16"
    else:
      np_arr = t.numpy()
      host_dtype = str(t.dtype)

    sh = shardings.get(name) if shardings is not None else None

    if sh is not None:
      # Scatter-shard-first path. Cast on host (cheap) so that
      # target_dtype matches the per-device shard dtype exactly — no
      # full-tensor device fp32 materialization.
      if target_dtype is not None:
        # Cast on host using numpy for fp32 / jnp for bf16.
        if target_dtype == jnp.bfloat16:
          # Go through jax.numpy on host to get bf16 (numpy has no
          # native bf16). Allocate a small CPU-device array then
          # move it to the sharding.
          # jax.device_put on a numpy array with a NamedSharding
          # scatters by sharding layout, so per-device bf16 is
          # materialized directly.
          pass  # numpy fp32/... stays; jax will cast on put below
        elif target_dtype == jnp.float32:
          if np_arr.dtype != _np.float32:
            np_arr = np_arr.astype(_np.float32)
        else:
          np_arr = np_arr.astype(target_dtype)  # trust np.dtype path
      # Shape check on host.
      if tuple(np_arr.shape) != tuple(param.value.shape):
        raise ValueError(
            f"Shape mismatch for {name!r}: HF={tuple(np_arr.shape)} "
            f"vs NNX={tuple(param.value.shape)}"
        )
      arr = jax.device_put(np_arr, sh)
      if target_dtype is not None and arr.dtype != target_dtype:
        # e.g. bf16 target from a fp32 host buffer: astype runs per-
        # shard on-device (cheap: shard is ~1/fsdp of the full size).
        arr = arr.astype(target_dtype)
    else:
      # Legacy un-sharded path (replicated on device 0 at first).
      arr = jnp.asarray(np_arr)
      if t.dtype == _torch.bfloat16:
        arr = arr.astype(jnp.bfloat16)
      if target_dtype is not None:
        arr = arr.astype(target_dtype)
      if tuple(arr.shape) != tuple(param.value.shape):
        raise ValueError(
            f"Shape mismatch for {name!r}: HF={tuple(arr.shape)} "
            f"vs NNX={tuple(param.value.shape)}"
        )
    param.value = arr
    stats["assigned"] += 1

  if verbose:
    print(f"[weight-loader] {stats}")
    if missing_names:
      print("[weight-loader] missing (first 20):")
      for n in missing_names:
        print(f"  - {n}")
  return stats


__all__ = ["load_hf_weights"]
