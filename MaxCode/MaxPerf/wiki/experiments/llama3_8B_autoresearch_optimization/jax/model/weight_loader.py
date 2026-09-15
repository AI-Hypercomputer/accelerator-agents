# pylint: skip-file
"""HuggingFace safetensors -> Flax NNX weight loader for Llama 3 8B.

Loads ``meta-llama/Meta-Llama-3-8B`` checkpoint files into a freshly built
``LlamaForCausalLM`` (or ``LlamaForCausalLMScan``) NNX tree. Each tensor
is `jax.device_put` with the matching ``NamedSharding`` so fp32-master
tensors never materialize fully on a single device (the lm_head + embed
together are 4 GiB at fp32 — fits per-chip but tight; layer stacks for
the scan path are 32× larger).

Llama 3 has its OWN lm_head — there is no weight-tying special case
(unlike Gemma 4 E4B). No multimodal / no shared-KV layers / no
modality-specific keys.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterator, Mapping, Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
# Safetensors discovery + iteration
# -----------------------------------------------------------------------------


def _iter_safetensors_paths(model_id_or_path: str) -> list[str]:
  """Resolve a HF model id or path to a list of safetensors shard paths."""
  if os.path.isdir(model_id_or_path):
    paths = sorted(glob.glob(os.path.join(model_id_or_path, "*.safetensors")))
    if paths:
      return paths
  if os.path.isfile(model_id_or_path):
    return [model_id_or_path]
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
  from safetensors import safe_open

  for p in paths:
    with safe_open(p, framework="pt") as f:
      for k in f.keys():
        yield k, f.get_tensor(k)


def _to_numpy(t) -> np.ndarray:
  """Convert a torch tensor to a host numpy array.

  torch.bfloat16 has no native numpy dtype so we promote to fp32 on host (cheap;
  cast on device after sharding).
  """
  import torch as _torch

  t = t.detach()
  if t.dtype == _torch.bfloat16:
    return t.to(_torch.float32).numpy()
  return t.numpy()


# -----------------------------------------------------------------------------
# NNX param tree resolution
# -----------------------------------------------------------------------------


def _resolve_nnx_param(
    model: nnx.Module, dotted_path: str
) -> Optional[nnx.Param]:
  """Walk ``model.a.b.c.weight``.

  Returns None if any segment is missing; supports list indexing on
  ``layers.<i>``.
  """
  node: object = model
  for p in dotted_path.split("."):
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


# -----------------------------------------------------------------------------
# HF -> NNX path translation
# -----------------------------------------------------------------------------


# Llama 3 8B's HF checkpoint keys are already in `model.<...>` /
# `lm_head.weight` form, so no prefix stripping is needed.
_LAYER_KEY_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")


def _hf_name_to_nnx_path(hf_name: str) -> str:
  """For the unscanned model, HF names map 1:1 to NNX paths."""
  return hf_name


def _scan_subpath_for(hf_layer_subname: str) -> str:
  """For the scan model, an HF per-layer key like

  ``self_attn.q_proj.weight`` becomes
  ``model.scanned_layers.self_attn.q_proj.weight``. The leading layer
  index is folded into the leading dim of the stacked tensor instead.
  """
  return f"model.scanned_layers.{hf_layer_subname}"


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def _cast_for_target(
    np_arr: np.ndarray,
    target_dtype: Optional[jnp.dtype],
) -> np.ndarray:
  """Cast on host so the per-device shard dtype matches `target_dtype` exactly.

  We avoid the full-tensor fp32 round-trip when the target is bf16: numpy has no
  bf16, so we leave the array as fp32 on host and let `jax.device_put(np_arr,
  sharding).astype(bf16)` cast per-shard on-device (each shard is ~1/fsdp of the
  full size).
  """
  if target_dtype is None:
    return np_arr
  if target_dtype == jnp.bfloat16:
    return np_arr  # cast happens after device_put
  if target_dtype == jnp.float32:
    if np_arr.dtype != np.float32:
      return np_arr.astype(np.float32)
    return np_arr
  # generic
  return np_arr.astype(target_dtype)


def _put_param(
    param: nnx.Param,
    np_arr: np.ndarray,
    *,
    sharding,
    target_dtype: Optional[jnp.dtype],
    name_for_error: str,
) -> None:
  if tuple(np_arr.shape) != tuple(param.value.shape):
    raise ValueError(
        f"Shape mismatch for {name_for_error!r}: HF={tuple(np_arr.shape)} "
        f"vs NNX={tuple(param.value.shape)}"
    )
  if sharding is not None:
    arr = jax.device_put(np_arr, sharding)
  else:
    arr = jnp.asarray(np_arr)
  if target_dtype is not None and arr.dtype != target_dtype:
    arr = arr.astype(target_dtype)
  param.value = arr


def load_hf_weights(
    model: nnx.Module,
    model_id_or_path: str,
    *,
    weights_dtype: Optional[jnp.dtype] = None,
    shardings: Optional[Mapping[str, "jax.sharding.NamedSharding"]] = None,
    use_scan: bool = False,
    verbose: bool = False,
) -> dict:
  """Copy HF weights into the NNX tree in place.

  For the unscanned model (``use_scan=False``), each HF tensor maps 1:1
  to an NNX param.

  For the scan model (``use_scan=True``), the 32 per-layer HF tensors
  of any given subpath (e.g. ``model.layers.{i}.self_attn.q_proj.weight``)
  are stacked along a new leading dim and written to a single NNX
  param at ``model.scanned_layers.self_attn.q_proj.weight``. The
  sharding plan in `model/sharding.py` adds a leading ``None`` for
  that dim.

  Returns a stats dict with counts of assigned and missing params.
  """
  paths = _iter_safetensors_paths(model_id_or_path)
  target_dtype = weights_dtype

  stats = {"assigned": 0, "missing": 0, "stacked_groups": 0}
  missing_names: list[str] = []

  if not use_scan:
    # ---------------------------------------------------------------
    # Unscanned: 1:1 HF -> NNX mapping.
    # ---------------------------------------------------------------
    for hf_name, tensor in _iter_hf_tensors(paths):
      np_arr = _to_numpy(tensor)
      np_arr = _cast_for_target(np_arr, target_dtype)
      nnx_path = _hf_name_to_nnx_path(hf_name)
      param = _resolve_nnx_param(model, nnx_path)
      if param is None:
        stats["missing"] += 1
        if len(missing_names) < 20:
          missing_names.append(nnx_path)
        continue
      sh = shardings.get(nnx_path) if shardings is not None else None
      _put_param(
          param,
          np_arr,
          sharding=sh,
          target_dtype=target_dtype,
          name_for_error=nnx_path,
      )
      stats["assigned"] += 1
  else:
    # ---------------------------------------------------------------
    # Scan: collect per-layer tensors first, then stack and assign
    # to the single stacked NNX param. Groups are keyed by the
    # post-`model.layers.<i>.` subpath.
    # ---------------------------------------------------------------
    groups: dict[str, dict[int, np.ndarray]] = {}
    non_layer: list[tuple[str, np.ndarray]] = []
    for hf_name, tensor in _iter_hf_tensors(paths):
      np_arr = _to_numpy(tensor)
      np_arr = _cast_for_target(np_arr, target_dtype)
      m = _LAYER_KEY_RE.match(hf_name)
      if m:
        layer_idx = int(m.group(1))
        subname = m.group(2)  # e.g. self_attn.q_proj.weight
        groups.setdefault(subname, {})[layer_idx] = np_arr
      else:
        # embed_tokens / norm / lm_head / etc. — not under a layer.
        non_layer.append((hf_name, np_arr))

    # Stack per-group along new leading dim, write to scanned param.
    for subname, by_idx in groups.items():
      indices = sorted(by_idx.keys())
      arrs = [by_idx[i] for i in indices]
      stacked = np.stack(arrs, axis=0)
      nnx_path = _scan_subpath_for(subname)
      param = _resolve_nnx_param(model, nnx_path)
      if param is None:
        stats["missing"] += 1
        if len(missing_names) < 20:
          missing_names.append(nnx_path)
        continue
      sh = shardings.get(nnx_path) if shardings is not None else None
      _put_param(
          param,
          stacked,
          sharding=sh,
          target_dtype=target_dtype,
          name_for_error=nnx_path,
      )
      stats["assigned"] += 1
      stats["stacked_groups"] += 1

    for hf_name, np_arr in non_layer:
      param = _resolve_nnx_param(model, hf_name)
      if param is None:
        stats["missing"] += 1
        if len(missing_names) < 20:
          missing_names.append(hf_name)
        continue
      sh = shardings.get(hf_name) if shardings is not None else None
      _put_param(
          param,
          np_arr,
          sharding=sh,
          target_dtype=target_dtype,
          name_for_error=hf_name,
      )
      stats["assigned"] += 1

  if verbose:
    print(f"[weight-loader] {stats}", flush=True)
    if missing_names:
      print("[weight-loader] missing (first 20):", flush=True)
      for n in missing_names:
        print(f"  - {n}", flush=True)
  return stats


__all__ = ["load_hf_weights"]
