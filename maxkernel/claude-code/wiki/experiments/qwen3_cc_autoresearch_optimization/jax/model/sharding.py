"""Mesh + per-parameter sharding for the Flax NNX Qwen3 — JAX-native.

Mirrors `../../torchax/model/sharding.py` (same tensor-axis mapping under FSDP
and Megatron-style TP) so the two lanes are directly comparable. Keys on NNX
attribute paths, which match HF names dot-for-dot.

Qwen3 vs Llama3: adds the QK-norm params `self_attn.q_norm.weight` /
`k_norm.weight` (shape `(head_dim,)` — tiny, replicated). Everything else
identical (GQA + SwiGLU + RMSNorm).

For `tp_parallelism=1` (pure FSDP) the 'tp' axis is absent from the mesh and the
plan's 'tp' entries collapse to None (that dim stays unsharded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from flax import nnx
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS_FSDP = "fsdp"
AXIS_TP = "tp"


def _iter_params(
    module: nnx.Module, prefix: str = ""
) -> Iterable[tuple[str, nnx.Param]]:
  """Yield (dotted_path, nnx.Param) for every Param in the tree."""
  seen = set()
  for name in sorted(vars(module).keys()):
    if name.startswith("_"):
      continue
    attr = getattr(module, name)
    path = f"{prefix}.{name}" if prefix else name
    if isinstance(attr, nnx.Param):
      if id(attr) in seen:
        continue
      seen.add(id(attr))
      yield path, attr
    elif isinstance(attr, nnx.Module):
      yield from _iter_params(attr, path)
    elif isinstance(attr, list):
      for i, elem in enumerate(attr):
        if isinstance(elem, nnx.Module):
          yield from _iter_params(elem, f"{path}.{i}")
    elif isinstance(attr, dict):
      for k, v in attr.items():
        if isinstance(v, nnx.Module):
          yield from _iter_params(v, f"{path}.{k}")


# Tuple = PartitionSpec across canonical mesh axes (fsdp, tp). () = replicated.
SHARDING_PLAN: Dict[str, Tuple] = {
    "model.embed_tokens.weight": ("fsdp", "tp"),  # (V, D)
    "model.layers.*.self_attn.q_proj.weight": ("tp", "fsdp"),  # (Hq*hd, D)
    "model.layers.*.self_attn.k_proj.weight": ("tp", "fsdp"),  # (Hkv*hd, D)
    "model.layers.*.self_attn.v_proj.weight": ("tp", "fsdp"),  # (Hkv*hd, D)
    "model.layers.*.self_attn.o_proj.weight": ("fsdp", "tp"),  # (D, Hq*hd)
    "model.layers.*.self_attn.q_norm.weight": (),  # (head_dim,) — Qwen3 QK-norm
    "model.layers.*.self_attn.k_norm.weight": (),
    "model.layers.*.mlp.gate_proj.weight": ("tp", "fsdp"),  # (ffn, D)
    "model.layers.*.mlp.up_proj.weight": ("tp", "fsdp"),
    "model.layers.*.mlp.down_proj.weight": ("fsdp", "tp"),  # (D, ffn)
    "model.layers.*.input_layernorm.weight": ("fsdp",),
    "model.layers.*.post_attention_layernorm.weight": ("fsdp",),
    "model.norm.weight": ("fsdp",),
    "lm_head.weight": ("tp", "fsdp"),  # (V, D)
}


def get_mesh(*, fsdp: Optional[int] = None, tp: int = 1) -> Mesh:
  n = jax.device_count()
  if fsdp is None or fsdp == 0:
    fsdp = n // max(tp, 1)
  if fsdp * tp != n:
    raise ValueError(f"mesh fsdp={fsdp} × tp={tp} != device_count={n}")
  if tp == 1:
    return Mesh(mesh_utils.create_device_mesh((fsdp,)), axis_names=(AXIS_FSDP,))
  return Mesh(
      mesh_utils.create_device_mesh((fsdp, tp)), axis_names=(AXIS_FSDP, AXIS_TP)
  )


def _process_sharding_name(name: str) -> str:
  def _is_int(t):
    try:
      int(t)
      return True
    except ValueError:
      return False

  return ".".join("*" if _is_int(t) else t for t in name.split("."))


def _resolve_axes_for_mesh(spec: Tuple, mesh: Mesh) -> Tuple:
  names = set(mesh.axis_names)
  return tuple(a if (a is None or a in names) else None for a in spec)


@dataclass
class ShardingPlan:
  shardings: Dict[str, NamedSharding]
  buckets: Dict[str, List[str]]
  notes: List[str] = field(default_factory=list)


def build_plan(model: nnx.Module, mesh: Mesh) -> ShardingPlan:
  shardings: Dict[str, NamedSharding] = {}
  buckets = {"matched": [], "replicated": []}
  notes: List[str] = []
  for path, _ in _iter_params(model):
    spec = SHARDING_PLAN.get(_process_sharding_name(path))
    # Stacked decoder layers: params are stored as ONE layer with a leading
    # [num_layers] axis, so the path is "model.layers.<rest>" (no per-layer
    # index). Map to the per-layer rule and prepend None (layer axis replicated).
    if spec is None and path.startswith("model.layers."):
      base = SHARDING_PLAN.get("model.layers.*." + path[len("model.layers.") :])
      if base is not None:
        spec = (None,) + base
    if spec is None:
      shardings[path] = NamedSharding(mesh, P())
      buckets["replicated"].append(path)
      continue
    shardings[path] = NamedSharding(
        mesh, P(*_resolve_axes_for_mesh(spec, mesh))
    )
    buckets["matched"].append(path)
  if buckets["replicated"]:
    notes.append(
        f"{len(buckets['replicated'])} param(s) replicated (no plan "
        f"entry): {buckets['replicated'][:5]}"
    )
  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


def apply_sharding(model: nnx.Module, plan: ShardingPlan) -> None:
  for path, param in _iter_params(model):
    sh = plan.shardings.get(path)
    if sh is not None:
      param.value = jax.device_put(param.value, sh)


def input_sharding(mesh: Mesh) -> NamedSharding:
  return NamedSharding(mesh, P(AXIS_FSDP, None))


__all__ = [
    "AXIS_FSDP",
    "AXIS_TP",
    "SHARDING_PLAN",
    "ShardingPlan",
    "get_mesh",
    "build_plan",
    "apply_sharding",
    "input_sharding",
    "_iter_params",
]
