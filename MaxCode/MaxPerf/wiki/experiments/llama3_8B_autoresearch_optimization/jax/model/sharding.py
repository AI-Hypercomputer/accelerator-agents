# pylint: skip-file
"""Mesh + per-parameter sharding rules for Llama 3 8B on TPU — JAX-native.

Mirrors `../../torchax/model/sharding.py` (same tensor-axis mapping under
FSDP and Megatron-style TP) and reuses the mesh helpers from the sibling
Gemma 4 jax sharding module conceptually (we re-implement them locally so
this folder is self-contained).

Two strategies:

  ``fsdp`` (default) — every >=2D param is sharded on its largest divisible
                       dim across a 1D ``'fsdp'`` mesh axis. RMSNorm
                       weights (1D) shard along `hidden=4096` if it
                       divides the FSDP size.
  ``tp``             — Megatron recipe: column-parallel Q/K/V/gate/up/lm_head;
                       row-parallel O/down; vocab-sharded embed. 2D
                       ``(fsdp, tp)`` mesh.

Llama 3 8B dims:
  hidden = 4096, ffn = 14336, vocab = 128256, n_heads = 32, n_kv_heads = 8,
  head_dim = 128. All hidden/ffn/vocab dims are divisible by 8 → both
  fsdp=8 and tp=8 are sharding-friendly.
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


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------


def get_fsdp_mesh(fsdp: Optional[int] = None) -> Mesh:
  n = jax.device_count()
  if fsdp is None or fsdp == 0:
    fsdp = n
  if fsdp != n:
    raise ValueError(f"FSDP mesh size {fsdp} != jax.device_count()={n}.")
  devices = mesh_utils.create_device_mesh((fsdp,))
  return Mesh(devices, axis_names=(AXIS_FSDP,))


def get_tp_mesh(fsdp: int = 1, tp: Optional[int] = None) -> Mesh:
  if tp is None:
    tp = jax.device_count() // max(fsdp, 1)
  total = fsdp * tp
  avail = jax.device_count()
  if total != avail:
    raise ValueError(
        f"2D mesh fsdp={fsdp} × tp={tp} requires {total} devices, "
        f"but jax.device_count()={avail}."
    )
  devices = mesh_utils.create_device_mesh((fsdp, tp))
  return Mesh(devices, axis_names=(AXIS_FSDP, AXIS_TP))


def get_mesh(
    strategy: str = "fsdp",
    *,
    fsdp: Optional[int] = None,
    tp: Optional[int] = None,
) -> Mesh:
  if strategy == "fsdp":
    return get_fsdp_mesh(fsdp)
  if strategy == "tp":
    return get_tp_mesh(fsdp=fsdp or 1, tp=tp)
  raise ValueError(f"unknown strategy: {strategy!r}")


def _is_tp_mesh(mesh: Mesh) -> bool:
  return AXIS_TP in mesh.axis_names


# -----------------------------------------------------------------------------
# Param tree traversal
# -----------------------------------------------------------------------------


def _iter_params(
    module: nnx.Module, prefix: str = ""
) -> Iterable[tuple[str, nnx.Param]]:
  """Yield (dotted_path, nnx.Param) pairs for every Param in the tree.

  Identical to the Gemma 4 trainer's helper — duplicated so this folder is
  self-contained per the program convention (no cross-experiment imports).
  """
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


# -----------------------------------------------------------------------------
# Wildcard-keyed sharding map (mirrors torchax sibling)
# -----------------------------------------------------------------------------
# The torchax SHARDING_MAP keys on HF safetensors names; here we key on NNX
# attribute paths. The shapes line up (we mirror the (out, in) Linear layout
# for HF parity) so the same partition specs apply.

# Unscanned plan. Tuple is the PartitionSpec across canonical mesh axes
# (fsdp, tp). Empty tuple = replicated. ``None`` entries leave a dim
# unsharded.
SHARDING_PLAN: Dict[str, Tuple] = {
    "model.embed_tokens.weight": ("fsdp", "tp"),  # (V, D)
    "model.layers.*.self_attn.q_proj.weight": ("tp", "fsdp"),  # (Hq*hd, D)
    "model.layers.*.self_attn.k_proj.weight": ("tp", "fsdp"),  # (Hkv*hd, D)
    "model.layers.*.self_attn.v_proj.weight": ("tp", "fsdp"),  # (Hkv*hd, D)
    "model.layers.*.self_attn.o_proj.weight": ("fsdp", "tp"),  # (D, Hq*hd)
    "model.layers.*.mlp.gate_proj.weight": ("tp", "fsdp"),  # (ffn, D)
    "model.layers.*.mlp.up_proj.weight": ("tp", "fsdp"),
    "model.layers.*.mlp.down_proj.weight": ("fsdp", "tp"),  # (D, ffn)
    "model.layers.*.input_layernorm.weight": ("fsdp",),
    "model.layers.*.post_attention_layernorm.weight": ("fsdp",),
    "model.norm.weight": ("fsdp",),
    "lm_head.weight": ("tp", "fsdp"),  # (V, D)
}


# Scan-over-layers plan: leading layer-stack dim is unsharded (None).
SCAN_SHARDING_PLAN: Dict[str, Tuple] = {
    "model.embed_tokens.weight": ("fsdp", "tp"),
    "model.scanned_layers.self_attn.q_proj.weight": (None, "tp", "fsdp"),
    "model.scanned_layers.self_attn.k_proj.weight": (None, "tp", "fsdp"),
    "model.scanned_layers.self_attn.v_proj.weight": (None, "tp", "fsdp"),
    "model.scanned_layers.self_attn.o_proj.weight": (None, "fsdp", "tp"),
    "model.scanned_layers.mlp.gate_proj.weight": (None, "tp", "fsdp"),
    "model.scanned_layers.mlp.up_proj.weight": (None, "tp", "fsdp"),
    "model.scanned_layers.mlp.down_proj.weight": (None, "fsdp", "tp"),
    "model.scanned_layers.input_layernorm.weight": (None, "fsdp"),
    "model.scanned_layers.post_attention_layernorm.weight": (None, "fsdp"),
    "model.norm.weight": ("fsdp",),
    "lm_head.weight": ("tp", "fsdp"),
}


def _process_sharding_name(name: str) -> str:
  """Replace integer tokens (layer indices) with `*` for wildcard match."""

  def _is_int(t):
    try:
      int(t)
      return True
    except ValueError:
      return False

  return ".".join("*" if _is_int(t) else t for t in name.split("."))


def _spec_for(plan: Mapping[str, Tuple], path: str) -> Optional[Tuple]:
  return plan.get(_process_sharding_name(path))


def _resolve_axes_for_mesh(spec: Tuple, mesh: Mesh) -> Tuple:
  """Drop any axis names not present in the mesh (e.g.

  for FSDP-only meshes we strip the 'tp' entries; the remaining dim becomes
  None).
  """
  axis_names = set(mesh.axis_names)
  out = tuple(a if (a is None or a in axis_names) else None for a in spec)
  return out


# -----------------------------------------------------------------------------
# Plan dataclass
# -----------------------------------------------------------------------------


@dataclass
class ShardingPlan:
  shardings: Dict[str, NamedSharding]
  buckets: Dict[str, List[str]]
  notes: List[str] = field(default_factory=list)


def build_plan(
    model: nnx.Module,
    mesh: Mesh,
    *,
    use_scan: bool = False,
) -> ShardingPlan:
  """Walk every Param in `model` and assign a NamedSharding by matching

  its dotted path against the relevant plan map (SHARDING_PLAN or
  SCAN_SHARDING_PLAN). Params without an explicit entry are replicated
  and listed under the ``replicated`` bucket so the user can audit.
  """
  plan_map = SCAN_SHARDING_PLAN if use_scan else SHARDING_PLAN
  shardings: Dict[str, NamedSharding] = {}
  buckets: Dict[str, List[str]] = {"matched": [], "replicated": []}
  notes: List[str] = []
  for path, param in _iter_params(model):
    spec = _spec_for(plan_map, path)
    if spec is None:
      shardings[path] = NamedSharding(mesh, P())
      buckets["replicated"].append(path)
      continue
    spec = _resolve_axes_for_mesh(spec, mesh)
    shardings[path] = NamedSharding(mesh, P(*spec))
    buckets["matched"].append(path)
  if buckets["replicated"]:
    notes.append(
        f"sharding: {len(buckets['replicated'])} param(s) had no plan "
        "entry and were replicated. Examples: "
        f"{buckets['replicated'][:5]}"
    )
  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


def apply_sharding(model: nnx.Module, plan: ShardingPlan) -> None:
  """Apply the plan in place: jax.device_put each NNX Param onto its

  NamedSharding. The NNX param tree is mutated; no return value.
  """
  for path, param in _iter_params(model):
    sh = plan.shardings.get(path)
    if sh is None:
      continue
    param.value = jax.device_put(param.value, sh)


# -----------------------------------------------------------------------------
# Activation-sharding helpers
# -----------------------------------------------------------------------------


def input_sharding(mesh: Mesh) -> NamedSharding:
  """Per-batch shard along ``fsdp``."""
  if _is_tp_mesh(mesh):
    return NamedSharding(mesh, P(AXIS_FSDP, None))
  return NamedSharding(mesh, P(AXIS_FSDP, None))


def replicated(mesh: Mesh) -> NamedSharding:
  return NamedSharding(mesh, P())


__all__ = [
    "AXIS_FSDP",
    "AXIS_TP",
    "SHARDING_PLAN",
    "SCAN_SHARDING_PLAN",
    "ShardingPlan",
    "build_plan",
    "apply_sharding",
    "get_mesh",
    "get_fsdp_mesh",
    "get_tp_mesh",
    "input_sharding",
    "replicated",
    "_iter_params",
    "_process_sharding_name",
    "_spec_for",
]
