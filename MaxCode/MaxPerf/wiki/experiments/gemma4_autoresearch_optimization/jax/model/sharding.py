# pylint: skip-file
"""Mesh + per-parameter sharding rules for Gemma 4 on TPU — JAX-native.

Mirrors `../../torchax/model/sharding.py` so the FSDP / TP plans match
between the torchax baseline and the native-JAX trainer.

Two strategies:

* ``fsdp`` (default) — every >=2D param is sharded along its largest
  divisible dim over a 1D ``'fsdp'`` mesh axis that spans all devices.
* ``tp`` — NeMo-Megatron tensor-parallel recipe (columnwise Q / MLP
  up/gate, rowwise O / MLP down, vocab-sharded embed). 2D (dp, tp) mesh.

The plan is built **off param path names** (traversal of the NNX graph)
not from HF state_dict keys like the torchax version — but the name
pattern matching keeps functional parity (``.q_proj.weight``,
``.gate_proj.weight`` etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


AXIS_FSDP = "fsdp"
AXIS_DP = "dp"
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


def get_tp_mesh(dp: int = 1, tp: Optional[int] = None) -> Mesh:
  if tp is None:
    tp = jax.device_count() // max(dp, 1)
  total = dp * tp
  avail = jax.device_count()
  if total != avail:
    raise ValueError(
        f"TP mesh (dp={dp}, tp={tp}) needs {total} devices, "
        f"but jax.device_count()={avail}."
    )
  devices = mesh_utils.create_device_mesh((dp, tp))
  return Mesh(devices, axis_names=(AXIS_DP, AXIS_TP))


def get_mesh(
    strategy: str = "fsdp",
    *,
    dp: int = 1,
    tp: Optional[int] = None,
    fsdp: Optional[int] = None,
) -> Mesh:
  if strategy == "fsdp":
    return get_fsdp_mesh(fsdp)
  if strategy == "tp":
    return get_tp_mesh(dp=dp, tp=tp)
  raise ValueError(f"unknown strategy: {strategy!r}")


def _is_fsdp_mesh(mesh: Mesh) -> bool:
  return AXIS_FSDP in mesh.axis_names


# -----------------------------------------------------------------------------
# Param tree traversal
# -----------------------------------------------------------------------------


def _iter_params(
    module: nnx.Module, prefix: str = ""
) -> Iterable[tuple[str, nnx.Param]]:
  """Yield (dotted_path, nnx.Param) pairs for every Param in the tree."""
  # NNX stores submodules as attributes. Walk via __dict__ like torch
  # nn.Module.named_parameters would.
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
# FSDP plan
# -----------------------------------------------------------------------------


@dataclass
class ShardingPlan:
  shardings: Dict[str, NamedSharding]
  buckets: Dict[str, List[str]]
  notes: List[str] = field(default_factory=list)


def plan_fsdp_shardings(
    param_shapes: Mapping[str, Tuple[int, ...]],
    mesh: Mesh,
) -> ShardingPlan:
  fsdp_size = mesh.shape[AXIS_FSDP]
  buckets: Dict[str, List[str]] = {
      "fsdp_shard": [],
      "replicated": [],
      "undivisible": [],
  }
  shardings: Dict[str, NamedSharding] = {}
  notes: List[str] = []

  for name, shape in param_shapes.items():
    if len(shape) == 0:
      shardings[name] = NamedSharding(mesh, P())
      buckets["replicated"].append(name)
      continue
    ordered = sorted(range(len(shape)), key=lambda i: -shape[i])
    shard_dim: Optional[int] = None
    for d in ordered:
      if shape[d] % fsdp_size == 0 and shape[d] >= fsdp_size:
        shard_dim = d
        break
    if shard_dim is None:
      shardings[name] = NamedSharding(mesh, P())
      buckets["undivisible"].append(name)
      continue
    spec = [None] * len(shape)
    spec[shard_dim] = AXIS_FSDP
    shardings[name] = NamedSharding(mesh, P(*spec))
    buckets["fsdp_shard"].append(name)

  if buckets["undivisible"]:
    notes.append(
        f"FSDP: {len(buckets['undivisible'])} param(s) have no dim "
        f"divisible by fsdp={fsdp_size}; replicated. Example: "
        f"{buckets['undivisible'][:3]}"
    )
  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# TP plan
# -----------------------------------------------------------------------------


_COL_SHARD = ("q_proj", "gate_proj", "up_proj")
_ROW_SHARD = ("o_proj", "down_proj")
_EMBED_LIKE = (
    "embed_tokens",
    "embed_tokens_per_layer",
    "lm_head",
    "per_layer_model_projection",
)


def _kv_head_shardable(num_kv_heads: int, tp: int) -> bool:
  return num_kv_heads % tp == 0 and num_kv_heads >= tp


def plan_tp_shardings(
    param_shapes: Mapping[str, Tuple[int, ...]],
    mesh: Mesh,
    *,
    num_kv_heads: int,
    num_attention_heads: int,
) -> ShardingPlan:
  tp = mesh.shape[AXIS_TP]
  dp = mesh.shape[AXIS_DP]

  buckets: Dict[str, List[str]] = {
      "col_shard": [],
      "row_shard": [],
      "embed_shard": [],
      "kv_replicated": [],
      "dp_shard": [],
      "replicated": [],
  }
  shardings: Dict[str, NamedSharding] = {}
  notes: List[str] = []
  if num_attention_heads % tp != 0:
    notes.append(
        f"WARNING: num_attention_heads={num_attention_heads} not "
        f"divisible by tp={tp}; Q/O sharding will be wrong."
    )
  kv_shardable = _kv_head_shardable(num_kv_heads, tp)
  if not kv_shardable:
    notes.append(
        f"GQA: num_kv_heads={num_kv_heads} does not divide tp={tp}. "
        "K/V projections replicated on tp axis."
    )

  def _shard_dp_like(shape: Tuple[int, ...]) -> Optional[P]:
    """FSDP-style: shard largest dim divisible by dp on 'dp'."""
    if dp <= 1 or len(shape) == 0:
      return None
    ordered = sorted(range(len(shape)), key=lambda i: -shape[i])
    for d in ordered:
      if shape[d] % dp == 0 and shape[d] >= dp:
        spec = [None] * len(shape)
        spec[d] = AXIS_DP
        return P(*spec)
    return None

  for name, shape in param_shapes.items():
    # K/V projections.
    if any(s in name for s in (".k_proj.", ".v_proj.")):
      if kv_shardable:
        spec = P(AXIS_TP, None) if len(shape) == 2 else P(AXIS_TP)
        shardings[name] = NamedSharding(mesh, spec)
        buckets["col_shard"].append(name)
      else:
        # Replicate on tp; dp-shard if possible for FSDP-over-TP.
        dp_spec = _shard_dp_like(shape)
        if dp_spec is not None:
          shardings[name] = NamedSharding(mesh, dp_spec)
          buckets["dp_shard"].append(name)
        else:
          shardings[name] = NamedSharding(mesh, P())
          buckets["kv_replicated"].append(name)
      continue
    if any(s in name for s in (f".{n}." for n in _COL_SHARD)):
      shardings[name] = NamedSharding(
          mesh, P(AXIS_TP, AXIS_DP) if dp > 1 else P(AXIS_TP, None)
      )
      buckets["col_shard"].append(name)
      continue
    if any(s in name for s in (f".{n}." for n in _ROW_SHARD)):
      shardings[name] = NamedSharding(
          mesh, P(AXIS_DP, AXIS_TP) if dp > 1 else P(None, AXIS_TP)
      )
      buckets["row_shard"].append(name)
      continue
    if any(s in name for s in (f"{n}." for n in _EMBED_LIKE)):
      # Shard vocab dim on tp.
      if len(shape) == 2:
        shardings[name] = NamedSharding(mesh, P(AXIS_TP, None))
        buckets["embed_shard"].append(name)
        continue
    # Fallback: try dp-shard, else replicate.
    dp_spec = _shard_dp_like(shape)
    if dp_spec is not None:
      shardings[name] = NamedSharding(mesh, dp_spec)
      buckets["dp_shard"].append(name)
    else:
      shardings[name] = NamedSharding(mesh, P())
      buckets["replicated"].append(name)
  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------


def get_param_sharding(model: nnx.Module, mesh: Mesh) -> ShardingPlan:
  """Build a sharding plan for every Param in the tree."""
  shapes = {path: tuple(p.value.shape) for path, p in _iter_params(model)}
  if _is_fsdp_mesh(mesh):
    return plan_fsdp_shardings(shapes, mesh)
  cfg = getattr(model, "config", None)
  if cfg is None:
    raise ValueError("model has no .config.")
  return plan_tp_shardings(
      shapes,
      mesh,
      num_kv_heads=cfg.num_key_value_heads,
      num_attention_heads=cfg.num_attention_heads,
  )


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
  if _is_fsdp_mesh(mesh):
    return NamedSharding(mesh, P(AXIS_FSDP, None))
  return NamedSharding(mesh, P(AXIS_DP, None))


def logits_sharding(mesh: Mesh) -> NamedSharding:
  if _is_fsdp_mesh(mesh):
    return NamedSharding(mesh, P(AXIS_FSDP, None, None))
  return NamedSharding(mesh, P(AXIS_DP, None, AXIS_TP))


def replicated(mesh: Mesh) -> NamedSharding:
  return NamedSharding(mesh, P())


__all__ = [
    "AXIS_FSDP",
    "AXIS_DP",
    "AXIS_TP",
    "ShardingPlan",
    "get_mesh",
    "get_fsdp_mesh",
    "get_tp_mesh",
    "get_param_sharding",
    "apply_sharding",
    "plan_fsdp_shardings",
    "plan_tp_shardings",
    "input_sharding",
    "logits_sharding",
    "replicated",
    "_iter_params",
]
