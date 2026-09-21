# pylint: skip-file
"""Mesh + per-parameter sharding rules for Gemma 4 on TPU.

Two strategies are supported:

* ``fsdp`` (**default**) — every ≥2D parameter is sharded along its largest
  divisible dim over a 1D ``'fsdp'`` mesh axis that spans all devices.
  GSPMD inserts the all-gather (fwd) / reduce-scatter (bwd) automatically.
  Matches PyTorch FSDP's ``FULL_SHARD`` semantics. Default for fine-tuning.
* ``tp`` — NeMo-Megatron tensor-parallel recipe (columnwise Q / MLP
  up/gate, rowwise O / MLP down, vocab-sharded embed). Adapted from the
  jax-huggingface Part 2 recipe for an 8-way TP on v6e-8. Use when a
  model doesn't fit on one chip's HBM or when decode latency is the
  headline metric.

For Gemma 4 E4B (~8B params with embeddings, ~4.5B effective via
Per-Layer Embeddings) FSDP is the right default — weights fit on one
v6e chip but optimizer state (AdamW fp32 ≈ 12 bytes/param = 96 GB) does
not. FSDP drops optimizer-state-per-chip to ~12 GB.

Not exercised — see the UNTESTED warning in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# -----------------------------------------------------------------------------
# Mesh axis names — only the axes actually present in a given mesh are used.
# -----------------------------------------------------------------------------

AXIS_FSDP = "fsdp"
AXIS_DP = "dp"
AXIS_TP = "tp"


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------


def get_fsdp_mesh(fsdp: Optional[int] = None) -> Mesh:
  """1D FSDP mesh spanning all visible devices.

  If ``fsdp`` is unset, uses ``jax.device_count()``. Refuses to silently
  drop devices.
  """
  n = jax.device_count()
  if fsdp is None:
    fsdp = n
  if fsdp != n:
    raise ValueError(
        f"FSDP mesh size {fsdp} != jax.device_count()={n}. "
        f"Pass --fsdp {n} or unset the flag."
    )
  devices = mesh_utils.create_device_mesh((fsdp,))
  return Mesh(devices, axis_names=(AXIS_FSDP,))


def get_tp_mesh(dp: int = 1, tp: Optional[int] = None) -> Mesh:
  """2D (dp, tp) mesh for the tensor-parallel strategy.

  Defaults ``tp`` to ``jax.device_count() // dp``. Raises if
  ``dp * tp != jax.device_count()``.
  """
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
  """Dispatch to the mesh matching ``strategy``."""
  if strategy == "fsdp":
    return get_fsdp_mesh(fsdp)
  if strategy == "tp":
    return get_tp_mesh(dp=dp, tp=tp)
  raise ValueError(f"unknown strategy: {strategy!r}")


def _is_fsdp_mesh(mesh: Mesh) -> bool:
  return AXIS_FSDP in mesh.axis_names


# -----------------------------------------------------------------------------
# Param sharding — FSDP strategy
# -----------------------------------------------------------------------------


@dataclass
class ShardingPlan:
  """What to shard and how. Returned by the plan_* functions."""

  shardings: Dict[str, NamedSharding]
  buckets: Dict[str, list]  # bucket name -> param names (for reporting)
  notes: list  # strings for the report / logging


def plan_fsdp_shardings(
    param_names: Iterable[str],
    param_shapes: Mapping[str, Tuple[int, ...]],
    mesh: Mesh,
) -> ShardingPlan:
  """FSDP: shard every ≥2D param on its largest divisible dim over ``'fsdp'``.

  1D params (norms, biases, rotary buffers) are sharded if the dim is
  divisible by ``fsdp``, else replicated (conservative).
  Params with no dim divisible by ``fsdp`` are replicated with a note.
  """
  fsdp_size = mesh.shape[AXIS_FSDP]
  buckets: Dict[str, list] = {
      "fsdp_shard": [],
      "replicated": [],
      "undivisible": [],
  }
  shardings: Dict[str, NamedSharding] = {}
  notes: list = []

  for name in param_names:
    shape = param_shapes.get(name)
    if shape is None or len(shape) == 0:
      shardings[name] = NamedSharding(mesh, P())
      buckets["replicated"].append(name)
      continue

    # Find largest dim divisible by fsdp_size.
    ordered = sorted(range(len(shape)), key=lambda i: -shape[i])
    shard_dim: Optional[int] = None
    for d in ordered:
      if shape[d] % fsdp_size == 0:
        shard_dim = d
        break

    if shard_dim is None:
      shardings[name] = NamedSharding(mesh, P())
      buckets["undivisible"].append(name)
      continue

    spec_axes = [None] * len(shape)
    spec_axes[shard_dim] = AXIS_FSDP
    shardings[name] = NamedSharding(mesh, P(*spec_axes))
    buckets["fsdp_shard"].append(name)

  if buckets["undivisible"]:
    notes.append(
        f"FSDP: {len(buckets['undivisible'])} param(s) have no dim divisible "
        f"by fsdp={fsdp_size}; they are replicated. Example: "
        f"{buckets['undivisible'][:3]}"
    )

  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Param sharding — TP strategy (NeMo-Megatron)
# -----------------------------------------------------------------------------

# Gemma-family HF param-name convention (as of transformers main, 2026-04).
_COL_SHARD_SUBSTR = ("q_proj", "gate_proj", "up_proj")
_ROW_SHARD_SUBSTR = ("o_proj", "down_proj")
_ROW_SHARD_EMBED = ("embed_tokens", "lm_head")


def _gqa_can_shard_kv(num_kv_heads: int, tp: int) -> bool:
  """Can we shard K/V heads over ``tp`` without replication?

  Requires tp to divide num_kv_heads. For Gemma 4 E4B (kv=2, tp=8) this is
  False: we would need a broadcasted GQA sharding, not the basic row/column
  pattern. Future hypothesis.
  """
  return num_kv_heads % tp == 0 and num_kv_heads >= tp


def plan_tp_shardings(
    param_names: Iterable[str],
    mesh: Mesh,
    *,
    num_kv_heads: int,
    num_attention_heads: int,
) -> ShardingPlan:
  """NeMo-Megatron TP recipe adapted for GQA."""
  tp = mesh.shape[AXIS_TP]

  notes: list = []
  if num_attention_heads % tp != 0:
    notes.append(
        f"WARNING: num_attention_heads={num_attention_heads} not divisible "
        f"by tp={tp}; Q/O sharding will be wrong. Consider a different tp."
    )
  kv_head_shardable = _gqa_can_shard_kv(num_kv_heads, tp)
  if not kv_head_shardable:
    notes.append(
        f"GQA: num_kv_heads={num_kv_heads} does not divide tp={tp}. "
        "K/V projections will be REPLICATED over the tp axis."
    )

  buckets: Dict[str, list] = {
      "col_shard": [],
      "row_shard": [],
      "row_shard_embed": [],
      "kv_replicated": [],
      "replicated": [],
  }
  shardings: Dict[str, NamedSharding] = {}

  def _assign(name: str, spec: P, bucket: str) -> None:
    shardings[name] = NamedSharding(mesh, spec)
    buckets[bucket].append(name)

  for name in param_names:
    if any(s in name for s in ("k_proj", "v_proj")):
      if kv_head_shardable:
        _assign(name, P(AXIS_TP, None), "col_shard")
      else:
        _assign(name, P(None, None), "kv_replicated")
      continue
    if any(s in name for s in _COL_SHARD_SUBSTR):
      _assign(name, P(AXIS_TP, None), "col_shard")
      continue
    if any(s in name for s in _ROW_SHARD_SUBSTR):
      _assign(name, P(None, AXIS_TP), "row_shard")
      continue
    if any(s in name for s in _ROW_SHARD_EMBED):
      _assign(name, P(AXIS_TP, None), "row_shard_embed")
      continue
    _assign(name, P(None), "replicated")

  return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Dispatch: inspect the mesh, pick the strategy.
# -----------------------------------------------------------------------------


def get_param_sharding(model: Any, mesh: Mesh) -> ShardingPlan:
  """Build a ShardingPlan for a loaded HF model, picking the strategy from

  the mesh's axis names (FSDP if the mesh carries a ``'fsdp'`` axis, TP
  otherwise).
  """
  state = model.state_dict()
  names = list(state.keys())

  if _is_fsdp_mesh(mesh):
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    return plan_fsdp_shardings(names, shapes, mesh)

  # TP strategy — needs head counts.
  cfg = getattr(model, "config", None)
  if cfg is None:
    raise ValueError(
        "Model has no .config attribute; cannot infer head counts."
    )
  text_cfg = getattr(cfg, "text_config", cfg)
  num_kv = getattr(text_cfg, "num_key_value_heads", None)
  num_q = getattr(text_cfg, "num_attention_heads", None)
  if num_kv is None or num_q is None:
    raise ValueError(
        "Model config missing num_key_value_heads / num_attention_heads. "
        f"Got: text_cfg={text_cfg!r}"
    )
  return plan_tp_shardings(
      names,
      mesh,
      num_kv_heads=num_kv,
      num_attention_heads=num_q,
  )


# Back-compat alias.
plan_shardings = plan_tp_shardings


# -----------------------------------------------------------------------------
# Activation-sharding helpers — called inside the compiled train step.
# -----------------------------------------------------------------------------


def input_sharding(mesh: Mesh) -> NamedSharding:
  """``input_ids`` / ``labels`` sharding: shard batch dim on the data axis.

  - FSDP mesh: batch shards on ``'fsdp'`` (each chip sees batch/fsdp rows).
  - TP mesh:  batch shards on ``'dp'``.
  """
  if _is_fsdp_mesh(mesh):
    return NamedSharding(mesh, P(AXIS_FSDP, None))
  return NamedSharding(mesh, P(AXIS_DP, None))


def logits_sharding(mesh: Mesh) -> NamedSharding:
  """Logits sharding after lm_head.

  - FSDP mesh: batch on ``'fsdp'``, vocab replicated (GSPMD will schedule).
  - TP mesh:  batch on ``'dp'``, vocab on ``'tp'``.
  """
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
    "plan_fsdp_shardings",
    "plan_tp_shardings",
    "plan_shardings",
    "input_sharding",
    "logits_sharding",
    "replicated",
]
