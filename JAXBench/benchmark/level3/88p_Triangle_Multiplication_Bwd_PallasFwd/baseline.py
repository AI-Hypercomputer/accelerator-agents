# Copyright 2026 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triangle multiplication backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

The op is the "outgoing edges" triangle multiplicative update of AlphaFold
(Jumper et al. 2021, Supplementary Algorithm 11) on a pair representation `x`
of shape `[N, N, C]`: layer norm, a gated projection to two `[H, N, N]` planes,
the triangle contraction `sum_k left[c, i, k] * right[c, j, k]`, a second layer
norm, an output projection and an output gate.

`forward_pallas(x, mask, projection_in_weights, gate_in_weights,
projection_out_weights, gate_out_weights, layernorm_in_scale,
layernorm_in_offset, layernorm_out_scale, layernorm_out_offset)` is the Tokamax
Pallas TPU forward pass: a fused prologue kernel (input layer norm, gated
projection, mask), an XLA batched matmul for the triangle contraction, and a
fused epilogue kernel (output layer norm, output projection, output gate). It
returns `(out, residuals)`, where `residuals = (left, right, t3)` are the
intermediates the Tokamax forward saves for a backward pass, all in the dtype
of `x`:
  left   `[H, N, N]`  masked left projection, indexed `[c, i, k]`
  right  `[H, N, N]`  masked right projection, indexed `[c, j, k]`
  t3     `[H, N, N]`  triangle contraction, indexed `[c, i, j]`, before the
                      output layer norm
The kernels accept bfloat16 arrays only, with `C`, `H` and `D` in
`{16, 32, 64, 128}`; `N` is zero-padded to a multiple of 128 internally.

`workload(x, mask, projection_in_weights, gate_in_weights,
projection_out_weights, gate_out_weights, layernorm_in_scale,
layernorm_in_offset, layernorm_out_scale, layernorm_out_offset, dout)` returns
the gradients of the forward pass with respect to `x` and the eight weight
arrays (nine arrays, in argument order) for the output cotangent `dout`. The
boolean `mask` is not differentiated. It is the correctness oracle: JAX
autodiff (`jax.vjp`) of the pure-JAX forward reference `forward`. This file
contains no backward kernel and no hand-derived backward pass.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu.py
                                             (forward pass around the kernels)
  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_kernel.py
                                             (prologue and epilogue kernels)
  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_common.py
                                             (kernel helpers)
  tokamax/_src/ops/triangle_multiplication/base.py       (forward)
  tokamax/_src/ops/normalization/base.py                 (layer norm)
  tokamax/_src/ops/gated_linear_unit/base.py             (gated linear unit)
  tokamax/_src/ops/triangle_multiplication/arg_specs.py  (shapes)
  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_test.py
                                                         (value ranges)
"""

# Dimensions.
#   N      number of residues (the pair representation is N x N)
#   C      input channels
#   H      hidden channels of the triangle contraction
#   D      output channels
#   N_pad  N rounded up to a multiple of 128
#   Bi     tile extent along the first pair axis `i` (`block_i`, divides N_pad)
#   Bj     tile extent along the second pair axis `j` (`block_j`, a multiple
#          of 128 that divides N_pad)

import dataclasses
import json
import math
from typing import Any, Callable, Final, Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/triangle_multiplication/base.py,
#  tokamax/_src/ops/normalization/base.py and
#  tokamax/_src/ops/gated_linear_unit/base.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

# Options of the Tokamax op, fixed to the configuration benchmarked here.
_TRIANGLE_TYPE = 'outgoing'  # Algorithm 11; 'incoming' is Algorithm 12.
_EPSILON = 1e-6  # Tokamax default.
_PRECISION = None  # Tokamax default: the backend's default matmul precision.


def _layer_norm(x, scale, offset):
  """Layer norm over the last axis, computed in at least float32.

  From `Normalization._fwd` in tokamax/_src/ops/normalization/base.py with
  `axis=-1`, `subtract_mean=True` and `scale_offset=0.0`.
  """
  axis = x.ndim - 1
  # Always calculate in at least `float32` precision.
  y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
  mean = jnp.mean(y, axis, keepdims=True)
  y -= mean

  var = jnp.mean(jnp.square(y), axis, keepdims=True)
  rstddev = jax.lax.rsqrt(var + _EPSILON)
  y *= rstddev

  param_bcast_axes = [a for a in range(x.ndim) if a != axis]
  scale = scale.astype(y.dtype)
  y *= jnp.expand_dims(scale, param_bcast_axes)
  offset = offset.astype(y.dtype)
  y += jnp.expand_dims(offset, param_bcast_axes)

  return y.astype(x.dtype)


def _gated_linear_unit(x, gate_weights, projection_weights):
  """Computes `sigmoid(x @ gate_weights) * (x @ projection_weights)`.

  From `GatedLinearUnit._fwd` in tokamax/_src/ops/gated_linear_unit/base.py
  with unfused weights and `activation=jax.nn.sigmoid`.
  """
  projection = jnp.matmul(x, projection_weights, precision=_PRECISION)
  gate = jnp.matmul(x, gate_weights, precision=_PRECISION)
  # Apply activation and compute product of FP8/FP16/BF16 in FP32.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  gate = gate.astype(dtype)
  projection = projection.astype(dtype)
  gate = jax.nn.sigmoid(gate)
  return (gate * projection).astype(x.dtype)


def forward(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
):
  """Triangle multiplicative update ("outgoing edges").

  Implements Supplementary Algorithm 11 of 'Highly accurate protein structure
  prediction with AlphaFold', Jumper et. al. 2021.

  Args:
    x: The input array of shape `[N, N, C]`.
    mask: A boolean mask of shape `[N, N]`.
    projection_in_weights: Weights for projection layer `[C, 2, H]`.
    gate_in_weights: Weights for gate layer `[C, 2, H]`.
    projection_out_weights: Weights for the output projection layer `[H, D]`.
    gate_out_weights: Weights for the output gate layer `[C, D]`.
    layernorm_in_scale: Scale for the input layer normalization `[C]`.
    layernorm_in_offset: Offset for the input layer normalization `[C]`.
    layernorm_out_scale: Scale for the output layer normalization `[H]`.
    layernorm_out_offset: Offset for the output layer normalization `[H]`.

  Returns:
    The result of triangle multiplication of shape `[N, N, D]`, in the dtype of
    `x`.
  """
  h_dim = layernorm_out_scale.shape[0]

  left_act = _layer_norm(x, layernorm_in_scale, layernorm_in_offset)

  channel_dim = len(gate_in_weights)
  gate_weights = jnp.reshape(gate_in_weights, (channel_dim, -1))  # (C, 2 * H)
  projection_weights = jnp.reshape(projection_in_weights, (channel_dim, -1))
  proj_act = _gated_linear_unit(left_act, gate_weights, projection_weights)
  proj_act = proj_act.reshape(proj_act.shape[0], proj_act.shape[1], 2, h_dim)
  proj_act = jnp.transpose(proj_act, (2, 3, 0, 1))  # 2 C N N
  proj_act = mask * proj_act
  left_proj_act, right_proj_act = jnp.split(proj_act, 2, axis=0)
  left_proj_act = jnp.squeeze(left_proj_act, axis=0)
  right_proj_act = jnp.squeeze(right_proj_act, axis=0)

  equation = 'cik,cjk->ijc' if _TRIANGLE_TYPE == 'outgoing' else 'ckj,cki->ijc'
  act = jnp.einsum(
      equation, left_proj_act, right_proj_act, precision=_PRECISION
  )

  act = _layer_norm(act, layernorm_out_scale, layernorm_out_offset)

  act = jnp.dot(act, projection_out_weights, precision=_PRECISION)

  gate_values = jnp.dot(left_act, gate_out_weights, precision=_PRECISION)
  act *= jax.nn.sigmoid(gate_values)

  return act


# ==============================================================================
# Pallas TPU forward kernel: shared helpers
# (inlined from
#  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_common.py,
#  Copyright 2026 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset(
    {jnp.dtype(jnp.bfloat16)}
)
SUPPORTED_CHANNEL_DIMS: Final[frozenset[int]] = frozenset({16, 32, 64, 128})
MAX_INT32: Final[int] = 2**31 - 1


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Configuration for Pallas-Mosaic-TPU Triangle Multiplication kernels.

  Attributes:
    block_m: Tile size control parameter in [128, 1024] (power of two). Mapped
      to 3D `(block_i, block_j)` tile dimensions dividing `N_pad`.
    epilogue_block_m: Optional tile size override for the epilogue kernel.
    block_i: Optional explicit tile height override along axis `i`.
    block_j: Optional explicit tile width override along axis `j`.
    epilogue_block_i: Optional explicit `block_i` override for the epilogue.
    epilogue_block_j: Optional explicit `block_j` override for the epilogue.
  """

  block_m: int = 512
  epilogue_block_m: int | None = None
  block_i: int | None = None
  block_j: int | None = None
  epilogue_block_i: int | None = None
  epilogue_block_j: int | None = None


def _infer_explicit_device(x: Any) -> jax.Device | None:
  """Returns concrete device of `x` if explicitly placed, else `None`."""
  sharding = getattr(x, "sharding", None)
  if sharding is not None:
    device_set = getattr(sharding, "device_set", None)
    if device_set:
      return next(iter(device_set))
  if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
    try:
      return next(iter(x.devices()), None)
    except jax.errors.ConcretizationTypeError:
      return None
  return None


def infer_tpu_device(x: Any = None) -> jax.Device | None:
  """Infers target TPU `jax.Device` from `x`, default device, or local TPUs."""
  if x is not None:
    explicit_dev = _infer_explicit_device(x)
    if explicit_dev is not None:
      return explicit_dev if explicit_dev.platform == "tpu" else None
  default_dev = backend.get_default_device()
  if default_dev.platform == "tpu":
    return default_dev
  try:
    tpu_devs = jax.devices("tpu")
    return tpu_devs[0] if tpu_devs else None
  except RuntimeError:
    return None


def get_tpu_info(
    x: Any = None,
) -> tuple[int, int, int, int | None, int | None]:
  """Returns `(gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes)`."""
  tpu_dev = infer_tpu_device(x)
  try:
    if (
        isinstance(tpu_dev, jax.Device)
        and backend.get_default_device() != tpu_dev
    ):
      with jax.default_device(tpu_dev):
        info = pltpu.get_tpu_info()
    else:
      info = pltpu.get_tpu_info()
    return (
        info.generation,
        info.vmem_capacity_bytes // (1024 * 1024),
        info.num_cores,
        int(0.90 * info.vmem_capacity_bytes),
        int(0.80 * info.vmem_capacity_bytes),
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return 6, 64, 1, None, 50 * 1024 * 1024


def validate_fwd_inputs(
    x: jax.Array,
    mask: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    triangle_type: str,
    precision: Any,
    epsilon: float,
) -> None:
  """Validates inputs for Pallas Mosaic TPU forward pass."""
  if triangle_type not in ("incoming", "outgoing"):
    raise NotImplementedError(
        f"Unsupported triangle_type={triangle_type!r}; expected 'incoming' or"
        " 'outgoing'."
    )

  if (
      isinstance(epsilon, bool)
      or not isinstance(epsilon, (int, float))
      or not math.isfinite(epsilon)
      or epsilon < 0.0
  ):
    raise NotImplementedError(
        f"epsilon must be a finite non-negative float, got {epsilon!r}."
    )

  all_arrays = (
      x,
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  for arr in all_arrays:
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
      raise NotImplementedError(f"Expected array input, got {type(arr)}.")
    if not all(isinstance(dim, int) for dim in arr.shape):
      raise NotImplementedError("Mosaic TPU does not support symbolic shapes.")

  if precision not in (
      (lax.Precision.DEFAULT, lax.Precision.DEFAULT),
      lax.DotAlgorithmPreset.DEFAULT,
      lax.DotAlgorithmPreset.BF16_BF16_F32,
  ):
    raise NotImplementedError(f"{precision=} is not supported.")

  if jnp.dtype(x.dtype) not in SUPPORTED_DTYPES:
    raise NotImplementedError(
        "Mosaic TPU triangle_multiplication only supports"
        f" {SUPPORTED_DTYPES}, got {x.dtype}."
    )

  for w in (
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  ):
    if jnp.dtype(w.dtype) != jnp.dtype(x.dtype):
      raise NotImplementedError(
          f"All weight and norm arrays must match x.dtype={x.dtype}, got"
          f" {w.dtype}."
      )

  if x.ndim != 3 or x.shape[0] != x.shape[1]:
    raise NotImplementedError(f"Expected x of shape (N, N, C), got {x.shape}.")
  n, _, c = x.shape

  if mask.shape != (n, n) or jnp.dtype(mask.dtype) != jnp.dtype(jnp.bool_):
    raise NotImplementedError(
        f"Expected bool mask of shape {(n, n)}, got shape={mask.shape},"
        f" dtype={mask.dtype}."
    )

  if projection_out_weights.ndim != 2:
    raise NotImplementedError(
        "Expected 2D projection_out_weights (H, D), got"
        f" {projection_out_weights.shape}."
    )
  h, d = projection_out_weights.shape

  if not {c, h, d} <= SUPPORTED_CHANNEL_DIMS:
    raise NotImplementedError(
        "Mosaic TPU TriangleMultiplication requires C, H, D in"
        f" {sorted(SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
    )

  expected_shapes = (
      ("projection_in_weights", projection_in_weights.shape, (c, 2, h)),
      ("gate_in_weights", gate_in_weights.shape, (c, 2, h)),
      ("gate_out_weights", gate_out_weights.shape, (c, d)),
      ("layernorm_in_scale", layernorm_in_scale.shape, (c,)),
      ("layernorm_in_offset", layernorm_in_offset.shape, (c,)),
      ("layernorm_out_scale", layernorm_out_scale.shape, (h,)),
      ("layernorm_out_offset", layernorm_out_offset.shape, (h,)),
  )
  for name, got_shape, exp_shape in expected_shapes:
    if got_shape != exp_shape:
      raise NotImplementedError(
          f"Expected {name} shape {exp_shape}, got {got_shape}."
      )

  if n == 0:
    raise NotImplementedError("Empty inputs are not supported.")
  n_pad = pl.cdiv(n, 128) * 128
  if n_pad * n_pad * max(c, h, d) - 1 > MAX_INT32:
    raise NotImplementedError(
        f"N={n} (padded to {n_pad}) with max(C, H, D)={max(c, h, d)} would"
        " overflow the int32 offsets used by the Mosaic TPU kernels."
    )


def contract(
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: Any,
) -> jax.Array:
  """Batched (over H) triangle contraction of two (H, N, N) planes."""
  if triangle_type == "outgoing":
    dims = (((2,), (2,)), ((0,), (0,)))
    return lax.dot_general(left, right, dims, precision=precision)
  else:
    dims = (((1,), (1,)), ((0,), (0,)))
    return lax.dot_general(right, left, dims, precision=precision)


def pad_dim(x: jax.Array, target: int, axis: int = 0) -> jax.Array:
  """Zero-pads `x` along `axis` to `target` if `x.shape[axis] < target`."""
  cur = x.shape[axis]
  if cur >= target:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, target - cur)
  return jnp.pad(x, pad_width)


def transpose_and_pad_icj(x: jax.Array, n_pad: int) -> jax.Array:
  """Transposes `(N, N, K)` -> `(N_pad, K, N_pad)` for 128-lane `Bj` kernels."""
  x_icj = jnp.transpose(x, (0, 2, 1))
  return pad_dim(pad_dim(x_icj, n_pad, 0), n_pad, 2)


def cast_and_pad_mask(
    mask: jax.Array, n_pad: int, dtype: Any, x: Any = None
) -> jax.Array:
  """Casts boolean `(N, N)` mask to `dtype` and pads to `(N_pad, N_pad)`."""
  gen, _, num_cores, _, _ = get_tpu_info(mask if x is None else x)
  if gen == 5 and num_cores > 1:
    m, k = mask.shape
    mask_bcast = jax.lax.optimization_barrier(
        lax.broadcast_in_dim(mask, (2, m, k), (1, 2))
    )
    mask_f = lax.index_in_dim(
        jax.lax.optimization_barrier(mask_bcast.astype(dtype)),
        0,
        axis=0,
        keepdims=False,
    )
  else:
    mask_f = mask.astype(dtype)
  return pad_dim(pad_dim(mask_f, n_pad, 0), n_pad, 1)


# ------------------------------------------------------------------------------
# Device-side helpers used inside Pallas Mosaic TPU kernels (128-lane dense).
# ------------------------------------------------------------------------------


def pack_icj_to_2d(x_ikj: jax.Array) -> jax.Array:
  """Packs `bf16`/`f32` `(Bi, K, Bj)` into `(K, Bi * Bj)` with zero VPU ops."""
  bi, k, bj = x_ikj.shape
  if bi == 1:
    return x_ikj.reshape(k, bj)
  return jnp.concatenate([x_ikj[i : i + 1] for i in range(bi)], axis=2).reshape(
      k, bi * bj
  )


def unpack_2d_to_icj(y_2d: jax.Array, bi: int, bj: int) -> jax.Array:
  """Unpacks `(M, Bi * Bj)` into `(Bi, M, Bj)` with zero VPU ops."""
  m = y_2d.shape[0]
  if bi == 1:
    return y_2d.reshape(1, m, bj)
  y_3d = y_2d.reshape(1, m, bi * bj)
  return jnp.concatenate(
      [y_3d[:, :, i * bj : (i + 1) * bj] for i in range(bi)], axis=0
  )


def matmul_2d(w_mk: jax.Array, x_kt: jax.Array) -> jax.Array:
  """Computes unbatched `(M, K) @ (K, T) -> (M, T)` in FP32 on the TPU MXU."""
  return lax.dot_general(
      w_mk,
      x_kt,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )


def layer_norm_3d_axis0_hat_rstd(
    x: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(x - mean) * rstd` and `rstd` along axis 0 of `(H, Bi, Bj)`."""
  h = x.shape[0]
  x_f32 = x.astype(jnp.float32)
  mean = jnp.sum(x_f32, axis=0, keepdims=True) * (1.0 / h)
  centered = x_f32 - mean
  var = jnp.sum(centered * centered, axis=0, keepdims=True) * (1.0 / h)
  var_eps = var if eps == 0.0 else var + jnp.asarray(eps, jnp.float32)
  rstd = jnp.where(var_eps == 0.0, 0.0, lax.rsqrt(var_eps))
  return centered * rstd, rstd


def apply_norm_affine_axis0(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along axis 0 of `(H, Bi, Bj)` in FP32."""
  scale_b = lax.broadcast_in_dim(scale.astype(jnp.float32), x_hat.shape, (0,))
  offset_b = lax.broadcast_in_dim(offset.astype(jnp.float32), x_hat.shape, (0,))
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1_hat_rstd(
    x: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(x - mean) * rstd` and `rstd` along axis 1 of `(Bi, K, Bj)`."""
  k = x.shape[1]
  x_f32 = x.astype(jnp.float32)
  mean = jnp.sum(x_f32, axis=1, keepdims=True) * (1.0 / k)
  centered = x_f32 - mean
  var = jnp.sum(centered * centered, axis=1, keepdims=True) * (1.0 / k)
  var_eps = var if eps == 0.0 else var + jnp.asarray(eps, jnp.float32)
  rstd = jnp.where(var_eps == 0.0, 0.0, lax.rsqrt(var_eps))
  return centered * rstd, rstd


def apply_norm_affine_axis1(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along sublane axis 1 of `(Bi, K, Bj)`."""
  scale_b = lax.broadcast_in_dim(scale.astype(jnp.float32), x_hat.shape, (1,))
  offset_b = lax.broadcast_in_dim(offset.astype(jnp.float32), x_hat.shape, (1,))
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1(
    x: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    eps: float,
) -> jax.Array:
  """Computes FP32 LayerNorm along sublane axis 1 of `(Bi, K, Bj)`."""
  x_hat, _ = layer_norm_3d_axis1_hat_rstd(x, eps)
  return apply_norm_affine_axis1(x_hat, scale, offset).astype(x.dtype)


# ==============================================================================
# Pallas TPU forward kernel: block-size heuristics, prologue and epilogue
# (inlined from
#  tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_kernel.py,
#  Copyright 2026 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def get_heuristics_config(x: jax.Array, w_out: jax.Array) -> Config:
  """Returns hardware-tailored forward config for TPU v4, v5p, v6e, and 7x."""
  n, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, num_cores, _, _ = get_tpu_info(x)

  if vmem_mib <= 16 or gen <= 4:
    return Config(block_m=256 if max_chd > 64 else 512)

  if max_chd > 64:
    return Config(block_m=512 if vmem_mib <= 64 else 1024)

  n_pad = pl.cdiv(n, 128) * 128
  if gen <= 6 and n_pad <= 384:
    return Config(block_m=512)
  if n_pad >= 512:
    if vmem_mib > 64 and n_pad % 2048 == 0:
      return Config(
          block_m=1024, epilogue_block_i=32, epilogue_block_j=2048
      )
    if vmem_mib > 64 and n_pad % 1024 == 0:
      return Config(
          block_m=1024,
          block_i=32,
          block_j=1024,
          epilogue_block_i=64,
          epilogue_block_j=1024,
      )
    if n_pad % 1024 == 0:
      return Config(
          block_m=1024, epilogue_block_i=32, epilogue_block_j=1024
      )
    if n_pad % 512 == 0:
      epi_bi = 64 if n_pad % 64 == 0 else 32
      if n_pad == 512 and num_cores > 1:
        epi_bi = min(epi_bi, n_pad // num_cores)
      return Config(
          block_m=1024, epilogue_block_i=epi_bi, epilogue_block_j=512
      )

  return Config(block_m=1024)


def _largest_aligned_divisor(n_pad: int, limit: int, align: int) -> int:
  """Returns largest multiple of `align` in `[align, min(limit, n_pad)]` dividing `n_pad`."""
  cand = (min(limit, n_pad) // align) * align
  while cand >= align:
    if n_pad % cand == 0:
      return cand
    cand -= align
  return align


def _default_blocks(
    n_pad: int, block_m: int, vmem_mib: int, num_cores: int = 1
) -> tuple[int, int]:
  """Hardware-tiered default `(block_i, block_j)` for `block_m`."""
  min_bi = 8 if n_pad < 16 else 16
  if block_m <= 128:
    return (8 if vmem_mib <= 16 or n_pad < 16 else 16), 128

  if block_m == 256:
    if vmem_mib <= 16:
      return (8, 256) if n_pad % 256 == 0 else (8, 128)
    if n_pad % 256 == 0:
      return min_bi, 256
    return _largest_aligned_divisor(n_pad, 32, min_bi), 128

  if block_m == 512:
    if vmem_mib <= 16:
      return min_bi, 128
    for cand_j in (640, 512, 384, 256):
      if n_pad % cand_j == 0:
        return min_bi, cand_j
    max_bi = max(min_bi, n_pad // num_cores) if num_cores > 1 else n_pad
    return _largest_aligned_divisor(n_pad, min(64, max_bi), min_bi), 128

  if vmem_mib <= 16:
    return min_bi, 128

  p_target = 16384
  top_cands = (
      (2048, 1024, 896, 768, 640, 512, 384, 256, 128)
      if vmem_mib > 64
      else (1024, 896, 768, 640, 512, 384, 256, 128)
  )
  for cand_j in top_cands:
    if n_pad % cand_j == 0:
      max_bi = min(64, max(min_bi, p_target // cand_j))
      if cand_j == n_pad and num_cores > 1 and n_pad >= min_bi * num_cores:
        max_bi = min(max_bi, n_pad // num_cores)
      return _largest_aligned_divisor(n_pad, max_bi, min_bi), cand_j
  return min_bi, 128


def resolve_blocks(
    n_pad: int,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
    *,
    x: Any = None,
) -> tuple[int, int]:
  """Maps `block_m` and optional independent axis overrides to `(block_i, block_j)`."""
  _, vmem_mib, num_cores, _, _ = get_tpu_info(x)
  def_bi, def_bj = _default_blocks(n_pad, block_m, vmem_mib, num_cores)
  bi = (
      _largest_aligned_divisor(n_pad, block_i, 8)
      if block_i is not None
      else def_bi
  )
  bj = (
      _largest_aligned_divisor(n_pad, block_j, 128)
      if block_j is not None
      else def_bj
  )
  return bi, bj


def _estimate_prologue_vmem(bi: int, bj: int, c: int, h: int) -> int:
  """Estimates live VMEM bytes for prologue kernel."""
  per_elem = max(8 * c + 8 * h + 4, 6 * c + 24 * h + 4)
  return bi * bj * per_elem + 8 * h * c + 64 * c + 512 * 1024


def _estimate_epilogue_vmem(bi: int, bj: int, c: int, h: int, d: int) -> int:
  """Estimates live VMEM bytes for epilogue kernel."""
  if c + h <= 128 and 2 * d <= 256:
    scratch = max(4 * c + 4 * h, 2 * c + 2 * h + 8 * d)
    w_bytes = 4 * d * (c + h)
  else:
    scratch = max(
        4 * c, 2 * c + 4 * h, 2 * c + 2 * h + 4 * d, 2 * h + 8 * d
    )
    w_bytes = 2 * d * (c + h)
  per_elem = 4 * c + 4 * h + 4 * d + scratch
  return bi * bj * per_elem + w_bytes + 64 * (c + h) + 512 * 1024


def _clamp_blocks_to_vmem(
    n_pad: int,
    bi: int,
    bj: int,
    vmem_budget_bytes: int | None,
    estimate_fn: Callable[[int, int], int],
) -> tuple[int, int]:
  """Steps `(bi, bj)` down through valid divisors of `n_pad` until under VMEM budget."""
  if vmem_budget_bytes is None or estimate_fn(bi, bj) <= vmem_budget_bytes:
    return bi, bj
  bi_align = 16 if (bi % 16 == 0 and n_pad >= 16) else 8
  while bi > bi_align and estimate_fn(bi, bj) > vmem_budget_bytes:
    bi = _largest_aligned_divisor(n_pad, bi - bi_align, bi_align)
  while bj > 128 and estimate_fn(bi, bj) > vmem_budget_bytes:
    bj = _largest_aligned_divisor(n_pad, bj - 128, 128)
  while (
      bi > 8 and n_pad % 8 == 0 and estimate_fn(bi, bj) > vmem_budget_bytes
  ):
    bi = _largest_aligned_divisor(n_pad, bi - 8, 8)
  return bi, bj


def _scoped_vmem_limit(
    gen: int,
    vmem_mib: int,
    est_vmem_bytes: int,
    vmem_limit_bytes: int | None,
    *,
    bi: int = 64,
    bj: int = 1024,
    max_dim: int = 128,
) -> int | None:
  """Leaves VMEM unreserved for XLA MSA when tiles fit inside default VMEM."""
  if (
      gen >= 6
      and vmem_mib > 16
      and bj <= 512
      and max_dim <= 64
      and bi * bj <= 16384
  ):
    if est_vmem_bytes <= 30 * 1024 * 1024:
      return None
    if est_vmem_bytes <= 38 * 1024 * 1024:
      return 44 * 1024 * 1024
  return vmem_limit_bytes


def _prologue_cost_estimate(n_pad: int, c: int, h: int) -> pl.CostEstimate:
  m = n_pad * n_pad
  return pl.CostEstimate(
      flops=m * (8 * c * h + 7 * c + 4 * h),
      transcendentals=m * (2 * h + 1),
      bytes_accessed=2 * m * (c + 2 * h + 1) + 8 * c * h + 64 * c,
  )


def _epilogue_cost_estimate(
    n_pad: int, c: int, h: int, d: int
) -> pl.CostEstimate:
  m = n_pad * n_pad
  use_fused_gp = c + h <= 128 and 2 * d <= 256
  w_elems = 2 * d * (c + h) if use_fused_gp else d * (c + h)
  return pl.CostEstimate(
      flops=m * (2 * w_elems + 7 * (c + h) + 2 * d),
      transcendentals=m * (d + 2),
      bytes_accessed=2 * m * (c + h + d) + 2 * w_elems + 64 * (c + h),
  )


def _bcast_norm_2d(v: jax.Array) -> jax.Array:
  """Broadcasts 1D `(K,)` norm param to 2D `(16, K)` for vmap & TPU 7x DMA."""
  return lax.broadcast_in_dim(v, (16, v.shape[0]), (1,))


def prologue(
    x_pad: jax.Array,
    mask_pad: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Fused Pallas Mosaic TPU prologue kernel."""
  n_pad, c, _ = x_pad.shape
  h = projection_in_weights.shape[2]
  dtype = x_pad.dtype
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  est_fn = lambda bi, bj: _estimate_prologue_vmem(bi, bj, c, h)
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h),
  )

  use_4d_t8 = gen >= 7
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
  else:
    x_in = x_pad
    x_spec = pl.BlockSpec((block_i, c, block_j), lambda i, j: (i, 0, j))

  if use_4d_hij:
    mask_in = mask_pad.reshape(n_pad // 8, 8, n_pad)
    mask_spec = pl.BlockSpec(
        (block_i // 8, 8, block_j), lambda i, j: (i, 0, j)
    )
    out_shape = jax.ShapeDtypeStruct((h, n_pad // 8, 8, n_pad), dtype)
    out_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j), lambda i, j: (0, i, 0, j)
    )
  else:
    mask_in = mask_pad
    mask_spec = pl.BlockSpec((block_i, block_j), lambda i, j: (i, j))
    out_shape = jax.ShapeDtypeStruct((h, n_pad, n_pad), dtype)
    out_spec = pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j))

  w_proj_t = projection_in_weights.reshape(c, 2 * h).T
  w_gate_t = gate_in_weights.reshape(c, 2 * h).T
  ln_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_c_spec = pl.BlockSpec((16, c), lambda i, j: (0, 0))

  if 4 * h <= 256:
    w_pg_arg = jnp.concatenate([w_proj_t, w_gate_t], axis=0)

    def kernel_4h(
        x_ref,
        mask_ref,
        w_pg_ref,
        ln_scale_ref,
        ln_offset_ref,
        left_ref,
        right_ref,
    ):
      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      mask_2d = (
          mask_ref[...].reshape(block_i, block_j)
          if use_4d_hij
          else mask_ref[...]
      )
      act = layer_norm_3d_axis1(
          x_raw, ln_scale_ref[0], ln_offset_ref[0], eps
      )
      pg_2d = matmul_2d(w_pg_ref[...], pack_icj_to_2d(act))
      lr_2d = (pg_2d[: 2 * h] * jax.nn.sigmoid(pg_2d[2 * h :])).astype(dtype)
      lr_hij = jnp.transpose(
          unpack_2d_to_icj(lr_2d, block_i, block_j), (1, 0, 2)
      )
      lr_hij = jnp.where(
          lax.broadcast_in_dim(mask_2d == 0, lr_hij.shape, (1, 2)),
          0,
          lr_hij,
      )
      left_val = lr_hij[:h]
      right_val = lr_hij[h:]
      left_ref[...] = (
          left_val.reshape(h, block_i // 8, 8, block_j)
          if use_4d_hij
          else left_val
      )
      right_ref[...] = (
          right_val.reshape(h, block_i // 8, 8, block_j)
          if use_4d_hij
          else right_val
      )

    left_out, right_out = pl.pallas_call(
        kernel_4h,
        out_shape=(out_shape, out_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_i, num_j),
            in_specs=[
                x_spec,
                mask_spec,
                pl.BlockSpec(w_pg_arg.shape, lambda i, j: (0, 0)),
                ln_c_spec,
                ln_c_spec,
            ],
            out_specs=(out_spec, out_spec),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(par_sem, par_sem),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        cost_estimate=(
            None
            if (gen == 5 and num_cores > 1)
            else _prologue_cost_estimate(n_pad, c, h)
        ),
        interpret=jax.default_backend() != "tpu",
    )(
        x_in,
        mask_in,
        w_pg_arg,
        ln_scale_in,
        ln_offset_in,
    )
    if use_4d_hij:
      return (
          left_out.reshape(h, n_pad, n_pad),
          right_out.reshape(h, n_pad, n_pad),
      )
    return left_out, right_out

  def kernel_2h(
      x_ref,
      mask_ref,
      w_proj_ref,
      w_gate_ref,
      ln_scale_ref,
      ln_offset_ref,
      left_ref,
      right_ref,
  ):
    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    mask_2d = (
        mask_ref[...].reshape(block_i, block_j) if use_4d_hij else mask_ref[...]
    )
    act = layer_norm_3d_axis1(
        x_raw, ln_scale_ref[0], ln_offset_ref[0], eps
    )
    act_2d = pack_icj_to_2d(act)
    p_2d = matmul_2d(w_proj_ref[...], act_2d)
    g_2d = matmul_2d(w_gate_ref[...], act_2d)
    lr_2d = (p_2d * jax.nn.sigmoid(g_2d)).astype(dtype)
    lr_hij = jnp.transpose(
        unpack_2d_to_icj(lr_2d, block_i, block_j), (1, 0, 2)
    )
    lr_hij = jnp.where(
        lax.broadcast_in_dim(mask_2d == 0, lr_hij.shape, (1, 2)),
        0,
        lr_hij,
    )
    left_val = lr_hij[:h]
    right_val = lr_hij[h:]
    left_ref[...] = (
        left_val.reshape(h, block_i // 8, 8, block_j)
        if use_4d_hij
        else left_val
    )
    right_ref[...] = (
        right_val.reshape(h, block_i // 8, 8, block_j)
        if use_4d_hij
        else right_val
    )

  left_out, right_out = pl.pallas_call(
      kernel_2h,
      out_shape=(out_shape, out_shape),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_i, num_j),
          in_specs=[
              x_spec,
              mask_spec,
              pl.BlockSpec(w_proj_t.shape, lambda i, j: (0, 0)),
              pl.BlockSpec(w_gate_t.shape, lambda i, j: (0, 0)),
              ln_c_spec,
              ln_c_spec,
          ],
          out_specs=(out_spec, out_spec),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
      cost_estimate=(
          None
          if (gen == 5 and num_cores > 1)
          else _prologue_cost_estimate(n_pad, c, h)
      ),
      interpret=jax.default_backend() != "tpu",
  )(
      x_in,
      mask_in,
      w_proj_t,
      w_gate_t,
      ln_scale_in,
      ln_offset_in,
  )
  if use_4d_hij:
    return (
        left_out.reshape(h, n_pad, n_pad),
        right_out.reshape(h, n_pad, n_pad),
    )
  return left_out, right_out


def epilogue(
    t3_pad: jax.Array,
    x_pad: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> jax.Array:
  """Fused Pallas Mosaic TPU epilogue kernel."""
  n_pad, c, _ = x_pad.shape
  h, d = projection_out_weights.shape
  dtype = x_pad.dtype
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  est_fn = lambda bi, bj: _estimate_epilogue_vmem(bi, bj, c, h, d)
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h, d),
  )

  use_4d_t8 = gen >= 7
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
    out_shape = jax.ShapeDtypeStruct((n_pad, d // 8, 8, n_pad), dtype)
    out_spec = pl.BlockSpec(
        (block_i, d // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
  else:
    x_in = x_pad
    x_spec = pl.BlockSpec((block_i, c, block_j), lambda i, j: (i, 0, j))
    out_shape = jax.ShapeDtypeStruct((n_pad, d, n_pad), dtype)
    out_spec = pl.BlockSpec((block_i, d, block_j), lambda i, j: (i, 0, j))

  if use_4d_hij:
    t3_in = t3_pad.reshape(h, n_pad // 8, 8, n_pad)
    t3_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j), lambda i, j: (0, i, 0, j)
    )
  else:
    t3_in = t3_pad
    t3_spec = pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j))

  ln_in_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_in_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_out_scale_in = _bcast_norm_2d(layernorm_out_scale)
  ln_out_offset_in = _bcast_norm_2d(layernorm_out_offset)
  epi_cost = (
      None
      if (gen == 5 and num_cores > 1)
      else _epilogue_cost_estimate(n_pad, c, h, d)
  )

  use_fused_gp = c + h <= 128 and 2 * d <= 256
  if use_fused_gp:
    w_gp_out_t = _build_block_diag_out_weights_t(
        gate_out_weights, projection_out_weights
    )

    def kernel_gp(
        t3_ref,
        x_ref,
        w_gp_ref,
        ln_in_scale_ref,
        ln_in_offset_ref,
        ln_out_scale_ref,
        ln_out_offset_ref,
        out_ref,
    ):
      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      t3_raw = (
          t3_ref[...].reshape(h, block_i, block_j)
          if use_4d_hij
          else t3_ref[...]
      )
      left_act = layer_norm_3d_axis1(
          x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
      )
      t_hat_3d, _ = layer_norm_3d_axis0_hat_rstd(t3_raw, eps)
      act_out = jnp.transpose(
          apply_norm_affine_axis0(
              t_hat_3d, ln_out_scale_ref[0], ln_out_offset_ref[0]
          ).astype(dtype),
          (1, 0, 2),
      )
      la_ao_2d = jnp.concatenate(
          [pack_icj_to_2d(left_act), pack_icj_to_2d(act_out)],
          axis=0,
      )
      gp_2d = matmul_2d(w_gp_ref[...], la_ao_2d)
      out_2d = (gp_2d[d:, :] * jax.nn.sigmoid(gp_2d[:d, :])).astype(dtype)
      out_idj = unpack_2d_to_icj(out_2d, block_i, block_j)
      out_ref[...] = (
          out_idj.reshape(block_i, d // 8, 8, block_j)
          if use_4d_t8
          else out_idj
      )

    out = pl.pallas_call(
        kernel_gp,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_i, num_j),
            in_specs=[
                t3_spec,
                x_spec,
                pl.BlockSpec((2 * d, c + h), lambda i, j: (0, 0)),
                pl.BlockSpec((16, c), lambda i, j: (0, 0)),
                pl.BlockSpec((16, c), lambda i, j: (0, 0)),
                pl.BlockSpec((16, h), lambda i, j: (0, 0)),
                pl.BlockSpec((16, h), lambda i, j: (0, 0)),
            ],
            out_specs=out_spec,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(par_sem, par_sem),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        cost_estimate=epi_cost,
        interpret=jax.default_backend() != "tpu",
    )(
        t3_in,
        x_in,
        w_gp_out_t,
        ln_in_scale_in,
        ln_in_offset_in,
        ln_out_scale_in,
        ln_out_offset_in,
    )
    return out.reshape(n_pad, d, n_pad) if use_4d_t8 else out

  w_gate_t = gate_out_weights.T
  w_proj_t = projection_out_weights.T

  def kernel_3d(
      t3_ref,
      x_ref,
      w_gate_ref,
      w_proj_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      ln_out_scale_ref,
      ln_out_offset_ref,
      out_ref,
  ):
    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    t3_raw = (
        t3_ref[...].reshape(h, block_i, block_j) if use_4d_hij else t3_ref[...]
    )
    left_act = layer_norm_3d_axis1(
        x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
    )
    t_hat_3d, _ = layer_norm_3d_axis0_hat_rstd(t3_raw, eps)
    act_out = jnp.transpose(
        apply_norm_affine_axis0(
            t_hat_3d, ln_out_scale_ref[0], ln_out_offset_ref[0]
        ).astype(dtype),
        (1, 0, 2),
    )
    g_2d = matmul_2d(w_gate_ref[...], pack_icj_to_2d(left_act))
    p_2d = matmul_2d(w_proj_ref[...], pack_icj_to_2d(act_out))
    out_2d = (p_2d * jax.nn.sigmoid(g_2d)).astype(dtype)
    out_idj = unpack_2d_to_icj(out_2d, block_i, block_j)
    out_ref[...] = (
        out_idj.reshape(block_i, d // 8, 8, block_j) if use_4d_t8 else out_idj
    )

  out = pl.pallas_call(
      kernel_3d,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_i, num_j),
          in_specs=[
              t3_spec,
              x_spec,
              pl.BlockSpec((d, c), lambda i, j: (0, 0)),
              pl.BlockSpec((d, h), lambda i, j: (0, 0)),
              pl.BlockSpec((16, c), lambda i, j: (0, 0)),
              pl.BlockSpec((16, c), lambda i, j: (0, 0)),
              pl.BlockSpec((16, h), lambda i, j: (0, 0)),
              pl.BlockSpec((16, h), lambda i, j: (0, 0)),
          ],
          out_specs=out_spec,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
      cost_estimate=epi_cost,
      interpret=jax.default_backend() != "tpu",
  )(
      t3_in,
      x_in,
      w_gate_t,
      w_proj_t,
      ln_in_scale_in,
      ln_in_offset_in,
      ln_out_scale_in,
      ln_out_offset_in,
  )
  return out.reshape(n_pad, d, n_pad) if use_4d_t8 else out


def _build_block_diag_out_weights_t(
    gate_out_weights: jax.Array,
    projection_out_weights: jax.Array,
) -> jax.Array:
  """Builds `(2 * D, C + H)` block-diagonal `[[W_gate^T, 0], [0, W_proj^T]]`."""
  c, d = gate_out_weights.shape
  h, _ = projection_out_weights.shape
  dtype = gate_out_weights.dtype
  top = jnp.concatenate(
      [gate_out_weights.T, jnp.zeros((d, h), dtype=dtype)], axis=1
  )
  bot = jnp.concatenate(
      [jnp.zeros((d, c), dtype=dtype), projection_out_weights.T], axis=1
  )
  return jnp.concatenate([top, bot], axis=0)


# ==============================================================================
# Pallas TPU forward pass
# (inlined from tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu.py,
#  `PallasMosaicTpuTriangleMultiplication._fwd`,
#  Copyright 2026 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def triangle_multiplication_pallas_fwd(
    x: jax.Array,
    mask: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    *,
    precision: Any,
    epsilon: float,
    return_residuals: bool,
    config: Config,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array] | None]:
  """Fused prologue and epilogue kernels around an XLA batched matmul."""
  validate_fwd_inputs(
      x,
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      triangle_type,
      precision,
      epsilon,
  )

  n, _, _ = x.shape
  _, d = projection_out_weights.shape
  dtype = x.dtype
  n_pad = pl.cdiv(n, 128) * 128

  x_pad = transpose_and_pad_icj(x, n_pad)
  mask_pad = cast_and_pad_mask(mask, n_pad, dtype, x)

  left_pad, right_pad = prologue(
      x_pad,
      mask_pad,
      projection_in_weights,
      gate_in_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      eps=epsilon,
      block_m=config.block_m,
      block_i=config.block_i,
      block_j=config.block_j,
  )

  if n != n_pad:
    left = left_pad[:, :n, :n]
    right = right_pad[:, :n, :n]
  else:
    left, right = left_pad, right_pad
  t3 = contract(left, right, triangle_type, precision)
  t3_pad = pad_dim(pad_dim(t3, n_pad, 1), n_pad, 2)

  epi_block_m = config.epilogue_block_m or config.block_m
  epi_block_i = (
      config.epilogue_block_i
      if config.epilogue_block_m is not None
      else (config.epilogue_block_i or config.block_i)
  )
  epi_block_j = (
      config.epilogue_block_j
      if config.epilogue_block_m is not None
      else (config.epilogue_block_j or config.block_j)
  )
  out_pad = epilogue(
      t3_pad,
      x_pad,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      eps=epsilon,
      block_m=epi_block_m,
      block_i=epi_block_i,
      block_j=epi_block_j,
  )

  out_idj = out_pad[:n, :d, :n] if n != n_pad else out_pad
  out = jnp.transpose(out_idj, (0, 2, 1))
  residuals = (left, right, t3) if return_residuals else None
  return out, residuals


def forward_pallas(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
):
  """Triangle multiplicative update on the Pallas TPU kernels.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `residuals = (left, right, t3)`; see the module docstring. The tile sizes
  come from `get_heuristics_config`. Off TPU the kernels run in Pallas
  interpret mode.
  """
  config = get_heuristics_config(x, projection_out_weights)
  return triangle_multiplication_pallas_fwd(
      x,
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      _TRIANGLE_TYPE,
      # Tokamax canonicalizes the default `precision=None` to this pair.
      precision=(lax.Precision.DEFAULT, lax.Precision.DEFAULT),
      epsilon=_EPSILON,
      return_residuals=True,
      config=config,
  )


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: seq_len=384, channels=64, triangle_type='outgoing'
    # (project 'alphafold'); the smallest of the 384/640/1024/2048 series.
    'trimul_n384_c64': {
        'name': 'trimul_outgoing_n384_c64_bwd_pallas_fwd',
        'model': 'AlphaFold',
        'operator': 'triangle_multiplication_backward',
        'seq_len': 384,
        'channels': 64,
        'hidden_channels': 64,
        'out_channels': 64,
    },
    # Small shapes for CPU smoke tests. N is a multiple of 128 and C, H, D
    # are in {16, 32, 64, 128}, as in the default config; the three channel
    # counts differ so that a swapped axis shows up as a shape error.
    'smoke': {
        'name': 'trimul_outgoing_smoke_bwd_pallas_fwd',
        'model': 'AlphaFold',
        'operator': 'triangle_multiplication_backward',
        'seq_len': 128,
        'channels': 32,
        'hidden_channels': 16,
        'out_channels': 64,
    },
}
CONFIG = CONFIGS['trimul_n384_c64']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns the ten arguments of `forward` followed by the cotangent `dout`.

  The cotangent is scaled by `N ** -0.5` so that the gradient with respect to
  `x` is above the absolute tolerance (1e-2) of the benchmark harness.
  """
  cfg = _resolve_config(config)
  N, C = cfg['seq_len'], cfg['channels']
  H, D = cfg['hidden_channels'], cfg['out_channels']
  keys = jax.random.split(jax.random.key(42), 11)
  # Value ranges follow the fan-in-scaled parameters of the Tokamax triangle
  # multiplication tests: standard normal activations, weights drawn with
  # standard deviation `fan_in ** -0.5`, layer norm scales near one and offsets
  # near zero, and a mask that keeps each pair with probability 0.5. The
  # cotangent is standard normal times `N ** -0.5`: the weight gradients are
  # sums over the N^2 pairs and about N times larger than the gradient with
  # respect to `x`, and this scale places the two on either side of one.
  x = jax.random.normal(keys[0], (N, N, C))
  mask = jax.random.bernoulli(keys[1], 0.5, (N, N))
  projection_in_weights = jax.random.normal(keys[2], (C, 2, H)) * C**-0.5
  gate_in_weights = jax.random.normal(keys[3], (C, 2, H)) * C**-0.5
  projection_out_weights = jax.random.normal(keys[4], (H, D)) * H**-0.5
  gate_out_weights = jax.random.normal(keys[5], (C, D)) * C**-0.5
  layernorm_in_scale = 1.0 + 0.1 * jax.random.normal(keys[6], (C,))
  layernorm_in_offset = 0.1 * jax.random.normal(keys[7], (C,))
  layernorm_out_scale = 1.0 + 0.1 * jax.random.normal(keys[8], (H,))
  layernorm_out_offset = 0.1 * jax.random.normal(keys[9], (H,))
  dout = jax.random.normal(keys[10], (N, N, D)) * N**-0.5
  # Sample in float32 and cast, so every dtype sees the same values. The mask
  # stays boolean.
  cast = lambda a: a.astype(dtype)
  return (
      cast(x),
      mask,
      cast(projection_in_weights),
      cast(gate_in_weights),
      cast(projection_out_weights),
      cast(gate_out_weights),
      cast(layernorm_in_scale),
      cast(layernorm_in_offset),
      cast(layernorm_out_scale),
      cast(layernorm_out_offset),
      cast(dout),
  )


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`
# (every argument except the boolean `mask`).
VJP_SPEC = {'num_primals': 10, 'diff_argnums': (0, 2, 3, 4, 5, 6, 7, 8, 9)}


def workload(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
    dout,
):
  """Returns the VJP of `forward` with `dout` for `x` and the eight weights.

  The result is the tuple `(dx, dprojection_in_weights, dgate_in_weights,
  dprojection_out_weights, dgate_out_weights, dlayernorm_in_scale,
  dlayernorm_in_offset, dlayernorm_out_scale, dlayernorm_out_offset)`.
  """
  fwd = lambda x, *weights: forward(x, mask, *weights)
  _, vjp_fn = jax.vjp(
      fwd,
      x,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward pass has four matmuls, counted at two FLOPs per multiply-add:
  the two input projections `[N, N, C] @ [C, 2H]` (8 N^2 C H together), the
  triangle contraction of two `[H, N, N]` planes (2 H N^3), the output
  projection `[N, N, H] @ [H, D]` (2 N^2 H D) and the output gate
  `[N, N, C] @ [C, D]` (2 N^2 C D). The VJP runs the forward once and a
  backward that costs about twice the forward, so 3x in total.
  """
  cfg = _resolve_config(config)
  N, C = cfg['seq_len'], cfg['channels']
  H, D = cfg['hidden_channels'], cfg['out_channels']
  forward_flops = 2 * H * N**3 + N**2 * (8 * C * H + 2 * H * D + 2 * C * D)
  return int(3 * forward_flops)


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = _resolve_config(config)
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = jax.block_until_ready(fn(*inputs))
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*inputs))
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  return {
      'name': cfg['name'],
      'model': cfg['model'],
      'operator': cfg['operator'],
      'config': {
          k: v for k, v in cfg.items() if k not in ('name', 'model', 'operator')
      },
      'time_ms': round(float(np.mean(times)), 4),
      'std_ms': round(float(np.std(times)), 4),
      'output_shape': [list(t.shape) for t in jax.tree.leaves(out)],
      'status': 'success',
  }


if __name__ == '__main__':
  print(json.dumps(benchmark()))
