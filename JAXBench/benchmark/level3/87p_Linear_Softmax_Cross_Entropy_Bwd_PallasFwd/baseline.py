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
"""Linear softmax cross-entropy loss backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

The op fuses the output projection `x @ w` of a language model with the softmax
cross-entropy loss against integer labels, and returns the loss averaged over
tokens as a float32 scalar.

`forward_pallas(x, labels, w)` is the Tokamax Pallas TPU forward kernel. It
tiles B, H and V, accumulates each `[tb, tv]` block of logits in VMEM and
combines the vocabulary blocks with a running log-sum-exp, so the `[B, V]`
logits are never written to HBM. It returns `(loss, residuals)`, where
`residuals = (lse,)` is what the kernel saves for a backward pass:
  lse  `[B]` float32  log-sum-exp of each token's logits over the vocabulary

`workload(x, labels, w, dloss)` returns `(dx, dw)`, the gradients of the mean
loss with respect to `x` and `w` multiplied by the scalar cotangent `dloss`.
`labels` is an integer input and has no gradient. `workload` is the correctness
oracle: JAX autodiff (`jax.vjp`) of the pure-JAX forward reference `forward`.
This file contains no backward kernel and no hand-derived backward pass.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved `lse`) so that your `workload` returns the same gradients
as this one.

The kernel and its block-size heuristic read the hardware description of the
TPU they run on. Off TPU, `forward_pallas` runs the kernel in Pallas TPU
interpret mode with the description of a single-core TPU v5e.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py  (forward kernel, block-size heuristic)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py  (how the op calls the heuristic and the kernel)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py   (forward reference)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/arg_specs.py   (shapes, reduction)
  tokamax/_src/ops/linear_softmax_cross_entropy_loss/test_utils.py  (input distributions)
"""

# Dimensions.
#   B   number of tokens (batch and sequence flattened)
#   H   hidden dimension
#   V   vocabulary size
#   tb  tile extent of B (`b_block_size`)
#   th  tile extent of H (`h_block_size`)
#   tv  tile extent of V (`v_block_size`)

import dataclasses
import functools
import json
import math
from typing import Literal, Sequence

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference
# (from tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================


def forward(x, labels, w):
  """Mean softmax cross-entropy loss of the logits `x @ w`.

  Args:
    x: Activations of the last layer, `[B, H]`.
    labels: Target class indices, `[B]`, integers in `[0, V)`.
    w: Output projection weights, `[H, V]`.

  Returns:
    The loss averaged over the `B` tokens, a float32 scalar. The logits and the
    log-softmax are computed in the dtype of the inputs; the sums over the
    vocabulary and over tokens accumulate in float32.
  """
  logits = x @ w
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  loss = -labels_one_hot * log_probs
  loss_per_sample = jnp.sum(loss, axis=-1, dtype=jnp.float32)
  return jnp.mean(loss_per_sample, dtype=jnp.float32)


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from
#  tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py,
#  Copyright 2025 DeepMind Technologies Limited, Apache-2.0)
# ==============================================================================

partial = functools.partial


@dataclasses.dataclass(frozen=True)
class Config:
  """The configuration specific for the Pallas Mosaic TPU kernel.

  Attributes:
    b_block_size: The block size for the batch dimension.
    h_block_size: The block size for the hidden dimension.
    v_block_size: The block size for the vocabulary dimension.
  """

  b_block_size: int = 1024
  h_block_size: int = 512
  v_block_size: int = 2048


def _calculate_fwd_vmem_bytes(
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
) -> int:
  """Calculates VMEM memory usage in bytes for the forward kernel."""
  dtype_bytes = jnp.dtype(dtype).itemsize
  h_alloc = 1 << (h_block_size - 1).bit_length()
  return (
      # x tile (B, H) in input dtype (double buffered for pallas_call input)
      2 * b_block_size * h_alloc * dtype_bytes
      # labels (8 bytes), lse (8 bytes), and loss (8 bytes) tiles (all double
      # buffered for pallas_call inputs/outputs, 24 bytes per batch elem total)
      + 2 * 3 * 4 * b_block_size
      # w tile (H, V) in input dtype (double buffered for pallas_call input)
      + 2 * h_alloc * v_block_size * dtype_bytes
      # logits/xw tile (B, V) in float32 accumulator:
      # We account for 4 simultaneous (B, V) float32 buffers
      # (4 * 4 = 16 bytes/elem):
      #   1) xw_tiled (explicit scratch buffer)
      #   2) labels_one_hot (HLO stack temporary in accumulate_loss)
      #   3) diff = xw_tiled - max (HLO stack temporary in logsumexp)
      #   4) exp(diff) (HLO stack temporary in logsumexp)
      + 4 * b_block_size * v_block_size * 4
  )


def _get_vmem_limit_bytes() -> int:
  return int(0.90 * pltpu.get_tpu_info().vmem_capacity_bytes)


def _get_heuristic_config(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    *,
    dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
    vmem_limit_bytes: int | None = None,
) -> Config:
  """Calculates heuristic config based on VMEM size, dtype, and divisibility."""
  if vmem_limit_bytes is None:
    vmem_limit_bytes = _get_vmem_limit_bytes()

  calc_vmem_fn = _calculate_fwd_vmem_bytes
  dtype_bytes = jnp.dtype(dtype).itemsize
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes

  # 1. Choose b_block_size: needs at least num_sublanes * num_lanes, multiple of
  # num_lanes. Prefer b_block_size such that b_dim % b_block_size == 0.
  min_b_block_size = tpu_info.num_sublanes * num_lanes
  b_candidates = [
      b
      for b in range(
          min_b_block_size, max(min_b_block_size + 1, b_dim + 1), num_lanes
      )
      if b_dim % b == 0
  ]
  if b_candidates:
    b_block_size = b_candidates[0]
  else:
    b_block_size = min_b_block_size

  # 2. Choose h_block_size: at least num_lanes, multiple of num_lanes, close to
  # h_dim_max. Prefer h_block_size such that h_dim % h_block_size == 0.
  # Align h_dim_max down to a multiple of num_lanes (and at least num_lanes) so
  # that the fallbacks below always produce a lane-aligned h_block_size.
  raw_h_dim_max = tpu_info.mxu_column_size * tpu_info.num_mxus
  h_dim_max = max(num_lanes, (raw_h_dim_max // num_lanes) * num_lanes)
  h_candidates = [
      h
      for h in range(num_lanes, max(num_lanes + 1, h_dim + 1), num_lanes)
      if h_dim % h == 0
  ]
  if h_candidates:
    h_block_size = min(
        h_candidates,
        key=lambda x: (
            0 if (x & (x - 1)) == 0 else 1,
            abs(x - h_dim_max),
        ),
    )
  else:
    if h_dim < h_dim_max:
      h_candidates_non_div = [
          h for h in range(num_lanes, h_dim_max + 1, num_lanes) if h >= h_dim
      ]
      if h_candidates_non_div:
        h_block_size = min(
            h_candidates_non_div,
            key=lambda x: (
                0 if (x & (x - 1)) == 0 else 1,
                x,
            ),
        )
      else:
        h_block_size = h_dim_max
    else:
      h_block_size = h_dim_max

  # 3. Choose v_block_size: as large as possible to fit VMEM.
  # Must be >= num_lanes, multiple of num_lanes. Divisible by v_dim if possible.
  h_alloc = 1 << (h_block_size - 1).bit_length()
  # fixed_bytes accounts for VMEM costs that do not scale with V:
  #   - x tile = 2 * b_block_size * h_alloc * dtype_bytes
  #   - labels (8 bytes) + lse (8 bytes) + loss (8 bytes) = 24 * b_block_size
  fixed_bytes = 2 * b_block_size * h_alloc * dtype_bytes + 24 * b_block_size
  # per_v_bytes accounts for all VMEM costs per column of V:
  #   - w tile (double-buff in dtype: 2 * dt) = 2 * h_alloc * dtype_bytes
  #   - 4 simultaneous float32 (B, V) buffers on the VMEM stack:
  #     (4 * 4 = 16 bytes per element across b_block_size):
  #       1) xw_tiled (explicit scratch buffer)
  #       2) labels_one_hot (HLO stack temporary in accumulate_loss)
  #       3) diff = xw_tiled - max (HLO stack temporary in logsumexp)
  #       4) exp(diff) (HLO stack temporary in logsumexp)
  per_v_bytes = 2 * h_alloc * dtype_bytes + 16 * b_block_size

  if vmem_limit_bytes > fixed_bytes:
    max_v_vmem = (vmem_limit_bytes - fixed_bytes) // per_v_bytes
  else:
    max_v_vmem = num_lanes

  max_v = min(v_dim, max_v_vmem)
  max_v_aligned = max(num_lanes, (max_v // num_lanes) * num_lanes)

  max_cores = tpu_info.num_cores

  def is_mc_compatible(v: int) -> bool:
    if max_cores <= 1:
      return True
    n_v_b = math.ceil(v_dim / v)
    return n_v_b % max_cores == 0

  # 1. First try to find a divisor of v_dim that fits in VMEM and is multi-core
  # compatible.
  v_block_size = None
  for v in range(max_v_aligned, num_lanes - 1, -num_lanes):
    if (
        v_dim % v == 0
        and is_mc_compatible(v)
        and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
        <= vmem_limit_bytes
    ):
      v_block_size = v
      break

  # 2. If no divisor fits, pick a multiple of num_lanes that is multi-core
  # compatible
  if v_block_size is None:
    for v in range(max_v_aligned, num_lanes - 1, -num_lanes):
      if (
          is_mc_compatible(v)
          and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break

  # 3. Fallback: try divisor of v_dim that fits in VMEM
  if v_block_size is None:
    for v in range(max_v_aligned, num_lanes - 1, -num_lanes):
      if (
          v_dim % v == 0
          and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break

  # 4. Fallback: pick the largest multiple of num_lanes that fits
  if v_block_size is None:
    for v in range(max_v_aligned, num_lanes - 1, -num_lanes):
      if (
          calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break
    if v_block_size is None:
      v_block_size = num_lanes

  return Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )


def get_heuristic_fwd_config(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
    vmem_limit_bytes: int | None = None,
) -> Config:
  """Returns heuristic config for forward pass based on VMEM size and dtype."""
  return _get_heuristic_config(
      b_dim,
      h_dim,
      v_dim,
      dtype=dtype,
      vmem_limit_bytes=vmem_limit_bytes,
  )


def validate_inputs(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> None:
  """Validates the inputs to the kernels.

  Validate inputs and raise ValueError if the inputs are invalid.

  Args:
    x: The last layer output in the dimension of (B, H) where B is the batch
      dimension, and H is the hidden dimension.
    labels: The ground truth label index in the dimension of (B,).
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The batch block size.
    h_block_size: The hidden block size.
    v_block_size: The vocabulary block size.

  Raises:
    ValueError: If the inputs are invalid.
  """
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes
  if b_block_size <= 0 or b_block_size % num_lanes != 0:
    raise ValueError(
        f"b_block_size must be a positive multiple of {num_lanes}, got"
        f" {b_block_size}."
    )
  if h_block_size <= 0 or h_block_size % num_lanes != 0:
    raise ValueError(
        f"h_block_size must be a positive multiple of {num_lanes}, got"
        f" {h_block_size}."
    )
  if v_block_size <= 0 or v_block_size % num_lanes != 0:
    raise ValueError(
        f"v_block_size must be a positive multiple of {num_lanes}, got"
        f" {v_block_size}."
    )

  if labels.shape[0] != x.shape[0]:
    raise ValueError(
        f"Batch dimension mismatch: labels batch dimension ({labels.shape[0]})"
        f" != x batch dimension ({x.shape[0]})."
    )

  if x.shape[-1] != w.shape[0]:
    raise ValueError(
        f"Hidden dimension mismatch: x hidden dimension ({x.shape[-1]}) !="
        f" w hidden dimension ({w.shape[0]})."
    )

  num_sublanes = tpu_info.num_sublanes
  if w.shape[0] % num_sublanes != 0:
    raise ValueError(
        f"The hidden dimension of w must be a multiple of {num_sublanes}, got"
        f" {w.shape[0]}."
    )


@jax.named_scope("calculate_xw_tiled")
def calculate_xw_tiled(
    x_ref,
    w_ref,
    xw_tiled,
    b_index,
    h_index,
    v_index,
    num_b_blocks,
    num_h_blocks,
    num_v_blocks,
    b_dim,
    h_dim,
    v_dim,
    preferred_element_type: jnp.dtype,
):
  """Calculates xw_tiled += x@w for forward/backward kernel common logic."""
  b_block_size, h_block_size, v_block_size = (
      x_ref.shape[0],
      x_ref.shape[1],
      w_ref.shape[1],
  )
  x_val = x_ref[...]
  if b_dim % b_block_size != 0:
    rem_b = b_dim % b_block_size
    row_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(b_block_size, 1), dimension=0
    )
    x_val = jnp.where(
        (b_index == num_b_blocks - 1) & (row_idx >= rem_b), 0.0, x_val
    )
  if h_dim % h_block_size != 0:
    rem_h = h_dim % h_block_size
    col_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(1, h_block_size), dimension=1
    )
    x_val = jnp.where(
        (h_index == num_h_blocks - 1) & (col_idx >= rem_h), 0.0, x_val
    )

  w_val = w_ref[...]
  if h_dim % h_block_size != 0:
    rem_h = h_dim % h_block_size
    row_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(h_block_size, 1), dimension=0
    )
    w_val = jnp.where(
        (h_index == num_h_blocks - 1) & (row_idx >= rem_h), 0.0, w_val
    )
  if v_dim % v_block_size != 0 or num_v_blocks * v_block_size > v_dim:
    col_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(1, v_block_size), dimension=1
    )
    w_val = jnp.where((v_index * v_block_size + col_idx) >= v_dim, 0.0, w_val)

  @pl.when(h_index == 0)
  def init_xw():
    xw_tiled[...] = jax.lax.dot_general(
        x_val,
        w_val,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )

  @pl.when(h_index != 0)
  def accumulate_xw():
    xw_tiled[...] += jax.lax.dot_general(
        x_val,
        w_val,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )


def _bytes(tensor: jax.Array | jax.ShapeDtypeStruct) -> int:
  """Computes total bytes of a tensor or ShapeDtypeStruct."""
  return math.prod(tensor.shape) * jnp.dtype(tensor.dtype).itemsize


def linear_softmax_cross_entropy_loss_fwd_cost_estimate(
    x: jax.Array | jax.ShapeDtypeStruct,
    labels: jax.Array | jax.ShapeDtypeStruct,
    w: jax.Array | jax.ShapeDtypeStruct,
    out_type: Sequence[jax.ShapeDtypeStruct] | None = None,
) -> pl.CostEstimate:
  """Calculates the theoretical hardware cost estimate for the forward pass kernel.

  This provides an accurate model of computational (FLOPs), transcendental,
  and memory bandwidth (bytes accessed) costs for profilers and runtime cost
  models
  (e.g., XProf / HLO CostAnalysis).

  FLOPs Breakdown:
    1. Logit Projection Matmul (X @ W):
       - Input X: (B, H), Weight W: (H, V)
       - Standard matmul FLOPs: 2 * B * H * V (1 multiply + 1 accumulate per
       element).
    2. Online LogSumExp Accumulation (jax.nn.logsumexp across V blocks):
       - Maximum subtraction: B * V ops
       - Exponentiation: B * V ops (tracked in transcendentals)
       - Reduction sum: B * V ops
       - Logarithm of sum: B ops (tracked in transcendentals)
       - Running logaddexp combining blocks: 2 * B * V ops
       - Total elementwise reduction FLOPs: 2 * B * V.
    3. Sparse Target Loss Reduction:
       - Target extraction / one-hot subtraction: B ops
       - Total loss FLOPs: B.

    Total FLOPs = 2 * B * H * V + 2 * B * V + B.

  Transcendental Operations Breakdown:
    - Exponentiation (exp) in logsumexp across vocabulary V: B * V ops.
    - Logarithm (log) in logsumexp per batch element: B ops.
    Total Transcendentals = B * V + B.

  HBM Memory Traffic Breakdown:
    - Inputs read from HBM:
      - Activations x: B * H * sizeof(dtype_x)
      - Labels: B * sizeof(dtype_labels)
      - Weights w: H * V * sizeof(dtype_w)
    - Outputs written to HBM:
      - Loss: B * sizeof(float32) (4 bytes)
      - LSE: B * sizeof(float32) (4 bytes)
    Total Bytes Accessed = sum(input_bytes) + sum(output_bytes).

  Args:
    x: Input activation tensor or ShapeDtypeStruct with shape (B, H).
    labels: Target label tensor or ShapeDtypeStruct with shape (B,).
    w: Weight projection tensor or ShapeDtypeStruct with shape (H, V).
    out_type: Optional sequence of output ShapeDtypeStructs for loss and LSE.

  Returns:
    A `pl.CostEstimate` object containing flops, transcendentals, and
    bytes_accessed.
  """
  b_dim = x.shape[0]
  h_dim = x.shape[1]
  v_dim = w.shape[1]

  matmul_flops = 2 * b_dim * h_dim * v_dim
  reduction_flops = 2 * b_dim * v_dim + b_dim
  total_flops = matmul_flops + reduction_flops

  transcendentals = b_dim * v_dim + b_dim

  inputs = [x, labels, w]
  input_bytes = sum(_bytes(t) for t in inputs)
  if out_type is not None:
    output_bytes = sum(_bytes(t) for t in out_type)
  else:
    output_bytes = 2 * b_dim * 4

  return pl.CostEstimate(
      flops=int(total_flops),
      transcendentals=int(transcendentals),
      bytes_accessed=int(input_bytes + output_bytes),
  )


def linear_softmax_cross_entropy_loss_forward_pallas_kernel(
    x,
    labels,
    w,
    *,
    reduction: Literal["sum", "mean", "none"],
    h_dim: int,
    v_dim: int,
    preferred_element_type: jnp.dtype,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Pallas kernel for the forward pass of Linear Softmax Cross-Entropy Loss.

  This kernel uses a block-wise algorithm on all B, H and V dimensions. The B
  and H dimensions can be accumulated linearly. The accumulation
  on V dimension is using the log linearity of log-sum-exp and log-softmax.
  The kernel will return both loss and additionally the log-sum-exp for the
  backward pass. However the x@w won't be returned to avoid additional buffer so
  backward pass
  will need to re-compute x@w. Overall, this kernel will keep all the
  intermediate buffers in VMEM without logits HBM materialization.

  Args:
    x: Input activations `x` (b_dim, h_dim).
    labels: One-hot encoded labels (b_dim, v_dim).
    w: LM Head projection weights `w` (h_dim, v_dim).
    reduction: The reduction method ("sum", "mean" or "none") for the loss
      accumulation.
    h_dim: Hidden dimension size.
    v_dim: Vocabulary dimension size.
    preferred_element_type: Preferred element type for computation.
    b_block_size: Block size for batch dimension.
    h_block_size: Block size for hidden dimension.
    v_block_size: Block size for vocabulary dimension.

  Returns:
    A tuple of (loss, lse).
  """
  b_dim = x.shape[0]
  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)

  out_type = [
      jax.ShapeDtypeStruct(shape=(b_dim,), dtype=jnp.float32),  # Loss
      jax.ShapeDtypeStruct(shape=(b_dim,), dtype=jnp.float32),  # LSE
  ]
  get_b_ds = lambda i: pl.ds(
      i * b_block_size, jnp.minimum(b_block_size, b_dim - i * b_block_size)
  )
  loss_out_spec = pl.BlockSpec(
      (pl.BoundedSlice(b_block_size),),
      lambda i, j, k: (get_b_ds(i),),
      memory_space=pltpu.VMEM,
  )
  lse_out_spec = pl.BlockSpec(
      (pl.BoundedSlice(b_block_size),),
      lambda i, j, k: (get_b_ds(i),),
      memory_space=pltpu.VMEM,
  )

  cost_estimate = linear_softmax_cross_entropy_loss_fwd_cost_estimate(
      x=x,
      labels=labels,
      w=w,
      out_type=out_type,
  )

  @pl.kernel(
      out_type=out_type,
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
      scratch_types=(
          pltpu.VMEM(
              (b_block_size, v_block_size), dtype=jnp.float32
          ),  # xw_tiled
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_get_vmem_limit_bytes(),
          disable_bounds_checks=True,
      ),
      cost_estimate=cost_estimate,
      name=f"lce_fwd_bt_{b_block_size}_ht_{h_block_size}_vt_{v_block_size}",
      interpret=(
          pltpu.InterpretParams() if jax.default_backend() != "tpu" else False
      ),
  )
  def fwd_kernel(
      x_hbm_ref,
      labels_hbm_ref,
      w_hbm_ref,
      loss_hbm_ref,
      lse_hbm_ref,
      xw_tiled_ref,
  ):
    def fwd_pipeline(
        step,
        x_ref,
        labels_ref,
        w_ref,
        loss_ref,
        lse_ref,
        xw_tiled,
    ):
      b_index, v_index, h_index = step.index

      # xw_tiled += x_ref @ w_ref
      calculate_xw_tiled(
          x_ref,
          w_ref,
          xw_tiled,
          b_index=b_index,
          h_index=h_index,
          v_index=v_index,
          num_b_blocks=num_b_blocks,
          num_h_blocks=num_h_blocks,
          num_v_blocks=num_v_blocks,
          b_dim=b_dim,
          h_dim=h_dim,
          v_dim=v_dim,
          preferred_element_type=preferred_element_type,
      )

      @pl.when(jnp.logical_and(v_index == 0, h_index == 0))
      @jax.named_scope("init_lse")
      def init_lse():
        lse_ref[...] = jnp.full_like(lse_ref, -jnp.inf)
        loss_ref[...] = jnp.zeros_like(loss_ref)

      @pl.when(h_index == num_h_blocks - 1)
      @jax.named_scope("accumulate_loss")
      def accumulate_loss():
        # Convert labels to one-hot, due to chunking on v dimension, the indices
        # needs to be shifted down by the v starting index. Negative or
        # out-of-bound indices are OK since jax.nn.one_hot will set them to 0.
        labels_adjusted = labels_ref[...] - v_index * v_block_size
        labels_one_hot = jax.nn.one_hot(
            labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
        )
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, labels_one_hot.shape, dimension=0
          )
          labels_one_hot = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              0.0,
              labels_one_hot,
          )
        loss_ref[...] -= jnp.sum(labels_one_hot * xw_tiled[...], axis=-1)
        if v_dim % v_block_size != 0:
          rem_v = v_dim % v_block_size
          col_idx = jax.lax.broadcasted_iota(
              jnp.int32, xw_tiled.shape, dimension=1
          )
          xw_tiled[...] = jnp.where(
              (v_index == num_v_blocks - 1) & (col_idx >= rem_v),
              -jnp.inf,  # forcing exp(-inf)=0, instead of 1
              xw_tiled[...],
          )
        lse_block = jax.nn.logsumexp(xw_tiled[...], axis=-1)
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, lse_block.shape, dimension=0
          )
          lse_block = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              -jnp.inf,
              lse_block,
          )
        lse_ref[...] = jnp.logaddexp(lse_ref[...], lse_block)

      @pl.when(
          jnp.logical_and(
              v_index == num_v_blocks - 1, h_index == num_h_blocks - 1
          )
      )
      def perform_loss_reduction():
        loss_ref[...] += lse_ref[...]

    get_h_ds = lambda k: pl.ds(
        k * h_block_size, jnp.minimum(h_block_size, h_dim - k * h_block_size)
    )
    get_v_ds = lambda j: pl.ds(
        j * v_block_size, jnp.minimum(v_block_size, v_dim - j * v_block_size)
    )

    pltpu.emit_pipeline(
        fwd_pipeline,
        grid=(num_b_blocks, num_v_blocks, num_h_blocks),
        in_specs=[
            pl.BlockSpec(
                (pl.BoundedSlice(b_block_size), pl.BoundedSlice(h_block_size)),
                lambda i, j, k: (get_b_ds(i), get_h_ds(k)),
                memory_space=pltpu.VMEM,
            ),  # x
            pl.BlockSpec(
                (pl.BoundedSlice(b_block_size),),
                lambda i, j, k: (get_b_ds(i),),
                memory_space=pltpu.VMEM,
            ),  # labels
            pl.BlockSpec(
                (pl.BoundedSlice(h_block_size), pl.BoundedSlice(v_block_size)),
                lambda i, j, k: (get_h_ds(k), get_v_ds(j)),
                memory_space=pltpu.VMEM,
            ),  # w
        ],
        out_specs=[
            loss_out_spec,  # loss
            lse_out_spec,  # lse
        ],
        core_axis_name="core",
        dimension_semantics=(
            pltpu.PARALLEL,
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
        ),
        _explicit_indices=True,
    )(
        x_hbm_ref,
        labels_hbm_ref,
        w_hbm_ref,
        loss_hbm_ref,
        lse_hbm_ref,
        scratches=(xw_tiled_ref,),
    )

  loss, lse = fwd_kernel(x, labels, w)  # pylint: disable=unpacking-non-sequence
  if reduction == "sum":
    return jnp.sum(loss), lse
  elif reduction == "mean":
    return jnp.mean(loss), lse
  else:
    return loss, lse


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
        "preferred_element_type",
    ],
)
def linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    reduction: Literal["sum", "mean", "none"] = "sum",
    preferred_element_type: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
) -> tuple[jax.Array, jax.Array]:
  """The pallas kernel implementation of linear softmax cross-entropy loss.

  This implementation is chunking the x, labels and w in all B, H and V
  dimensions so it can fit in the TPU VMEM, resulting in almost 0 additional
  buffer overhead. The V dimension chunking is non-linear so this kernel uses
  online softmax algorithm to chunk.

  Args:
    x: The last layer output in the dimension of (B, H) where B is the batch and
      H is the hidden dimension.
    labels: The ground truth labels index in the dimension of (B,)
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The batch block size.
    h_block_size: The hidden block size.
    v_block_size: The vocabulary block size.
    reduction: The reduction method ("sum", "mean" or "none") for the loss
      accumulation.

  Returns:
    The loss in scalar and the log-sum-exp of the used for backward pass.

  Raises:
    ValueError: If the invalid configuration is provided.
  """
  validate_inputs(
      x,
      labels,
      w,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )

  if x.dtype == jnp.float16:
    x = x.astype(preferred_element_type)
  if w.dtype == jnp.float16:
    w = w.astype(preferred_element_type)

  h_dim = x.shape[-1]
  v_dim = w.shape[1]

  # Constrain the memory spaces for x, labels, and w to prevent OOB accesses
  # that occur when the memory spaces is placed in VMEM.
  x = pltpu.with_memory_space_constraint(x, memory_space=pltpu.HBM)
  labels = pltpu.with_memory_space_constraint(labels, memory_space=pltpu.HBM)
  w = pltpu.with_memory_space_constraint(w, memory_space=pltpu.HBM)

  # Forward
  loss, lse = linear_softmax_cross_entropy_loss_forward_pallas_kernel(
      x,
      labels,
      w,
      reduction=reduction,
      h_dim=h_dim,
      v_dim=v_dim,
      preferred_element_type=preferred_element_type,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )
  return loss, lse


# `pltpu.get_tpu_info()`, which the heuristic and the kernel call, needs a TPU
# device to describe. Off TPU this single-core TPU v5e stands in for it.
_OFF_TPU_DEVICE = jax.sharding.AbstractDevice(
    device_kind='TPU v5e', num_cores=1, platform='tpu'
)


def _fwd_pallas_heuristic_blocks(x, labels, w):
  """Runs the forward kernel with the block sizes Tokamax's heuristic picks."""
  config = get_heuristic_fwd_config(
      x.shape[0], x.shape[1], w.shape[1], dtype=x.dtype
  )
  return linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
      x,
      labels,
      w,
      b_block_size=config.b_block_size,
      h_block_size=config.h_block_size,
      v_block_size=config.v_block_size,
      reduction='mean',
  )


def forward_pallas(x, labels, w):
  """Mean softmax cross-entropy loss on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(loss, residuals)` with
  `residuals = (lse,)`; see the module docstring. Off TPU the kernel runs in
  Pallas TPU interpret mode, tiled as it would be on a single-core TPU v5e.
  """
  if jax.default_backend() == 'tpu':
    loss, lse = _fwd_pallas_heuristic_blocks(x, labels, w)
  else:
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=_OFF_TPU_DEVICE)
    with jax.sharding.use_abstract_mesh(mesh):
      loss, lse = _fwd_pallas_heuristic_blocks(x, labels, w)
  return loss, (lse,)


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: the llama3.1-8b entry tagged 'primary' (batch_dim=1024,
    # hidden_dim=4096, output_dim=128256, reduction='mean', bfloat16).
    'llama3_8b_b1024': {
        'name': 'lsce_llama3_8b_b1024_bwd_pallas_fwd',
        'model': 'Llama-3.1-8B',
        'operator': 'linear_softmax_cross_entropy_loss_backward',
        'batch_tokens': 1024,
        'hidden_dim': 4096,
        'vocab_size': 128256,
    },
    # Small shapes for CPU smoke tests. Like the default, every dimension is a
    # whole number of the blocks the Tokamax TPU kernel tiles it into (1024
    # tokens, 512 hidden units, vocabulary blocks that are multiples of 128).
    'smoke': {
        'name': 'lsce_smoke_bwd_pallas_fwd',
        'model': 'Llama-3.1-8B',
        'operator': 'linear_softmax_cross_entropy_loss_backward',
        'batch_tokens': 1024,
        'hidden_dim': 1024,
        'vocab_size': 8192,
    },
}
CONFIG = CONFIGS['llama3_8b_b1024']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(x, labels, w, dloss)`; `dloss` is the float32 scalar cotangent.

  The gradient of a mean loss is O(1/B), far below the absolute tolerance of
  1e-2 that the benchmark harness compares with. The cotangent is therefore
  `B * sqrt(H) / 4` instead of one, which scales the gradient entries to order
  one: large enough that zeros fail that tolerance and small enough that
  bfloat16 rounding in the reference does not exceed it. The VJP is linear in
  the cotangent.
  """
  cfg = _resolve_config(config)
  B, H, V = cfg['batch_tokens'], cfg['hidden_dim'], cfg['vocab_size']
  k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
  # Standard-normal activations and uniformly drawn labels, as in the Tokamax
  # tests of this op (test_utils.generate_random_data). The weights are scaled
  # to fan-in, `H ** -0.5`, so the logits are O(1) and the softmax is not
  # saturated.
  x = jax.random.normal(k1, (B, H))
  labels = jax.random.randint(k2, (B,), 0, V)
  w = jax.random.normal(k3, (H, V)) * H**-0.5
  dloss = jnp.asarray(B * H**0.5 / 4, dtype=jnp.float32)
  # Sample in float32 and cast, so every dtype sees the same values. The
  # cotangent keeps the float32 dtype of the loss.
  return x.astype(dtype), labels, w.astype(dtype), dloss


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 2)}


def workload(x, labels, w, dloss):
  """Returns `(dx, dw)`, the VJP of `forward` in `x` and `w` with `dloss`."""
  _, vjp_fn = jax.vjp(lambda x_, w_: forward(x_, labels, w_), x, w)
  return vjp_fn(dloss)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  The forward is dominated by the `[B, H] x [H, V]` projection, `2 * B * H * V`
  FLOPs. The backward does two matmuls of the same size, one per gradient
  (`[B, V] x [V, H]` for `dx` and `[H, B] x [B, V]` for `dw`), so it costs
  twice the forward: 3x in total. The elementwise softmax terms are `O(B * V)`
  and are not counted.
  """
  cfg = _resolve_config(config)
  B, H, V = cfg['batch_tokens'], cfg['hidden_dim'], cfg['vocab_size']
  return int(3 * 2 * B * H * V)


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
