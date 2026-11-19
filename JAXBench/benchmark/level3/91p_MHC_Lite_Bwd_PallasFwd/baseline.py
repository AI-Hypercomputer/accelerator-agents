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
"""mHC-lite fused post-and-pre backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

mHC-lite (Manifold-Constrained Hyper-Connections) carries `S` residual streams
per token. Between two wrapped layers the fused `post_and_pre` op does:
  post  `x_next[j] = sum_k residual[j, k] * x[k] + h_post[j] * layer_output`
  pre   RMS-normalises `x_next`, projects it with the learned weights of the
        next layer into three gates, and returns
          layer_input    `sum_k h_pre[k] * x_next[k]`, `h_pre = sigmoid + eps`
          h_post_next    `2 * sigmoid`
          residual_next  a softmax-weighted mix of the `S!` permutation matrices
`forward` returns `(layer_input, x_next, h_post_next, residual_next)`.

`forward_pallas(...)` takes the arguments of `forward` and is the Tokamax
Pallas TPU forward kernel for this op. It returns `(out, residuals)`, where
`out` matches `forward` and `residuals = (phi, h_pre)` are the intermediates
the kernel saves for a backward pass:
  phi    `[S * E, 2 * S + P]` float32  `norm_scale[:, None] * concat(w_pre,
                                       w_post, w_res)`, not scaled by the alphas
  h_pre  `[T, S]` float32              pre gate of the next layer
The kernel supports bfloat16 activations only, with `S = 4`, `P = 24`, `E` a
multiple of the TPU lane count and `T` a multiple of the sublane count (128
and 8 for the TPU description Tokamax uses off TPU); it raises on other
inputs. Block sizes come from the Tokamax heuristic for the input shape.

`workload(layer_output, x, h_post, residual, norm_scale, w_pre, b_pre,
alpha_pre, w_post, b_post, alpha_post, w_res, b_res, alpha_res, permutations,
d_layer_input, d_x_next, d_h_post_next, d_residual_next)` returns the 14
gradients `(d_layer_output, d_x, d_h_post, d_residual, d_norm_scale, d_w_pre,
d_b_pre, d_alpha_pre, d_w_post, d_b_post, d_alpha_post, d_w_res, d_b_res,
d_alpha_res)` for the four output cotangents. It is the correctness oracle: JAX
autodiff (`jax.vjp`) of the pure-JAX reference `forward`. This file contains no
backward kernel and no hand-derived backward pass. `permutations` is a constant
and has no gradient.

`h_post`, `h_post_next` and their cotangent and gradient are float32 in every
precision, and `permutations` is float32, as in the Tokamax op. All other
arrays take the benchmark dtype.

In bfloat16 the gradients are those of the op whose pre stage consumes its own
bfloat16 `x_next` output. A kernel that feeds an unrounded float32 `x_next`
into the pre stage differs from `workload` by more than 1e-2 on a few percent
of the elements of the pre-gate weight gradient `d_w_pre`.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` (it
may consume the saved residuals) so that your `workload` returns the same
gradients as this one.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/mhc_lite/mhc_kernels_fwd.py    (forward kernel)
  tokamax/_src/ops/experimental/mhc_lite/common.py     (kernel config, tiling
                                                        heuristic, FLOP model)
  tokamax/_src/ops/experimental/mhc_lite/pallas_mosaic_tpu.py  (op config,
                                                        heuristic entry point)
  tokamax/_src/ops/experimental/mhc_lite/mhc_xla.py    (forward reference)
  tokamax/_src/ops/experimental/mhc_lite/base.py       (op outputs, epsilons)
  tokamax/_src/ops/experimental/mhc_lite/arg_specs.py  (shapes, dtypes)
  tokamax/_src/ops/experimental/mhc_lite/mhc_test.py   (value ranges)
"""

# Dimensions.
#   T   number of tokens (batch * sequence)
#   S   number of residual streams (expansion rate)
#   E   embedding dimension
#   P   number of permutation matrices (S!)

import contextlib
import dataclasses
import functools
import itertools
import json
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/mhc_lite/mhc_xla.py and
# base.py)
# ==============================================================================

# Defaults of the Tokamax op.
_RMS_EPSILON = 1e-5
_PRE_MAPPING_EPSILON = 1e-6


def _gate_logits(normalized, w, alpha, b):
  """Computes float32 gate logits `alpha * (normalized @ w) + b`."""
  with jax.named_scope("gate_logits"):
    return alpha * jnp.dot(normalized, w) + b


def _pre_xla_raw(
    x,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
    *,
    rms_epsilon=1e-5,
    pre_mapping_epsilon=1e-6,
):
  """Computes `(layer_input, x_2d, h_post, residual_f32, h_pre)` in JAX/XLA."""
  if x.ndim == 2:
    tokens, flattened_size = x.shape
    streams = permutations.shape[1]
    embedding = flattened_size // streams
    layer_input_shape = (tokens, embedding)
  elif x.ndim == 4:
    batch, sequence, streams, embedding = x.shape
    tokens = batch * sequence
    flattened_size = streams * embedding
    layer_input_shape = (batch, sequence, embedding)
  else:
    raise ValueError(
        "Expected x to have shape (batch, sequence, streams, embedding) or"
        f" (tokens, streams * embedding); got {x.shape}."
    )
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  with jax.named_scope("rms_norm"):
    x_f32 = x.reshape(tokens, streams, embedding).astype(dtype)
    flat_f32 = x_f32.reshape(tokens, flattened_size)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True) + rms_epsilon
    )
    normalized = flat_f32 * inv_rms * norm_scale

  with jax.named_scope("h_pre_gate"):
    h_pre = (
        jax.nn.sigmoid(_gate_logits(normalized, w_pre, alpha_pre, b_pre))
        + pre_mapping_epsilon
    )
  with jax.named_scope("stream_collapse"):
    layer_input = (
        jnp.sum(h_pre[:, :, None] * x_f32, axis=1)
        .astype(x.dtype)
        .reshape(layer_input_shape)
    )

  with jax.named_scope("h_post_gate"):
    h_post = 2.0 * jax.nn.sigmoid(
        _gate_logits(normalized, w_post, alpha_post, b_post)
    )
  with jax.named_scope("residual_bvn_mix"):
    weights_res = jax.nn.softmax(
        _gate_logits(normalized, w_res, alpha_res, b_res),
        axis=-1,
    )
    residual = jnp.einsum("tp,pkj->tjk", weights_res, permutations)
  return layer_input, x.reshape(tokens, flattened_size), h_post, residual, h_pre


def _post_xla(layer_output, x, h_post, residual):
  """Reference JAX-only implementation of the mHC-lite post-branch stage."""
  with jax.named_scope("mhc_post_xla"):
    embedding = layer_output.shape[-1]
    streams = h_post.shape[-1]
    tokens = (
        layer_output.shape[0]
        if layer_output.ndim == 2
        else layer_output.shape[0] * layer_output.shape[1]
    )
    out_shape = (
        (tokens, streams * embedding)
        if layer_output.ndim == 2
        else (*layer_output.shape[:2], streams, embedding)
    )
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    x_f32 = x.reshape(tokens, streams, embedding).astype(dtype)
    layer_output_f32 = layer_output.reshape(tokens, embedding).astype(dtype)
    residual = residual.reshape(tokens, streams, streams)
    with jax.named_scope("residual_stream_mix"):
      residual_mix = jnp.einsum("tjk,tkd->tjd", residual, x_f32)
    with jax.named_scope("post_gate_scale"):
      post_mix = h_post[:, :, None] * layer_output_f32[:, None, :]
    with jax.named_scope("combine_post_output"):
      return (residual_mix + post_mix).astype(x.dtype).reshape(out_shape)


def forward(
    layer_output,
    x,
    h_post,
    residual,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
):
  """mHC-lite fused post-branch mixing and next-layer pre-branch gating.

  Args:
    layer_output: Output of the wrapped layer, `[T, E]`.
    x: Residual streams entering the wrapped layer, `[T, S * E]` (stream-major:
      stream `k` is `x[:, k * E:(k + 1) * E]`).
    h_post: Post gate of the wrapped layer, `[T, S]`, float32, values in
      `(0, 2)`.
    residual: Stream mixing matrices of the wrapped layer, `[T, S * S]`
      (row-major `[S, S]`, doubly stochastic).
    norm_scale: RMSNorm channel scale of the next layer, `[S * E]`.
    w_pre: Pre-gate projection, `[S * E, S]`.
    b_pre: Pre-gate bias, `[S]`.
    alpha_pre: Pre-gate logit scale, `[1]`.
    w_post: Post-gate projection, `[S * E, S]`.
    b_post: Post-gate bias, `[S]`.
    alpha_post: Post-gate logit scale, `[1]`.
    w_res: Residual-mix projection, `[S * E, P]`.
    b_res: Residual-mix bias, `[P]`.
    alpha_res: Residual-mix logit scale, `[1]`.
    permutations: All `S x S` permutation matrices, `[P, S, S]`, float32.

  Returns:
    `(layer_input, x_next, h_post_next, residual_next)`:
      layer_input    `[T, E]`      input of the next wrapped layer, dtype of `x`
      x_next         `[T, S * E]`  mixed residual streams, dtype of `x`
      h_post_next    `[T, S]`      post gate of the next layer, float32
      residual_next  `[T, S * S]`  mixing matrices of the next layer, dtype of
                                   `x`
  """
  with jax.named_scope("mhc_post_and_pre_xla"):
    x_next = _post_xla(layer_output, x, h_post, residual)
    layer_input, x_2d, h_post_next, residual_f32, _ = _pre_xla_raw(
        x_next,
        norm_scale,
        w_pre,
        b_pre,
        alpha_pre,
        w_post,
        b_post,
        alpha_post,
        w_res,
        b_res,
        alpha_res,
        permutations,
        rms_epsilon=_RMS_EPSILON,
        pre_mapping_epsilon=_PRE_MAPPING_EPSILON,
    )
    tokens = x_2d.shape[0]
    residual_next = residual_f32.astype(x_next.dtype).reshape(tokens, -1)
    return layer_input, x_2d, h_post_next, residual_next


# ==============================================================================
# Pallas TPU forward kernel: shared data structures, tiling and cost models
# (inlined from tokamax/_src/ops/experimental/mhc_lite/common.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================


@contextlib.contextmanager
def tpu_mesh_context():
  """Ensures a valid TPU AbstractMesh is available when running on CPU / interpret mode."""
  needs_abstract_mesh = False
  try:
    if not pltpu.is_tpu_device():
      needs_abstract_mesh = True
  except (RuntimeError, ValueError, AttributeError, TypeError):
    needs_abstract_mesh = True

  if needs_abstract_mesh:
    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def get_tpu_info() -> pltpu.TpuInfo:
  """Returns the active TPU hardware info, using an abstract TPU mesh on CPU."""
  with tpu_mesh_context():
    return pltpu.get_tpu_info()


HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)


class UnsupportedInputError(ValueError):
  """Known Mosaic shape, dtype, or tiling restriction."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MHCContext:
  """Opaque token-local context passed from `pre` to `post`."""

  x: jax.Array
  h_post: jax.Array
  residual: jax.Array
  implementation: str = dataclasses.field(metadata={"static": True})


_POSITIVE_CONFIG_FIELDS = (
    "block_size",
    "fwd_pipeline_blocks",
    "fwd_sub_tokens",
    "fwd_strip_tokens",
    "fwd_band_width",
    "bwd_block_size",
    "bwd_num_chunks",
    "bwd_sub_tokens",
    "vmem_limit_bytes",
)


@dataclasses.dataclass(frozen=True)
class MhcKernelConfig:
  """Compiler and tiling configuration for mHC Pallas kernels."""

  block_size: int | None = None
  fwd_pipeline_blocks: int | None = None
  fwd_sub_tokens: int | None = None
  fwd_strip_tokens: int | None = None
  fwd_band_width: int | None = None
  bwd_block_size: int | None = None
  bwd_num_chunks: int | None = None
  bwd_sub_tokens: int | None = None
  vmem_limit_bytes: int | None = None
  rms_epsilon: float = 1e-5
  pre_mapping_epsilon: float = 1e-6
  interpret: bool = False


def _default(val: int | None, fallback: int) -> int:
  """Returns `fallback` when `val` is `None`, otherwise `val`."""
  return fallback if val is None else val


def chunk_tokens(tokens: int, block_size: int, fwd_pipeline_blocks: int) -> int:
  """Returns `block_size * fwd_pipeline_blocks` when dividing `tokens`."""
  chunk = block_size * fwd_pipeline_blocks
  if fwd_pipeline_blocks > 1 and tokens % chunk == 0:
    return chunk
  return block_size


def validate_config_fields(
    config: object,
    num_lanes: int,
    num_sublanes: int,
    *,
    check_fwd: bool = True,
    check_bwd: bool = True,
) -> None:
  """Validates positive and alignment constraints on config fields."""
  for name in _POSITIVE_CONFIG_FIELDS:
    if (val := getattr(config, name, None)) is not None and val < 1:
      raise UnsupportedInputError(
          f"Expected positive {name}; got {val}."
      )
  aligned_fields: list[tuple[str, int]] = []
  if check_fwd:
    aligned_fields.extend((
        ("fwd_sub_tokens", num_sublanes),
        ("fwd_strip_tokens", num_sublanes),
        ("fwd_band_width", num_lanes),
    ))
  if check_bwd:
    aligned_fields.append(("bwd_sub_tokens", num_sublanes))
  for name, align in aligned_fields:
    if (val := getattr(config, name, None)) is not None and val % align:
      raise UnsupportedInputError(
          f"Expected {name} to be a positive multiple of {align}; got {val}."
      )


def token_block_candidates(
    tokens: int,
    num_lanes: int,
    num_sublanes: int,
    multipliers: int | tuple[int, ...] = (1, 2, 3, 4, 6, 8),
) -> list[int]:
  """Returns lane-aligned or sublane-aligned block divisors of `tokens`."""
  if tokens <= 0:
    return [num_lanes]
  mults = (
      range(1, multipliers + 1)
      if isinstance(multipliers, int)
      else multipliers
  )
  lane_divs = [
      mult * num_lanes for mult in mults if tokens % (mult * num_lanes) == 0
  ]
  if lane_divs:
    return lane_divs
  sublane_divs = [
      mult * num_sublanes
      for mult in range(2 * num_lanes // num_sublanes, 0, -1)
      if tokens % (mult * num_sublanes) == 0
  ]
  return sublane_divs or [num_lanes]


def resolve_config(
    config: MhcKernelConfig,
    *,
    tokens: int | None = None,
    streams: int = 4,
    embedding: int = 512,
    num_permutations: int = 24,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    check_fwd: bool = True,
    check_bwd: bool = True,
) -> MhcKernelConfig:
  """Resolves `None` tiling and VMEM fields with safe hardware defaults."""
  del streams, num_permutations, dtype
  tpu_info = get_tpu_info()
  num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
  validate_config_fields(
      config, num_lanes, num_sublanes, check_fwd=check_fwd, check_bwd=check_bwd
  )
  vmem_limit = _default(config.vmem_limit_bytes, tpu_info.vmem_capacity_bytes)
  wide_or_interpret = config.interpret or embedding > 1024
  if tokens is None or tokens <= 0:
    block_size = _default(config.block_size, num_lanes)
    fwd_pipe = _default(config.fwd_pipeline_blocks, 2)
    chunk = block_size * fwd_pipe
    bwd_block_size = _default(config.bwd_block_size, 2 * num_lanes)
    num_bwd_blocks = 4
  else:
    cands = token_block_candidates(tokens, num_lanes, num_sublanes, (1, 2))
    default_blk = min(cands) if wide_or_interpret else max(cands)
    block_size = _default(config.block_size, default_blk)
    default_pipe = (
        2
        if not wide_or_interpret and tokens % (2 * block_size) == 0
        else 1
    )
    fwd_pipe = _default(config.fwd_pipeline_blocks, default_pipe)
    chunk = chunk_tokens(tokens, block_size, fwd_pipe)
    bwd_block_size = _default(config.bwd_block_size, default_blk)
    num_bwd_blocks = max(1, tokens // max(1, bwd_block_size))

  max_strip_elems = 12 * num_lanes * 512
  if config.interpret or chunk * embedding > max_strip_elems:
    default_strip = math.gcd(chunk, num_lanes // 2) or num_sublanes
  else:
    default_strip = chunk
  default_fwd_sub = math.gcd(chunk, 2 * num_sublanes) or num_sublanes
  default_band = math.gcd(embedding, 2 * num_lanes) or num_lanes
  default_bwd_sub = math.gcd(bwd_block_size, 2 * num_sublanes) or num_sublanes
  default_bwd_chunks = math.gcd(num_bwd_blocks, 4)
  return dataclasses.replace(
      config,
      block_size=block_size,
      fwd_pipeline_blocks=fwd_pipe,
      fwd_sub_tokens=_default(config.fwd_sub_tokens, default_fwd_sub),
      fwd_strip_tokens=_default(config.fwd_strip_tokens, default_strip),
      fwd_band_width=_default(config.fwd_band_width, default_band),
      bwd_block_size=bwd_block_size,
      bwd_num_chunks=_default(config.bwd_num_chunks, default_bwd_chunks),
      bwd_sub_tokens=_default(config.bwd_sub_tokens, default_bwd_sub),
      vmem_limit_bytes=vmem_limit,
  )


def prepare_kernel_config(
    config: Any,
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int,
    dtype: jax.typing.DTypeLike,
    *,
    is_bwd: bool = False,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[Any, float, float, int, bool]:
  """Validates `config` and resolves concrete kernel execution parameters."""
  check_fwd, check_bwd = (False, True) if is_bwd else (True, False)
  if isinstance(config, MhcKernelConfig):
    config = resolve_config(
        config,
        tokens=tokens,
        streams=streams,
        embedding=embedding,
        num_permutations=num_permutations,
        dtype=dtype,
        check_fwd=check_fwd,
        check_bwd=check_bwd,
    )
  else:
    tpu_info = get_tpu_info()
    validate_config_fields(
        config,
        tpu_info.num_lanes,
        tpu_info.num_sublanes,
        check_fwd=check_fwd,
        check_bwd=check_bwd,
    )
  if rms_epsilon is None:
    rms_epsilon = getattr(config, "rms_epsilon", 1e-5)
  if pre_mapping_epsilon is None:
    pre_mapping_epsilon = getattr(config, "pre_mapping_epsilon", 1e-6)
  if vmem_limit_bytes is None:
    vmem_limit_bytes = (
        getattr(config, "vmem_limit_bytes", None)
        or get_tpu_info().vmem_capacity_bytes
    )
  resolved_interpret = interpret or getattr(config, "interpret", False)
  return (
      config,
      rms_epsilon,
      pre_mapping_epsilon,
      vmem_limit_bytes,
      resolved_interpret,
  )


def pre_cost(
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int = 24,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for `pre` forward or backward (`T=tokens, k=streams, d=emb, P=perms, C=2k+P`).

  Returns:
    pl.CostEstimate with:
    - FLOPs (fwd): `2*T*(k*d)*C` (x @ phi) + `2*T*P*k^2` (w_res @ P) + `2*T*k*d`
      (h_pre * x); backward scales by 2x (recompute + VJP GEMMs and
      d_x/d_h_pre).
    - Transcendentals: `T*C` (sigmoid + softmax).
    - Bytes: `2*T*k*d` (read bf16 x; 3x in bwd for x, d_x_acc, write d_x) +
      `4*(k*d)*C` (read f32 phi; 2x in bwd for read phi, write d_phi).
  """
  flat = streams * embedding
  phi_cols = 2 * streams + num_permutations
  scale = 2 if is_bwd else 1
  x_reads = 3 if is_bwd else 1
  return pl.CostEstimate(
      flops=scale
      * 2
      * tokens
      * (flat * (phi_cols + 1) + num_permutations * streams * streams),
      transcendentals=tokens * phi_cols,
      bytes_accessed=x_reads * 2 * tokens * flat + scale * 4 * flat * phi_cols,
  )


def post_cost(
    tokens: int,
    streams: int,
    embedding: int,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for `post` forward or backward (`T=tokens, k=streams, d=emb`).

  Returns:
    pl.CostEstimate with:
    - FLOPs (fwd): `2*T*k^2*d` (residual @ x) + `2*T*k*d` (h_post * layer_out +
      res_mix); backward scales by 2x (VJP of residual^T @ d_out, d_layer_out,
      d_h_post, d_res).
    - Transcendentals: `0`.
    - Bytes: `4*T*k*d` (read/write bf16 x, out; `6*T*k*d` in bwd for read x,
      d_out, write d_x) + `4*T*d` (read layer_out + gates; `8*T*d` in bwd for
      read/write layer_out, d_layer_out).
  """
  scale = 2 if is_bwd else 1
  x_reads = 3 if is_bwd else 2
  return pl.CostEstimate(
      flops=scale * 2 * tokens * streams * (streams + 1) * embedding,
      transcendentals=0,
      bytes_accessed=x_reads * 2 * tokens * streams * embedding
      + scale * 4 * tokens * embedding,
  )


def post_and_pre_cost(
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int = 24,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for fused `post_and_pre` forward or backward (`T=tokens, k=streams, d=emb`).

  Returns:
    pl.CostEstimate summing `post_cost` and `pre_cost` while subtracting the
    avoided HBM roundtrip (`2*T*(k-1)*d` in fwd for in-VMEM `x_next` reuse, or
    `4*T*k*d` in bwd for fused `d_x_next` handoff).
  """
  post = post_cost(tokens, streams, embedding, is_bwd=is_bwd)
  pre = pre_cost(tokens, streams, embedding, num_permutations, is_bwd=is_bwd)
  saved_bytes = (
      4 * tokens * streams * embedding
      if is_bwd
      else 2 * tokens * (streams - 1) * embedding
  )
  return pl.CostEstimate(
      flops=post.flops + pre.flops,
      transcendentals=post.transcendentals + pre.transcendentals,
      bytes_accessed=post.bytes_accessed + pre.bytes_accessed - saved_bytes,
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcWeights:
  """Structured layer weights container for mHC-lite."""

  norm_scale: jax.Array
  w_pre: jax.Array
  b_pre: jax.Array
  alpha_pre: jax.Array
  w_post: jax.Array
  b_post: jax.Array
  alpha_post: jax.Array
  w_res: jax.Array
  b_res: jax.Array
  alpha_res: jax.Array

  def fold_norm_scale(self) -> jax.Array:
    return fold_norm_scale(self.norm_scale, self.w_pre, self.w_post, self.w_res)


def _token_index_map(num_trailing_dims: int, i):
  """Index map tiling the leading token dimension."""
  return (i,) + (0,) * num_trailing_dims


def token_block_spec(shape: tuple[int, ...], block_size: int) -> pl.BlockSpec:
  """Returns a BlockSpec tiling the leading token dimension."""
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(
      block_shape, functools.partial(_token_index_map, len(shape) - 1)
  )


def fold_norm_scale(norm_scale, w_pre, w_post, w_res) -> jax.Array:
  """Folds the RMSNorm channel scale into the three projection matrices."""
  with jax.named_scope("fold_norm_scale"):
    weights = jnp.concatenate((w_pre, w_post, w_res), axis=-1).astype(
        jnp.float32
    )
    return norm_scale.astype(jnp.float32)[:, None] * weights


def validate_token_block_size(
    tokens: int, block_size: int, *, name: str = "block_size"
) -> None:
  """Validates a token-axis Pallas block size."""
  if tokens <= 0:
    raise UnsupportedInputError(f"Expected positive token count; got {tokens}.")
  num_sublanes = get_tpu_info().num_sublanes
  if block_size < num_sublanes or block_size % num_sublanes:
    raise UnsupportedInputError(
        f"Expected {name} to be a positive multiple of {num_sublanes}; got"
        f" {block_size}."
    )
  if tokens % block_size:
    raise UnsupportedInputError(
        f"The per-device token count ({tokens}) must be divisible by {name}"
        f" ({block_size})."
    )


def validate_inputs(
    x: jax.Array,
    block_size: int | None = None,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    block_size_name: str = "block_size",
) -> None:
  """Validates the shape, dtype, and forward token block constraints."""
  if x.dtype != jnp.bfloat16:
    raise UnsupportedInputError(
        f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}."
    )
  if x.ndim == 2:
    tokens, flattened_size = x.shape
    streams = 4
    if flattened_size % streams:
      raise UnsupportedInputError(
          "Expected 2D x to have trailing dimension divisible by 4; got"
          f" {x.shape}."
      )
    embedding = flattened_size // streams
  elif x.ndim == 4:
    batch, sequence, streams, embedding = x.shape
    tokens = batch * sequence
  else:
    raise UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding) or"
        f" (tokens, streams * embedding); got {x.shape}."
    )
  if streams != 4 or (
      permutations_shape is not None and permutations_shape != (24, 4, 4)
  ):
    raise UnsupportedInputError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with"
        f" expansion rate 4 only; got x.shape={x.shape} and"
        f" permutations.shape={permutations_shape}."
    )
  tpu_info = get_tpu_info()
  if embedding % tpu_info.num_lanes:
    raise UnsupportedInputError(
        f"Expected embedding dim to be a multiple of {tpu_info.num_lanes}; got"
        f" {embedding}."
    )
  resolved_block_size = tpu_info.num_lanes if block_size is None else block_size
  validate_token_block_size(tokens, resolved_block_size, name=block_size_name)


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/experimental/mhc_lite/mhc_kernels_fwd.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================


def _full_param_block_spec(param: jax.Array) -> pl.BlockSpec:
  """Creates an unsharded BlockSpec covering the full parameter shape."""
  return pl.BlockSpec(param.shape, lambda *_: (0,) * param.ndim)


def _prepare_fwd_params(
    weights: MhcWeights,
    permutations: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  """Folds `phi`, pre-scales by `alpha`, concatenates biases, and pads permutations."""
  with jax.named_scope("prepare_fwd_params"):
    phi = weights.fold_norm_scale()
    num_permutations, streams, _ = permutations.shape
    with jax.named_scope("scale_phi_by_alpha"):
      alpha_vec = jnp.concatenate(
          [
              jnp.broadcast_to(a, (n,))
              for a, n in (
                  (weights.alpha_pre, streams),
                  (weights.alpha_post, streams),
                  (weights.alpha_res, num_permutations),
              )
          ],
          axis=0,
      )[None, :]
      phi_scaled = phi * alpha_vec
    with jax.named_scope("pack_gate_bias"):
      gate_bias = jnp.concatenate(
          (weights.b_pre, weights.b_post, weights.b_res),
          axis=0,
          dtype=jnp.float32,
      )[None, :]
    with jax.named_scope("pad_permutations_32_lanes"):
      permutations_2d = (
          permutations.astype(jnp.float32)
          .swapaxes(-1, -2)
          .reshape(num_permutations, streams * streams)
      )
      permutations_padded = jnp.pad(permutations_2d, ((2 * streams, 0), (0, 0)))
    return phi, (phi_scaled, gate_bias, permutations_padded)


def _run_fwd_pipeline(
    scope_name: str,
    body_fn,
    tokens: int,
    chunk_tokens: int,
    token_args: tuple[jax.Array, ...],
    out_shapes_dtypes: tuple[
        tuple[tuple[int, ...], jax.typing.DTypeLike], ...
    ],
    cost_estimate: pl.CostEstimate,
    *,
    vmem_limit_bytes: int,
    interpret: bool,
    param_args: tuple[jax.Array, ...] = (),
):
  """Emits and executes a 1D token-pipelined Pallas forward kernel."""
  in_args = token_args + param_args
  in_specs = tuple(
      token_block_spec(arg.shape, chunk_tokens) for arg in token_args
  ) + tuple(_full_param_block_spec(arg) for arg in param_args)
  out_specs = tuple(
      token_block_spec(shape, chunk_tokens)
      for shape, _ in out_shapes_dtypes
  )
  out_shape = tuple(
      jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in out_shapes_dtypes
  )
  kernel_main = pltpu.emit_pipeline(
      lambda unused_step, *refs: body_fn(*refs),
      grid=(tokens // chunk_tokens,),
      in_specs=in_specs,
      out_specs=out_specs,
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  with tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        name=scope_name,
        out_shape=out_shape,
        in_specs=(HBM_SPEC,) * len(in_args),
        out_specs=(HBM_SPEC,) * len(out_shapes_dtypes),
        cost_estimate=cost_estimate,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(*in_args)


def _pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    pre_mapping_epsilon: float,
    fwd_strip_tokens: int,
    fwd_band_width: int,
    x_ref,
    phi_ref,
    gate_bias_ref,
    permutations_ref,
    layer_input_ref,
    h_pre_ref,
    h_post_ref,
    residual_ref,
) -> None:
  """Pallas pipeline body for fused pre-branch forward kernel."""
  with jax.named_scope("load_params_and_x"):
    gate_bias = gate_bias_ref[...]
    permutations = permutations_ref[...]
    phi = phi_ref[...]
    x_full = x_ref[...]

  # Full-chunk MXU projection overlapped with hardware bf16 sum-of-squares
  # reduction on VPU (opcode 81), sharing the bfloat16 vector registers of x.
  with jax.named_scope("mxu_proj_and_rms_norm"):
    with jax.named_scope("mxu_gate_proj"):
      raw_proj = jnp.dot(x_full, phi, preferred_element_type=jnp.float32)
    with jax.named_scope("vpu_rms_norm"):
      flattened_f32 = x_full.astype(jnp.float32)
      mean_square = jnp.mean(jnp.square(flattened_f32), axis=-1, keepdims=True)
      inv_rms = jax.lax.rsqrt(mean_square + rms_epsilon)
      logits = raw_proj * inv_rms + gate_bias

  # Fused pre/post sigmoid on lanes 0..2*S-1.
  with jax.named_scope("sigmoid_pre_post_gates"):
    gates = jax.nn.sigmoid(logits[:, : 2 * streams])
    h_pre = gates[:, :streams] + pre_mapping_epsilon
    h_post = 2.0 * gates[:, streams : 2 * streams]

  # Zero-shift 32-lane softmax + padded (32, S^2) permutation projection:
  # Masking lanes 0..2*S-1 with -inf avoids cross-lane shifts (XLU opcode 58).
  with jax.named_scope("softmax_bvn_residual_mix"):
    col_idx = jax.lax.broadcasted_iota(jnp.int32, logits.shape, dimension=1)
    res_logits = jnp.where(col_idx >= 2 * streams, logits, -jnp.inf)
    res_weights = jax.nn.softmax(res_logits, axis=-1)
    residual = jnp.dot(
        res_weights,
        permutations,
        preferred_element_type=jnp.float32,
    )

  with jax.named_scope("store_gates_and_residual"):
    h_pre_ref[...] = h_pre
    h_post_ref[...] = h_post
    residual_ref[...] = residual.astype(residual_ref.dtype)

  # Strip-mined stream collapse over token strips and column bands to bound
  # live float32 vector registers and prevent VMEM register spills.
  with jax.named_scope("strip_mined_stream_collapse"):
    strip_tokens = math.gcd(x_full.shape[0], fwd_strip_tokens)
    num_strips = x_full.shape[0] // strip_tokens
    band_width = math.gcd(embedding, fwd_band_width)

    def _collapse_bands(t_slice, h_strip) -> None:
      with jax.named_scope("stream_collapse_strip"):
        for band_col in range(0, embedding, band_width):
          band_end = band_col + band_width
          acc = h_strip[:, 0:1] * x_ref[t_slice, band_col:band_end]
          for k in range(1, streams):
            col = k * embedding + band_col
            acc = (
                acc
                + h_strip[:, k : k + 1] * x_ref[t_slice, col : col + band_width]
            )
          layer_input_ref[t_slice, band_col:band_end] = acc.astype(jnp.bfloat16)

    if num_strips == 1:
      _collapse_bands(slice(None), h_pre)
    else:
      jax.lax.fori_loop(
          0,
          num_strips,
          lambda strip_idx, _: _collapse_bands(
              pl.dslice(strip_idx * strip_tokens, strip_tokens),
              h_pre_ref[pl.dslice(strip_idx * strip_tokens, strip_tokens), :],
          ),
          None,
      )


def _post_apply_fwd_pipeline_body(
    streams: int,
    embedding: int,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    output_ref,
) -> None:
  """Subtiled VPU pipeline body for 2D post-branch stream mixing."""
  with jax.named_scope("post_stream_mix_subtiles"):
    for sub_idx in range(num_subtiles):
      with jax.named_scope(f"subtile_{sub_idx}"):
        s_slice = slice(None)
        if num_subtiles > 1:
          s_slice = slice(sub_idx * sub_tokens, (sub_idx + 1) * sub_tokens)
        with jax.named_scope("load_subtile_inputs"):
          res_sub = residual_ref[s_slice, :].astype(jnp.float32)
          h_post_sub = h_post_ref[s_slice, :]
          l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
          x_sub = [
              x_ref[s_slice, k * embedding : (k + 1) * embedding].astype(
                  jnp.float32
              )
              for k in range(streams)
          ]
        with jax.named_scope("mix_and_store_streams"):
          for j in range(streams):
            out_j = h_post_sub[:, j : j + 1] * l_sub
            for k in range(streams):
              out_j = (
                  out_j
                  + res_sub[:, j * streams + k : j * streams + k + 1] * x_sub[k]
              )
            output_ref[s_slice, j * embedding : (j + 1) * embedding] = (
                out_j.astype(output_ref.dtype)
            )


def _post_and_pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    pre_mapping_epsilon: float,
    fwd_strip_tokens: int,
    fwd_band_width: int,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    phi_ref,
    gate_bias_ref,
    permutations_ref,
    layer_input_ref,
    x_next_out_ref,
    *out_refs,
) -> None:
  """Pallas pipeline body for fused Stage-1 post + Stage-2 pre forward kernel."""
  with jax.named_scope("stage1_post_apply_fwd"):
    _post_apply_fwd_pipeline_body(
        streams,
        embedding,
        sub_tokens,
        num_subtiles,
        x_ref,
        layer_output_ref,
        h_post_ref,
        residual_ref,
        x_next_out_ref,
    )
  with jax.named_scope("stage2_pre_fwd"):
    _pre_fwd_pipeline_body(
        streams,
        embedding,
        rms_epsilon,
        pre_mapping_epsilon,
        fwd_strip_tokens,
        fwd_band_width,
        x_next_out_ref,
        phi_ref,
        gate_bias_ref,
        permutations_ref,
        layer_input_ref,
        *out_refs,
    )


def _prepare_post_fwd(
    layer_output: jax.Array,
    context: MHCContext,
    config: Any,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Resolves config, validates shapes, and packs post-branch forward args."""
  x, h_post, residual = context.x, context.h_post, context.residual
  embedding = layer_output.shape[-1]
  tokens = layer_output.size // embedding
  streams = h_post.shape[-1]
  num_perms = permutations_shape[0] if permutations_shape else 24
  config, rms_eps, pre_eps, vmem_limit, interpret_flag = (
      prepare_kernel_config(
          config,
          tokens,
          streams,
          embedding,
          num_perms,
          x.dtype,
          rms_epsilon=rms_epsilon,
          pre_mapping_epsilon=pre_mapping_epsilon,
          vmem_limit_bytes=vmem_limit_bytes,
          interpret=interpret,
      )
  )
  x_in = x.reshape(tokens, streams * embedding)
  x_check = x_in
  if layer_output.ndim == 3:
    x_check = x.reshape(*layer_output.shape[:2], streams, embedding)
  validate_inputs(x_check, config.block_size, permutations_shape)
  chunk = chunk_tokens(
      tokens, config.block_size, config.fwd_pipeline_blocks
  )
  sub_tokens = math.gcd(chunk, config.fwd_sub_tokens)
  token_args = (
      x_in,
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams * streams),
  )
  return (
      config,
      rms_eps,
      pre_eps,
      vmem_limit,
      interpret_flag,
      tokens,
      streams,
      embedding,
      chunk,
      sub_tokens,
      token_args,
  )


def post_and_pre_fwd(
    layer_output: jax.Array,
    context: MHCContext,
    next_weights: MhcWeights,
    permutations: jax.Array,
    config: Any,
    *,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Forward execution for `post_and_pre`, returning outputs and residuals."""
  (
      config,
      rms_eps,
      pre_eps,
      vmem_limit,
      interpret_flag,
      tokens,
      streams,
      embedding,
      chunk,
      sub_tokens,
      token_args,
  ) = _prepare_post_fwd(
      layer_output,
      context,
      config,
      permutations.shape,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  phi, param_args = _prepare_fwd_params(next_weights, permutations)
  next_layer_input, x_next, next_h_pre, next_h_post, next_residual = (
      _run_fwd_pipeline(
          "mhc_post_and_pre_fwd",
          functools.partial(
              _post_and_pre_fwd_pipeline_body,
              streams,
              embedding,
              rms_eps,
              pre_eps,
              config.fwd_strip_tokens,
              config.fwd_band_width,
              sub_tokens,
              chunk // sub_tokens,
          ),
          tokens,
          chunk,
          token_args=token_args,
          param_args=param_args,
          out_shapes_dtypes=(
              ((tokens, embedding), jnp.bfloat16),
              (token_args[0].shape, jnp.bfloat16),
              ((tokens, streams), jnp.float32),
              ((tokens, streams), jnp.float32),
              ((tokens, streams * streams), jnp.bfloat16),
          ),
          cost_estimate=post_and_pre_cost(
              tokens, streams, embedding, permutations.shape[0]
          ),
          vmem_limit_bytes=vmem_limit,
          interpret=interpret_flag,
      )
  )
  next_context = MHCContext(
      x=x_next,
      h_post=next_h_post,
      residual=next_residual,
      implementation="mosaic_tpu",
  )
  return (next_layer_input.reshape(layer_output.shape), next_context), (
      (phi, next_h_pre),
      (layer_output, context),
      (x_next, next_weights, permutations),
  )


# ==============================================================================
# Pallas TPU forward kernel: op config and tiling heuristic
# (inlined from tokamax/_src/ops/experimental/mhc_lite/pallas_mosaic_tpu.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PostAndPreConfig:
  """Autotuning config for `PallasTpuMhcLitePostAndPre`."""

  block_size: int
  fwd_pipeline_blocks: int = 2
  fwd_sub_tokens: int
  fwd_strip_tokens: int
  fwd_band_width: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Heuristics:
  block_size: int
  fwd_pipeline_blocks: int
  fwd_sub_tokens: int
  fwd_strip_tokens: int
  fwd_band_width: int
  bwd_block_size: int
  bwd_num_chunks: int
  bwd_sub_tokens: int


def _to_op_config(cfg_cls: type[Any], resolved: Any) -> Any:
  return cfg_cls(
      **{f.name: getattr(resolved, f.name) for f in dataclasses.fields(cfg_cls)}
  )


def _heuristics_from_dims(
    tokens: int,
    streams: int,
    embedding: int,
    dtype: jax.typing.DTypeLike,
    interpret: bool,
) -> _Heuristics:
  """Resolves safe heuristic defaults for the given op dimensions."""
  resolved = resolve_config(
      MhcKernelConfig(interpret=interpret),
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      dtype=dtype,
  )
  assert (
      resolved.block_size is not None
      and resolved.fwd_pipeline_blocks is not None
      and resolved.fwd_sub_tokens is not None
      and resolved.fwd_strip_tokens is not None
      and resolved.fwd_band_width is not None
      and resolved.bwd_block_size is not None
      and resolved.bwd_num_chunks is not None
      and resolved.bwd_sub_tokens is not None
  )
  return _Heuristics(
      block_size=resolved.block_size,
      fwd_pipeline_blocks=resolved.fwd_pipeline_blocks,
      fwd_sub_tokens=resolved.fwd_sub_tokens,
      fwd_strip_tokens=resolved.fwd_strip_tokens,
      fwd_band_width=resolved.fwd_band_width,
      bwd_block_size=resolved.bwd_block_size,
      bwd_num_chunks=resolved.bwd_num_chunks,
      bwd_sub_tokens=resolved.bwd_sub_tokens,
  )


def forward_pallas(
    layer_output,
    x,
    h_post,
    residual,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
):
  """mHC-lite fused post-and-pre forward pass on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(out, residuals)` with
  `out = (layer_input, x_next, h_post_next, residual_next)` and
  `residuals = (phi, h_pre)`; see the module docstring. Block sizes are the
  Tokamax heuristic for the input shape. Off TPU the kernel runs in Pallas
  interpret mode.
  """
  interpret = jax.default_backend() != "tpu"
  tokens, flat = x.shape
  streams = h_post.shape[-1]
  embedding = flat // streams
  config = _to_op_config(
      PostAndPreConfig,
      _heuristics_from_dims(tokens, streams, embedding, x.dtype, interpret),
  )
  next_weights = MhcWeights(
      norm_scale=norm_scale,
      w_pre=w_pre,
      b_pre=b_pre,
      alpha_pre=alpha_pre,
      w_post=w_post,
      b_post=b_post,
      alpha_post=alpha_post,
      w_res=w_res,
      b_res=b_res,
      alpha_res=alpha_res,
  )
  ctx = MHCContext(
      x=x, h_post=h_post, residual=residual, implementation="mosaic_tpu"
  )
  (next_li, next_ctx), (residuals, _, _) = post_and_pre_fwd(
      layer_output,
      ctx,
      next_weights,
      permutations,
      config,
      rms_epsilon=_RMS_EPSILON,
      pre_mapping_epsilon=_PRE_MAPPING_EPSILON,
      vmem_limit_bytes=None,
      interpret=interpret,
  )
  out = (next_li, next_ctx.x, next_ctx.h_post, next_ctx.residual)
  return out, residuals


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs (primary): t8192_s4_e512. The default: it is the
    # largest primary spec for which the memory plan of the pure-JAX `workload`
    # stays small. XLA plans 0.75 GiB of temporaries and 0.08 GiB of arguments
    # for it. The memory figures in this table are XLA estimates from compiling
    # `jax.jit(workload)` on CPU in bfloat16, not TPU measurements.
    'mhc_lite_t8192_s4_e512': {
        'name': 'mhc_lite_t8192_s4_e512_bwd_pallas_fwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 8192,
        'streams': 4,
        'embedding': 512,
    },
    # tokamax arg_specs (primary): t196608_s4_e512, i.e. batch 128 x sequence
    # 1536, the shape of the Tokamax mHC-lite benchmark. Not the default
    # because of its memory plan: 18.0 GiB of temporaries and 1.9 GiB of
    # arguments (the plan grows linearly with the token count).
    'mhc_lite_t196608_s4_e512': {
        'name': 'mhc_lite_t196608_s4_e512_bwd_pallas_fwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 196608,
        'streams': 4,
        'embedding': 512,
    },
    # tokamax arg_specs (primary, ci_tests): t512_s4_e256. 0.02 GiB of
    # temporaries.
    'mhc_lite_t512_s4_e256': {
        'name': 'mhc_lite_t512_s4_e256_bwd_pallas_fwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 512,
        'streams': 4,
        'embedding': 256,
    },
    # Tiny shapes for CPU smoke tests.
    'smoke': {
        'name': 'mhc_lite_smoke_bwd_pallas_fwd',
        'model': 'mHC-lite',
        'operator': 'mhc_lite_post_and_pre_backward',
        'tokens': 256,
        'streams': 4,
        'embedding': 512,
    },
}
CONFIG = CONFIGS['mhc_lite_t8192_s4_e512']


def _select_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns the 15 arguments of `forward` followed by the 4 output cotangents.

  `(layer_output, x, h_post, residual, norm_scale, w_pre, b_pre, alpha_pre,
  w_post, b_post, alpha_post, w_res, b_res, alpha_res, permutations,
  d_layer_input, d_x_next, d_h_post_next, d_residual_next)`.
  """
  cfg = _select_config(config)
  T, S, E = cfg['tokens'], cfg['streams'], cfg['embedding']
  P = math.factorial(S)
  keys = jax.random.split(jax.random.key(42), 16)

  # Sample in float32 and cast, so every dtype sees the same values. Each
  # array is cast as soon as it is drawn to bound peak memory.
  def normal(key, shape, scale=1.0, dtype=dtype):
    return (scale * jax.random.normal(key, shape)).astype(dtype)

  permutations = jnp.eye(S, dtype=jnp.float32)[
      jnp.array(list(itertools.permutations(range(S))))
  ]

  # Activations follow the Tokamax mHC-lite tests: unit-normal streams and
  # layer output. `h_post` and `residual` are what a preceding pre stage
  # emits: a `2 * sigmoid` gate and a convex mix of permutation matrices.
  layer_output = normal(keys[0], (T, E))
  x = normal(keys[1], (T, S * E))
  h_post = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (T, S)))
  mix = jax.nn.softmax(jax.random.normal(keys[3], (T, P)), axis=-1)
  residual = (
      jnp.einsum('tp,pkj->tjk', mix, permutations)
      .reshape(T, S * S)
      .astype(dtype)
  )

  # Weights follow the Tokamax mHC-lite tests: unit-normal norm scale,
  # projections and biases scaled by 0.1, logit scales of one.
  norm_scale = normal(keys[4], (S * E,))
  w_pre = normal(keys[5], (S * E, S), 0.1)
  b_pre = normal(keys[6], (S,), 0.1)
  alpha_pre = jnp.ones((1,), dtype)
  w_post = normal(keys[7], (S * E, S), 0.1)
  b_post = normal(keys[8], (S,), 0.1)
  alpha_post = jnp.ones((1,), dtype)
  w_res = normal(keys[9], (S * E, P), 0.1)
  b_res = normal(keys[10], (P,), 0.1)
  alpha_res = jnp.ones((1,), dtype)

  # Output cotangents, scaled by 0.01 as in the Tokamax tests.
  d_layer_input = normal(keys[11], (T, E), 0.01)
  d_x_next = normal(keys[12], (T, S * E), 0.01)
  d_h_post_next = normal(keys[13], (T, S), 0.01, jnp.float32)
  d_residual_next = normal(keys[14], (T, S * S), 0.01)

  return (
      layer_output,
      x,
      h_post,
      residual,
      norm_scale,
      w_pre,
      b_pre,
      alpha_pre,
      w_post,
      b_post,
      alpha_post,
      w_res,
      b_res,
      alpha_res,
      permutations,
      d_layer_input,
      d_x_next,
      d_h_post_next,
      d_residual_next,
  )


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangents, and returns one gradient per entry of `diff_argnums`.
# `permutations` (argument 14) is a constant and is not differentiated.
VJP_SPEC = {
    'num_primals': 15,
    'diff_argnums': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
}


def workload(
    layer_output,
    x,
    h_post,
    residual,
    norm_scale,
    w_pre,
    b_pre,
    alpha_pre,
    w_post,
    b_post,
    alpha_post,
    w_res,
    b_res,
    alpha_res,
    permutations,
    d_layer_input,
    d_x_next,
    d_h_post_next,
    d_residual_next,
):
  """Returns the VJP of `forward` for every argument except `permutations`.

  The result is `(d_layer_output, d_x, d_h_post, d_residual, d_norm_scale,
  d_w_pre, d_b_pre, d_alpha_pre, d_w_post, d_b_post, d_alpha_post, d_w_res,
  d_b_res, d_alpha_res)`, each with the shape and dtype of its primal.
  """
  primals = (
      layer_output,
      x,
      h_post,
      residual,
      norm_scale,
      w_pre,
      b_pre,
      alpha_pre,
      w_post,
      b_post,
      alpha_post,
      w_res,
      b_res,
      alpha_res,
  )
  _, vjp_fn = jax.vjp(lambda *args: forward(*args, permutations), *primals)
  return vjp_fn((d_layer_input, d_x_next, d_h_post_next, d_residual_next))


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per token the forward does: the residual stream mix, an `S x S` by `S x E`
  product (2*S*S*E); the post-gate scale and add (2*S*E); the three gate
  projections of the `S*E` normalised vector onto `2*S + P` columns
  (2*S*E*(2*S + P)); the stream collapse (2*S*E); and the `P`-term mix of
  `S x S` permutation matrices (2*P*S*S). This is the forward FLOP model of
  Tokamax's `post_and_pre_cost`. The VJP runs the forward once and a backward
  that costs about twice the forward, so 3x in total.
  """
  cfg = _select_config(config)
  T, S, E = cfg['tokens'], cfg['streams'], cfg['embedding']
  P = math.factorial(S)
  post = 2 * S * (S + 1) * E
  pre = 2 * (S * E * (2 * S + P + 1) + P * S * S)
  return int(3 * T * (post + pre))


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = _select_config(config)
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
