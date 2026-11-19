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
"""Mixture of Block Attention (MoBA) backward pass -- Pallas TPU forward kernel given.

JAXBench level 3, track 2 (forward-only Pallas kernel -> Pallas backward kernel).

MoBA splits the sequence into blocks of `CHUNK_SIZE` tokens. Every query block
attends causally to its own block and to the `TOPK` earlier blocks whose mean
key has the largest dot product with the block's mean query.

`forward_pallas(q, k, v)` is the Tokamax Pallas TPU forward kernel. It returns
`(out, residuals)` with `residuals = None`: the Tokamax forward saves nothing
for a backward pass. The kernel selects the blocks in JAX, passes the indices
`[B, H, nt, K]` to the kernel as a scalar-prefetch operand, and runs an
online-softmax accumulation over a `(B, H, nt, K + 1)` grid: step 0 of the last
axis is the query block's own (causal) block, steps 1..K the routed blocks.

`workload(q, k, v, dout)` returns the gradients `(dq, dk, dv)` for the output
cotangent `dout`. It is the correctness oracle: JAX autodiff (`jax.vjp`) of the
pure-JAX reference `forward`. This file contains no backward kernel and no
hand-derived backward pass.

The top-k block selection is not differentiable. The selected block indices
are treated as constants (`stop_gradient`), so no gradient flows through the
routing scores; `dq` and `dk` come from the attention logits only. Tokamax's
own VJP for this op does the same: it is `jax.vjp` of the same reference.

Task: write a Pallas TPU backward kernel that pairs with `forward_pallas` so
that your `workload` returns the same gradients as this one. Tokamax ships no
Pallas backward kernel for this op, so there is no hand-written backward to
compare speed against.

Self-contained: depends only on JAX and NumPy.

Source: openxla/tokamax @ b4b6eb35b4f05fc885f8582d360ff10e2cfc0625
  tokamax/_src/ops/experimental/moba/pallas_mosaic_tpu_kernel.py  (forward kernel)
  tokamax/_src/ops/experimental/moba/pallas_mosaic_tpu.py  (kernel arguments)
  tokamax/_src/ops/experimental/moba/base.py       (forward reference)
  tokamax/_src/ops/experimental/moba/arg_specs.py  (shapes)
"""

# Dimensions.
#   B      batch
#   H      number of attention heads
#   T      sequence length
#   D      head dimension
#   tt     query/key block chunk size (chunk_size)
#   nt     grid extent over T: nt = T // tt
#   K      effective number of routed historical blocks per query block

import functools
import json

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Forward reference (from tokamax/_src/ops/experimental/moba/base.py)
# ==============================================================================

# Tokamax defaults for `chunk_size` and `topk`; every entry of arg_specs.py
# uses them.
CHUNK_SIZE = 256
TOPK = 4


def _single_seq_moba(q_s, k_s, v_s, scale, topk, chunk_size=256):
  """Computes reference MoBA attention for a single sequence."""
  orig_dtype = q_s.dtype
  q_s = q_s.astype(jnp.float32)
  k_s = k_s.astype(jnp.float32)
  v_s = v_s.astype(jnp.float32)

  H, T, D = q_s.shape
  tt = chunk_size
  nt = T // tt
  K = min(topk, nt)
  tk_sub = (K + 1) * tt

  q_blocks = q_s.reshape(H, nt, tt, D)
  k_blocks = k_s.reshape(H, nt, tt, D)
  v_blocks = v_s.reshape(H, nt, tt, D)

  q_mean = jnp.mean(q_blocks, axis=2)
  k_mean = jnp.mean(k_blocks, axis=2)

  routing_scores = (
      jnp.einsum(
          "hqd,hnd->hqn",
          q_mean,
          k_mean,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  q_idx = jnp.arange(nt)[:, None]
  k_idx = jnp.arange(nt)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(historical_mask[None], routing_scores, -jnp.inf)
  _, topk_idx = jax.lax.top_k(masked_routing, K)
  topk_idx = jax.lax.stop_gradient(topk_idx)

  causal_diag = jnp.tril(jnp.ones((tt, tt), dtype=bool))
  ranks = jnp.arange(K)

  def head_fn(q_h, k_h, v_h, topk_h):
    def block_fn(block_idx):
      q_i = q_h[block_idx]
      k_diag = k_h[block_idx][None, :, :]
      v_diag = v_h[block_idx][None, :, :]
      k_hist = k_h[topk_h[block_idx]]
      v_hist = v_h[topk_h[block_idx]]
      k_sub = jnp.concatenate([k_diag, k_hist], axis=0).reshape(tk_sub, D)
      v_sub = jnp.concatenate([v_diag, v_hist], axis=0).reshape(tk_sub, D)

      valid_hist = jnp.broadcast_to(
          jnp.repeat(ranks < block_idx, tt)[None, :],
          (tt, K * tt),
      )
      mask_sub = jnp.concatenate([causal_diag, valid_hist], axis=1)

      logits = (
          jnp.matmul(q_i, k_sub.T, precision=jax.lax.Precision.HIGHEST) * scale
      )
      logits = jnp.where(mask_sub, logits, -1e9)
      weights = jax.nn.softmax(logits, axis=-1)
      return jnp.matmul(weights, v_sub, precision=jax.lax.Precision.HIGHEST)

    return jax.vmap(block_fn)(jnp.arange(nt)).reshape(T, D)

  out = jax.vmap(head_fn)(q_blocks, k_blocks, v_blocks, topk_idx)
  return out.astype(orig_dtype)


def forward(q, k, v):
  """MoBA forward pass: block-routed causal attention.

  Each query block `i` attends to block `i` under a causal mask and, in full,
  to the `min(TOPK, nt)` earlier blocks with the largest routing score
  `mean(q_block_i) . mean(k_block_j) * scale`; block `i` has only `i` earlier
  blocks, so fewer are used for `i < TOPK`. The softmax runs over the union.
  The selected block indices are constants for differentiation. All maths is
  done in float32.

  Args:
    q: Queries, `[B, H, T, D]`. `T` must be a multiple of `CHUNK_SIZE`.
    k: Keys, `[B, H, T, D]`.
    v: Values, `[B, H, T, D]`.

  Returns:
    The output, `[B, H, T, D]`, in the dtype of `q`. Logits are scaled by
    `D ** -0.5`.
  """
  _, _, T, D = q.shape
  if T % CHUNK_SIZE != 0:
    raise ValueError(
        f"Sequence length ({T}) must be divisible by chunk_size ({CHUNK_SIZE})."
    )
  scale = float(D**-0.5)

  def batch_fn(q_b, k_b, v_b):
    return _single_seq_moba(q_b, k_b, v_b, scale, TOPK, CHUNK_SIZE)

  return jax.vmap(batch_fn)(q, k, v)


# ==============================================================================
# Pallas TPU forward kernel
# (inlined from tokamax/_src/ops/experimental/moba/pallas_mosaic_tpu_kernel.py,
#  Copyright 2026 Google LLC, Apache-2.0)
# ==============================================================================

_CHUNK_SIZE = 256
_TOPK = 4


def _compute_block_selection(
    q: jax.Array,
    k: jax.Array,
    scale: float,
    topk: int,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  """Hardware-aligned Option 1 Query-Block routing."""
  B, H, T, D = q.shape
  tt = chunk_size
  nt = T // tt
  K = min(topk, nt)

  q_blocks = q.reshape(B, H, nt, tt, D)
  k_blocks = k.reshape(B, H, nt, tt, D)

  q_mean = jnp.mean(q_blocks, axis=3)
  k_mean = jnp.mean(k_blocks, axis=3)

  routing_scores = (
      jnp.einsum(
          "bhqd,bhnd->bhqn",
          q_mean,
          k_mean,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )

  q_idx = jnp.arange(nt)[:, None]
  k_idx = jnp.arange(nt)[None, :]
  historical_mask = k_idx < q_idx

  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )

  _, topk_idx = jax.lax.top_k(masked_routing, K)
  return jax.lax.stop_gradient(topk_idx).astype(jnp.int32)


def _flash_moba_kernel(
    # Inputs.
    topk_smem_ref,  # [B, H, nt, K] int32   SMEM scalar prefetch
    q_vmem_ref,     # [1, 1, 1, tt, D] dtype  one block in VMEM
    k_vmem_ref,     # [1, 1, 1, tt, D] dtype  one block in VMEM
    v_vmem_ref,     # [1, 1, 1, tt, D] dtype  one block in VMEM
    # Outputs.
    o_vmem_ref,     # [1, 1, 1, tt, D] dtype  one block in VMEM
    # Scratch.
    m_scratch_ref,  # [tt, D] float32       running row-wise max in VMEM
    l_scratch_ref,  # [tt, D] float32       running softmax denominator in VMEM
    o_scratch_ref,  # [tt, D] float32       running output accumulator in VMEM
    *,
    scale: float,
) -> None:
  t_i = pl.program_id(2)
  s_i = pl.program_id(3)
  _, _, _, tt, D = q_vmem_ref.shape
  _, _, _, K = topk_smem_ref.shape

  @pl.when(s_i == 0)
  def _init_diag():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
    qk = jnp.where(col_idx <= row_idx, qk, -1e9)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.exp(qk - m_curr)
    l_curr = jnp.sum(p, axis=-1, keepdims=True)
    o_curr = jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    # Broadcast row-wise scalars across D to satisfy TPU VMEM 2D shape rules.
    m_scratch_ref[...] = jnp.broadcast_to(m_curr, (tt, D))
    l_scratch_ref[...] = jnp.broadcast_to(l_curr, (tt, D))
    o_scratch_ref[...] = o_curr

  @pl.when((s_i > 0) & ((s_i - 1) < t_i))
  def _acc_hist():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[...]

    m_curr = jnp.broadcast_to(jnp.max(qk, axis=-1, keepdims=True), (tt, D))
    m_new = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(qk - m_new[:, :1])
    l_curr = jnp.broadcast_to(jnp.sum(p, axis=-1, keepdims=True), (tt, D))
    l_new = alpha * l_prev + l_curr
    o_new = alpha * o_prev + jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = m_new
    l_scratch_ref[...] = l_new
    o_scratch_ref[...] = o_new

  @pl.when(s_i == K)
  def _finalize():
    o_final = o_scratch_ref[...] / l_scratch_ref[...]
    o_vmem_ref[0, 0, 0] = o_final.astype(o_vmem_ref.dtype)


def _flash_moba_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    topk_idx: jax.Array,
    scale: float,
    chunk_size: int = _CHUNK_SIZE,
    interpret: bool | None = None,
) -> jax.Array:
  """Executes MoBA attention via Pallas TPU kernel with SMEM block routing."""
  B, H, T, D = q.shape
  tt = chunk_size
  nt = T // tt
  _, _, _, K = topk_idx.shape

  q_blocks = q.reshape(B, H, nt, tt, D)
  k_blocks = k.reshape(B, H, nt, tt, D)
  v_blocks = v.reshape(B, H, nt, tt, D)
  topk_idx = topk_idx.astype(jnp.int32)

  def _q_index_map(b_i, h_i, t_i, s_i, topk_ref):
    del s_i, topk_ref
    return (b_i, h_i, t_i, 0, 0)

  def _kv_index_map(b_i, h_i, t_i, s_i, topk_ref):
    routed_idx = topk_ref[b_i, h_i, t_i, jnp.maximum(s_i - 1, 0)]
    kv_block_idx = jnp.where(s_i == 0, t_i, routed_idx)
    return (b_i, h_i, kv_block_idx, 0, 0)

  q_spec = pl.BlockSpec((1, 1, 1, tt, D), _q_index_map)
  kv_spec = pl.BlockSpec((1, 1, 1, tt, D), _kv_index_map)
  scratch_shape = pltpu.VMEM((tt, D), jnp.float32)

  if interpret is None:
    interpret = jax.default_backend() != "tpu"
  out_blocks = pl.pallas_call(
      functools.partial(_flash_moba_kernel, scale=scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(B, H, nt, K + 1),
          in_specs=[q_spec, kv_spec, kv_spec],
          out_specs=q_spec,
          scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct((B, H, nt, tt, D), dtype=q.dtype),
      interpret=interpret,
  )(topk_idx, q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(B, H, T, D)


def moba_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    topk: int = _TOPK,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, None]:
  """Executes the Mixture of Block Attention forward pass."""
  if scale is None:
    _, _, _, D = q.shape
    scale = D**-0.5

  topk_idx = _compute_block_selection(q, k, scale, topk, chunk_size)
  out = _flash_moba_pallas(
      q, k, v, topk_idx, scale, chunk_size, interpret=interpret
  )
  return out, None


def forward_pallas(q, k, v):
  """MoBA forward pass on the Pallas TPU kernel.

  Takes the same arguments as `forward`. Returns `(out, None)`: the Tokamax
  forward saves no residuals. Off TPU the kernel runs in Pallas interpret mode.
  """
  out, _ = moba_pallas_fwd(q, k, v, topk=TOPK, chunk_size=CHUNK_SIZE)
  return out, None


# ==============================================================================
# Benchmark harness
# ==============================================================================

CONFIGS = {
    # tokamax arg_specs: training_b1_t8192_h16_d128_bf16.
    'moba_b1_t8192': {
        'name': 'moba_b1_t8192_bwd_pallas_fwd',
        'model': 'MoBA',
        'operator': 'moba_backward',
        'batch': 1,
        'num_heads': 16,
        'seq_len': 8192,
        'head_dim': 128,
    },
    # Tiny shapes for CPU smoke tests: 8 blocks, so the last blocks choose 4
    # of up to 7 earlier blocks and the first blocks have fewer than 4.
    'smoke': {
        'name': 'moba_smoke_bwd_pallas_fwd',
        'model': 'MoBA',
        'operator': 'moba_backward',
        'batch': 2,
        'num_heads': 2,
        'seq_len': 2048,
        'head_dim': 16,
    },
}
CONFIG = CONFIGS['moba_b1_t8192']


def _resolve_config(config):
  if config is None:
    return CONFIG
  return CONFIGS[config] if isinstance(config, str) else config


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns `(q, k, v, dout)`; `dout` is the output cotangent.

  Keys and queries carry block structure so that the top-k routing is well
  separated: with unstructured random inputs the routing scores are near ties
  and the selected blocks depend on rounding. `v` and `dout` are standard
  normal.
  """
  cfg = _resolve_config(config)
  B, H = cfg['batch'], cfg['num_heads']
  T, D = cfg['seq_len'], cfg['head_dim']
  nt = T // CHUNK_SIZE
  if nt > D:
    raise ValueError(
        f'create_inputs needs seq_len // {CHUNK_SIZE} <= head_dim, got {nt} > '
        f'{D}.'
    )
  shape = (B, H, T, D)
  keys = jax.random.split(jax.random.key(42), 5)
  # Key block j: standard-normal noise plus an offset along coordinate j.
  k_dirs = jnp.eye(nt, D)  # [nt, D]
  k = jax.random.normal(keys[1], shape)
  k = k + (D / 2) ** 0.5 * jnp.repeat(k_dirs, CHUNK_SIZE, axis=0)
  # Query block i: noise plus equal offsets along the coordinates of block i
  # and of up to TOPK earlier blocks, drawn at random per batch and head. Those
  # earlier blocks get a routing score near `D**-0.5 * (D/2)**0.5 * (D/10)**0.5
  # = 0.22 * D**0.5`, all others a score near 0. The offset along block i
  # gives the block itself the same logit shift as the routed blocks, so
  # attention is not pulled towards either.
  u = jax.random.uniform(keys[4], (B, H, nt, nt))
  earlier = jnp.arange(nt)[None, :] < jnp.arange(nt)[:, None]  # [nt, nt]
  order = jnp.argsort(jnp.where(earlier, -u, 1.0), axis=-1)
  rank = jnp.argsort(order, axis=-1)
  # [B, H, nt, nt]
  targets = (earlier & (rank < TOPK)) | jnp.eye(nt, dtype=bool)
  q_dirs = jnp.pad(
      targets.astype(jnp.float32), ((0, 0), (0, 0), (0, 0), (0, D - nt))
  )  # [B, H, nt, D]
  q = jax.random.normal(keys[0], shape)
  q = q + (D / 10) ** 0.5 * jnp.repeat(q_dirs, CHUNK_SIZE, axis=2)
  v = jax.random.normal(keys[2], shape)
  dout = jax.random.normal(keys[3], shape)
  # Sample in float32 and cast, so every dtype sees the same values.
  return tuple(x.astype(dtype) for x in (q, k, v, dout))


# `workload` takes the `num_primals` arguments of `forward` followed by the
# output cotangent, and returns one gradient per entry of `diff_argnums`.
VJP_SPEC = {'num_primals': 3, 'diff_argnums': (0, 1, 2)}


def workload(q, k, v, dout):
  """Returns `(dq, dk, dv)`, the VJP of `forward` with `dout`."""
  _, vjp_fn = jax.vjp(forward, q, k, v)
  return vjp_fn(dout)


def get_flops(config=None):
  """Estimated FLOPs of `workload` (forward + backward).

  Per query block `forward` does two matmuls: the `tt x D` queries against
  `(K + 1) * tt` keys (`2 * tt * (K + 1) * tt * D`) and the attention weights
  against the same number of values (the same count), so `4 * (K + 1) * tt *
  tt * D` per block and `4 * (K + 1) * tt * B * H * T * D` in total. The
  routing matmul is `nt x nt x D` per head and is ignored. The VJP runs the
  forward once and a backward that costs about twice the forward, so 3x in
  total.
  """
  cfg = _resolve_config(config)
  B, H = cfg['batch'], cfg['num_heads']
  T, D = cfg['seq_len'], cfg['head_dim']
  tt = CHUNK_SIZE
  K = min(TOPK, T // tt)
  return int(3 * 4 * (K + 1) * tt * B * H * T * D)


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
