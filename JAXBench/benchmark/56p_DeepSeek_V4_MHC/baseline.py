"""DeepSeek V4 Multi-Head Concatenation (MHC) — Pallas TPU Kernel.

Self-contained implementation.
"""

# ==============================================================================
# Inlined mhc/utils.py
# ==============================================================================
from collections.abc import Callable
import json
import sys
import types
from typing import overload

import jax
import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16

# Explicit scoped-VMEM budget, repo convention (mla/v1, deepseek_v4 and
# fused_moe use the same constant). Never rely on the backend default —
# it varies by Mosaic version.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def round_up(x: int, multiple: int) -> int:
  return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
  """Grid token block, and the token count padded to a multiple of it.

  ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
  ``tb`` tokens. When supplied, the block halves until it fits within
  ``vmem_limit_bytes`` — degrading to a smaller block instead of a
  compile-time VMEM OOM — with ``SUBLANE`` as the floor.

  Returns (token_block, padded_tokens).
  """
  tb = min(token_block_size, round_up(num_tokens, SUBLANE))
  if vmem_need is not None:
    while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
      tb //= 2
  return tb, round_up(num_tokens, tb)


@overload
def pad_to(padded_tokens: int, a1: jax.Array, /) -> tuple[jax.Array]:
  ...


@overload
def pad_to(
    padded_tokens: int, a1: jax.Array, a2: jax.Array, /
) -> tuple[jax.Array, jax.Array]:
  ...


@overload
def pad_to(
    padded_tokens: int, a1: jax.Array, a2: jax.Array, a3: jax.Array, /
) -> tuple[jax.Array, jax.Array, jax.Array]:
  ...


@overload
def pad_to(
    padded_tokens: int,
    a1: jax.Array,
    a2: jax.Array,
    a3: jax.Array,
    a4: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  ...


@overload
def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  ...


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  """Zero-pad the leading (token) axis of each array to ``padded_tokens``."""
  pad = padded_tokens - arrays[0].shape[0]
  if pad == 0:
    return arrays
  return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


@overload
def trim_to(num_tokens: int, a1: jax.Array, /) -> tuple[jax.Array]:
  ...


@overload
def trim_to(
    num_tokens: int, a1: jax.Array, a2: jax.Array, /
) -> tuple[jax.Array, jax.Array]:
  ...


@overload
def trim_to(
    num_tokens: int, a1: jax.Array, a2: jax.Array, a3: jax.Array, /
) -> tuple[jax.Array, jax.Array, jax.Array]:
  ...


@overload
def trim_to(
    num_tokens: int,
    a1: jax.Array,
    a2: jax.Array,
    a3: jax.Array,
    a4: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  ...


@overload
def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  ...


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  """Inverse of ``pad_to``: drop the rows the padding added."""
  if arrays[0].shape[0] == num_tokens:
    return arrays
  return tuple(a[:num_tokens] for a in arrays)


def split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

  Computed in XLA outside the kernel; the chunks are what stays
  VMEM-resident. Chunks MUST be built with reduce_precision, not dtype
  round-trips: XLA's excess-precision simplification folds
  f32->bf16->f32 into the identity, which silently zeroes the mid/lo
  chunks.
  """
  fn_hi = jax.lax.reduce_precision(fn, 8, 7)
  rem = fn - fn_hi
  fn_mid = jax.lax.reduce_precision(rem, 8, 7)
  fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
  return (
      fn_hi.astype(jnp.bfloat16),
      fn_mid.astype(jnp.bfloat16),
      fn_lo.astype(jnp.bfloat16),
  )


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

  Shared by ``mhc_pre``, the Pallas ``pre_kernel`` and the fused seam
  op. Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
  """
  num_tokens = mixes.shape[0]

  mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

  pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
  pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

  post_logits = (
      mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
      + hc_base[hc_mult : 2 * hc_mult]
  )
  post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

  comb_logits = mixes[:, 2 * hc_mult :].reshape(
      num_tokens, hc_mult, hc_mult
  ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
  comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
  comb_mix = comb_mix / (
      jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
  )
  for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
    )
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
    )
  return pre_mix, post_mix, comb_mix


_utils_ns = types.SimpleNamespace(
    SUBLANE=SUBLANE,
    DEFAULT_VMEM_LIMIT_BYTES=DEFAULT_VMEM_LIMIT_BYTES,
    round_up=round_up,
    select_token_block=select_token_block,
    pad_to=pad_to,
    trim_to=trim_to,
    split_fn3=split_fn3,
    mhc_pre_gates=mhc_pre_gates,
)
utils = _utils_ns

# ==============================================================================
# Inlined mhc/fused_post_pre_kernel.py
# ==============================================================================
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined utils
# inlined DEFAULT_VMEM_LIMIT_BYTES


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    mixes_ref,
    sqrsum_ref,
    layer_ref,
    *,
    hc_mult,
    hidden_size,
    gemm_precision,
    rms_eps,
    hc_pre_eps,
):
  """One token block: recombine, store once, 3-pass GEMM, collapse."""
  post = post_ref[...]  # (tb, hc_mult) f32
  comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

  # Keep the streams (and x) in bf16 and upcast per use: transient f32
  # values instead of 4 held f32 copies halves the resident working set,
  # leaving VMEM headroom for the pipeline to prefetch the next block.
  # bf16 -> f32 casts are exact, so results are unchanged.
  x_bf = x_ref[...]  # (tb, hidden) bf16
  old_bf = [
      res_ref[:, i * hidden_size : (i + 1) * hidden_size]
      for i in range(hc_mult)
  ]

  streams_bf = []
  for j in range(hc_mult):
    acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
    for i in range(hc_mult):
      k = i * hc_mult + j
      acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
    # Round to bf16 BEFORE the GEMM: the unfused path's pre reads the
    # bf16 residual that post stored, and parity requires matching it.
    # The rounding is also what makes the 3-pass trick exact: the
    # streams' own bf16 split chunks 2 and 3 are identically zero.
    new_bf = acc.astype(jnp.bfloat16)
    newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
    streams_bf.append(new_bf)

  # ONE full-K dot per block, not one lane-sliced dot per stream: the
  # 4-dot decomposition starved the MXU (measured 3.6x slower than XLA's
  # equivalent GEMM). HIGHEST runs as the explicit 3-pass bf16
  # decomposition against the resident fn chunks (see module docstring);
  # "default" is the hi-chunk pass alone.
  g_bf = jnp.concatenate(streams_bf, axis=1)  # (tb, hc_mult * hidden_size)
  dn = (((1,), (1,)), ((), ()))
  acc = jax.lax.dot_general(
      g_bf, fn_hi_ref[...], dn, preferred_element_type=jnp.float32
  )
  if gemm_precision == 'highest':
    acc = acc + jax.lax.dot_general(
        g_bf, fn_mid_ref[...], dn, preferred_element_type=jnp.float32
    )
    acc = acc + jax.lax.dot_general(
        g_bf, fn_lo_ref[...], dn, preferred_element_type=jnp.float32
    )
  mixes_ref[...] = acc
  gf = g_bf.astype(jnp.float32)  # f32 for the squared sum + collapse
  sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)
  sqrsum_ref[...] = sqr

  # In-block collapse (mirrors pre_kernel._mixes_collapse_kernel): the
  # collapse's weights are ``pre_mix`` — the sigmoid read gates, which
  # need no Sinkhorn and are per-token, so they are computable entirely
  # in-block. Collapsing against the bf16-ROUNDED streams (gf, their
  # exact f32 upcast) matches the unfused semantics, where pre reads
  # the bf16 residual post stored — never the f32 accumulators. Under
  # gemm_precision="default" pre_mix comes from the hi-only mixes: the
  # documented lower-precision mode.
  # Same op order as reference.mhc_pre_gates for bit-level agreement.
  m, h = hc_mult, hidden_size
  scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
  pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
  lay = pre_mix[:, 0:1] * gf[:, :h]
  for i in range(1, m):
    lay = lay + pre_mix[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
  layer_ref[...] = lay.astype(jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=(
        'rms_eps',
        'hc_pre_eps',
        'token_block_size',
        'gemm_precision',
        'vmem_limit_bytes',
    ),
)
def fused_post_pre_mixes(
    x2d: jax.Array,
    res2d: jax.Array,
    post2d: jax.Array,
    comb2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 32,
    gemm_precision: str = 'highest',
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """The Pallas seam kernel: post recombine fused with pre's mix GEMM

  and in-block collapse.

  Args:
      x2d: (num_tokens, hidden_size), bfloat16 — sublayer output.
      res2d: (num_tokens, hc_mult * hidden_size), bfloat16 — old streams.
      post2d: (num_tokens, hc_mult), float32.
      comb2d: (num_tokens, hc_mult * hc_mult), float32, row-major (i, j).
      fn: (hc_mult3, hc_mult * hidden_size), float32 — next sublayer's gate
        projection.
      hc_scale: (3,), float32 — pre/post/comb logit scales (the kernel uses only
        the pre entry; the rest feed the XLA gates).
      hc_base: (hc_mult3,), float32 — logit biases, pre entries first.
      rms_eps: static; RMS-norm epsilon for the in-block pre gates.
      hc_pre_eps: static; additive floor on the sigmoid pre gates.
      token_block_size: tokens per grid step. Default 32 keeps the worst-case
        VMEM footprint (f32 old streams + bf16 blocks + the three resident bf16
        fn chunks) near 12 MB at DeepSeek-V4 shapes; 64 also fits.
      interpret: run in Pallas interpret mode (any backend; for tests).
      gemm_precision: "highest" (3-pass exact decomposition, the default) or
        "default" (hi-chunk pass only; pre_mix and the collapse then also see
        the hi-only mixes).

  Returns:
      new_res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
      mixes: (num_tokens, hc_mult3), float32 — raw, unscaled, for the
          XLA gates.
      sqrsum: (num_tokens, 1), float32.
      layer2d: (num_tokens, hidden_size), bfloat16.
  """
  assert x2d.dtype == jnp.bfloat16, x2d.dtype
  assert res2d.dtype == jnp.bfloat16, res2d.dtype
  assert post2d.dtype == jnp.float32, post2d.dtype
  assert comb2d.dtype == jnp.float32, comb2d.dtype
  assert fn.dtype == jnp.float32, fn.dtype

  num_tokens, hidden_size = x2d.shape
  hc_hidden = res2d.shape[1]
  hc_mult = hc_hidden // hidden_size
  hc_mult3 = fn.shape[0]
  assert res2d.shape == (num_tokens, hc_mult * hidden_size)
  assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, res2d.shape)
  assert gemm_precision in ('highest', 'default'), gemm_precision

  # This kernel keeps a fixed block: its default (32) already fits the
  # budget at DeepSeek-V4 shapes, so there is no vmem_need to shrink by.
  tb, padded_tokens = utils.select_token_block(num_tokens, token_block_size)
  x2d, res2d, post2d, comb2d = utils.pad_to(
      padded_tokens, x2d, res2d, post2d, comb2d
  )

  fn_hi, fn_mid, fn_lo = utils.split_fn3(fn)
  sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
  hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

  resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
  new_res, mixes, sqrsum, layer2d = pl.pallas_call(
      functools.partial(
          _fused_kernel,
          hc_mult=hc_mult,
          hidden_size=hidden_size,
          gemm_precision=gemm_precision,
          rms_eps=rms_eps,
          hc_pre_eps=hc_pre_eps,
      ),
      grid=(padded_tokens // tb,),
      in_specs=[
          pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
          resident,
          resident,
          resident,
          pl.BlockSpec((1, 3), lambda i: (0, 0)),
          pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult3), lambda i: (i, 0)),
          pl.BlockSpec((tb, 1), lambda i: (i, 0)),
          pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((padded_tokens, hc_hidden), jnp.bfloat16),
          jax.ShapeDtypeStruct((padded_tokens, hc_mult3), jnp.float32),
          jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
          jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel',),
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
  )(x2d, res2d, post2d, comb2d, fn_hi, fn_mid, fn_lo, sc2d, hb2d)

  return utils.trim_to(num_tokens, new_res, mixes, sqrsum, layer2d)


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 32,
    gemm_precision: str = 'highest',
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fused seam op: Pallas post+GEMM+collapse kernel, XLA gates epilogue.

  Same contract as ``reference.mhc_fused_post_pre`` (and vLLM's
  ``MHCFusedPostPreOp``): returns
  (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
  The gates recompute ``pre_mix`` on the raw mixes; its cost is a few
  tiny VPU ops and keeping ``mhc_pre_gates`` shared and
  unsliced is worth more than removing them.
  """
  assert residual.dtype == jnp.bfloat16

  outer_shape = residual.shape[:-2]
  hc_mult, hidden_size = residual.shape[-2:]

  x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
  res2d = residual.reshape(-1, hc_mult * hidden_size)
  post2d = post_layer_mix.reshape(-1, hc_mult)
  comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)

  new_res2d, mixes, sqrsum, layer_input_cur = fused_post_pre_mixes(
      x2d,
      res2d,
      post2d,
      comb2d,
      fn,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      token_block_size=token_block_size,
      gemm_precision=gemm_precision,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  _, post_mix_cur, comb_mix_cur = utils.mhc_pre_gates(
      mixes,
      sqrsum,
      hc_mult,
      hidden_size,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )
  return (
      new_res2d.reshape(*outer_shape, hc_mult, hidden_size),
      post_mix_cur.reshape(*outer_shape, hc_mult, 1),
      comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
      layer_input_cur.reshape(*outer_shape, hidden_size),
  )


# ==============================================================================
# Benchmark Harness
# ==============================================================================
CONFIGS = {
    'dsv4_pro_decode': {
        'name': 'dsv4_pro_decode',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'multi_head_concatenation',
        'num_tokens': 256,
        'hidden_size': 7168,
        'hc_mult': 4,
        'token_block_size': 32,
    },
    'dsv4_pro_prefill': {
        'name': 'dsv4_pro_prefill',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'multi_head_concatenation',
        'num_tokens': 2048,
        'hidden_size': 7168,
        'hc_mult': 4,
        'token_block_size': 32,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'multi_head_concatenation',
        'num_tokens': 256,
        'hidden_size': 4096,
        'hc_mult': 4,
        'token_block_size': 64,
    },
    'dsv4_flash_prefill': {
        'name': 'dsv4_flash_prefill',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'multi_head_concatenation',
        'num_tokens': 2048,
        'hidden_size': 4096,
        'hc_mult': 4,
        'token_block_size': 64,
    },
    'mhc_small': {
        'name': 'mhc_small',
        'model': 'DeepSeek-V4',
        'operator': 'multi_head_concatenation',
        'num_tokens': 128,
        'hidden_size': 2048,
        'hc_mult': 4,
        'token_block_size': 32,
    },
}

CONFIG = CONFIGS['dsv4_pro_decode']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base)."""
  cfg = config or CONFIG
  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

  num_tokens = cfg['num_tokens']
  hidden_size = cfg['hidden_size']
  hc_mult = cfg['hc_mult']
  hc_mult3 = hc_mult * (hc_mult + 2)

  x = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
  residual = jax.random.normal(
      k2, (num_tokens, hc_mult, hidden_size), dtype=dtype
  )
  post_layer_mix = jax.random.uniform(
      k3, (num_tokens, hc_mult, 1), dtype=jnp.float32
  )
  comb_res_mix = jax.random.uniform(
      k4, (num_tokens, hc_mult, hc_mult), dtype=jnp.float32
  )
  fn = (
      jax.random.normal(
          k5, (hc_mult3, hc_mult * hidden_size), dtype=jnp.float32
      )
      * 0.02
  )
  hc_scale = jax.random.uniform(k6, (3,), dtype=jnp.float32)
  hc_base = jax.random.uniform(k7, (hc_mult3,), dtype=jnp.float32)

  return x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of (dynamic_args, static_args) for all configs."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def workload(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base):
  """Pallas fused MHC post/pre kernel."""
  hidden_size = x.shape[-1]
  token_block_size = 64 if hidden_size <= 4096 else 32
  return mhc_fused_post_pre(
      x,
      residual,
      post_layer_mix,
      comb_res_mix,
      fn,
      hc_scale,
      hc_base,
      rms_eps=1e-6,
      hc_pre_eps=1e-6,
      hc_sinkhorn_eps=1e-6,
      hc_post_mult_value=2.0,
      sinkhorn_repeat=20,
      token_block_size=token_block_size,
  )


def get_flops(config=None):
  """Total FLOPs for the MHC operations."""
  cfg = config or CONFIG
  num_tokens = cfg['num_tokens']
  hidden_size = cfg['hidden_size']
  hc_mult = cfg['hc_mult']
  hc_mult3 = hc_mult * (hc_mult + 2)
  post_flops = num_tokens * (
      hc_mult * hidden_size + 2 * hc_mult * hc_mult * hidden_size
  )
  pre_flops = 2 * num_tokens * (hc_mult * hidden_size) * hc_mult3
  return int(post_flops + pre_flops)


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = config or CONFIG
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = fn(*inputs)
    jax.tree.map(
        lambda x: x.block_until_ready()
        if hasattr(x, 'block_until_ready')
        else None,
        out,
    )
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = fn(*inputs)
    jax.tree.map(
        lambda x: x.block_until_ready()
        if hasattr(x, 'block_until_ready')
        else None,
        out,
    )
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  flops = get_flops(cfg)
  avg = float(np.mean(times))
  return {
      'name': cfg['name'],
      'model': cfg['model'],
      'operator': cfg['operator'],
      'config': {
          k: v for k, v in cfg.items() if k not in ('name', 'model', 'operator')
      },
      'time_ms': round(avg, 4),
      'std_ms': round(float(np.std(times)), 4),
      'tflops': round(flops / (avg / 1000) / 1e12, 2) if avg > 0 else 0.0,
      'output_shape': [list(x.shape) for x in out],
      'status': 'success',
  }


if __name__ == '__main__':
  print(json.dumps(benchmark()))
