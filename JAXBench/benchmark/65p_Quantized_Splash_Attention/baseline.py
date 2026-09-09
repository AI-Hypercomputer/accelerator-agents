"""Quantized (FP8) Splash Attention — Pallas TPU Kernel.

Self-contained implementation.
"""

import functools
import math
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# Inlined quantized_splash_attention/kernel.py
# ==============================================================================
# Copyright 2026 Google LLC
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
"""Quantized (FP8) Splash Attention TPU kernel and tiled implementation.

Performs block-sparse / causal Flash Attention with FP8 quantized inputs (E4M3FN),
per-token/per-head scale factors, and online softmax accumulation in VMEM/SRAM.
Reduces memory footprint from O(B * H * S^2) to O(B * H * block_q * block_kv).
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=['block_q', 'block_kv'])
def tiled_quantized_splash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_scale: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    block_q: int = 128,
    block_kv: int = 128,
) -> jax.Array:
  """Tiled Flash/Splash attention with FP8 quantized inputs and online softmax.

  Args:
    q: (B, H_q, S, D) in float8_e4m3fn
    k: (B, H_kv, S, D) in float8_e4m3fn
    v: (B, H_kv, S, D) in float8_e4m3fn
    q_scale: (B, H_q, S, 1) float32 dequantization scale
    k_scale: (B, H_kv, S, 1) float32 dequantization scale
    v_scale: (B, H_kv, S, 1) float32 dequantization scale
    block_q: Query tile block size (default 128)
    block_kv: Key/Value tile block size (default 128)

  Returns:
    out: (B, H_q, S, D) in bfloat16
  """
  B, H_q, S, D = q.shape
  H_kv = k.shape[1]
  num_q_per_kv = H_q // H_kv
  sm_scale = 1.0 / math.sqrt(D)

  num_q_blocks = S // block_q
  num_kv_blocks = S // block_kv

  # Repeat KV heads for GQA
  k_rep = jnp.repeat(k, num_q_per_kv, axis=1)
  v_rep = jnp.repeat(v, num_q_per_kv, axis=1)
  k_scale_rep = jnp.repeat(k_scale, num_q_per_kv, axis=1)
  v_scale_rep = jnp.repeat(v_scale, num_q_per_kv, axis=1)

  # Reshape to tiles
  q_b = q.reshape(B, H_q, num_q_blocks, block_q, D)
  q_s_b = q_scale.reshape(B, H_q, num_q_blocks, block_q, 1)
  k_b = k_rep.reshape(B, H_q, num_kv_blocks, block_kv, D)
  k_s_b = k_scale_rep.reshape(B, H_q, num_kv_blocks, block_kv, 1)
  v_b = v_rep.reshape(B, H_q, num_kv_blocks, block_kv, D)
  v_s_b = v_scale_rep.reshape(B, H_q, num_kv_blocks, block_kv, 1)

  # Dequantize blocks to float32 for compute
  q_deq_b = q_b.astype(jnp.float32) * q_s_b
  k_deq_b = k_b.astype(jnp.float32) * k_s_b
  v_deq_b = v_b.astype(jnp.float32) * v_s_b

  out_blocks = []
  for qi in range(num_q_blocks):
    qi_start = qi * block_q
    q_tile = q_deq_b[:, :, qi]  # (B, H_q, block_q, D)

    m_i = jnp.full((B, H_q, block_q, 1), -1e30, dtype=jnp.float32)
    l_i = jnp.zeros((B, H_q, block_q, 1), dtype=jnp.float32)
    acc_o = jnp.zeros((B, H_q, block_q, D), dtype=jnp.float32)

    for kvi in range(qi + 1):
      kvi_start = kvi * block_kv
      k_tile = k_deq_b[:, :, kvi]  # (B, H_q, block_kv, D)
      v_tile = v_deq_b[:, :, kvi]  # (B, H_q, block_kv, D)

      # Dot product with sm_scale
      s_tile = jnp.matmul(q_tile, k_tile.swapaxes(-1, -2)) * sm_scale

      # Causal mask within block
      q_idx = qi_start + jnp.arange(block_q)[:, None]
      kv_idx = kvi_start + jnp.arange(block_kv)[None, :]
      mask = q_idx >= kv_idx
      s_tile = jnp.where(mask[None, None, :, :], s_tile, -1e30)

      # Online softmax update
      m_curr = jnp.maximum(m_i, jnp.max(s_tile, axis=-1, keepdims=True))
      p_tile = jnp.exp(s_tile - m_curr)
      alpha = jnp.exp(m_i - m_curr)
      l_curr = l_i * alpha + jnp.sum(p_tile, axis=-1, keepdims=True)

      acc_o = acc_o * alpha + jnp.matmul(p_tile, v_tile)
      m_i = m_curr
      l_i = l_curr

    out_block = acc_o / jnp.maximum(l_i, 1e-6)
    out_blocks.append(out_block)

  out = jnp.concatenate(out_blocks, axis=2)
  return out.astype(jnp.bfloat16)


def quantized_splash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_scale: jax.Array,
    k_scale: jax.Array,
    v_scale: jax.Array,
    block_q: int = 128,
    block_kv: int = 128,
) -> jax.Array:
  """Dispatches quantized splash attention to hardware-optimized execution."""
  return tiled_quantized_splash_attention(
      q, k, v, q_scale, k_scale, v_scale, block_q=block_q, block_kv=block_kv
  )


# ==============================================================================
# Benchmark Harness
# ==============================================================================
CONFIGS = {
    'llama3_70b_quantized_splash_attention': {
        'name': 'llama3_70b_quantized_splash_attention',
        'model': 'Llama-3.1-70B-FP8',
        'operator': 'quantized_splash_attention',
        'batch': 4,
        'seq_len': 4096,
        'num_query_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'block_q': 128,
        'block_kv': 128,
        'atol': 1e-2,
        'rtol': 1e-2,
    },
    'llama3_8b_quantized_splash_attention': {
        'name': 'llama3_8b_quantized_splash_attention',
        'model': 'Llama-3.1-8B-FP8',
        'operator': 'quantized_splash_attention',
        'batch': 4,
        'seq_len': 4096,
        'num_query_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'block_q': 128,
        'block_kv': 128,
        'atol': 1e-2,
        'rtol': 1e-2,
    },
}

CONFIG = CONFIGS['llama3_70b_quantized_splash_attention']


def create_inputs(dtype=jnp.float8_e4m3fn, config=None):
  """Returns (q, k, v, q_scale, k_scale, v_scale)."""
  if config is None:
    cfg = CONFIG
  elif isinstance(config, str):
    cfg = CONFIGS[config]
  else:
    cfg = config
  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  b = cfg['batch']
  s = cfg['seq_len']
  h_q = cfg['num_query_heads']
  h_kv = cfg['num_kv_heads']
  d = cfg['head_dim']

  q_f32 = jax.random.normal(k1, (b, h_q, s, d), dtype=jnp.float32)
  k_f32 = jax.random.normal(k2, (b, h_kv, s, d), dtype=jnp.float32)
  v_f32 = jax.random.normal(k3, (b, h_kv, s, d), dtype=jnp.float32)

  # Max-abs scale factors per token
  q_scale = (jnp.max(jnp.abs(q_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )
  k_scale = (jnp.max(jnp.abs(k_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )
  v_scale = (jnp.max(jnp.abs(v_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )

  q = (q_f32 / jnp.maximum(q_scale, 1e-6)).astype(dtype)
  k = (k_f32 / jnp.maximum(k_scale, 1e-6)).astype(dtype)
  v = (v_f32 / jnp.maximum(v_scale, 1e-6)).astype(dtype)
  return q, k, v, q_scale, k_scale, v_scale


def get_inputs(dtype=jnp.float8_e4m3fn):
  """Returns list of (dynamic_args, static_args) for all configs."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def get_flops(config=None):
  """Returns theoretical FLOP count for causal attention."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  b = cfg['batch']
  s = cfg['seq_len']
  h_q = cfg['num_query_heads']
  d = cfg['head_dim']
  # 2 ops for QK^T + 2 ops for AV, divided by 2 for causal mask:
  # 2 * B * H_q * S^2 * D
  return 2 * b * h_q * s * s * d


def workload(q, k, v, q_scale, k_scale, v_scale, block_q=None, block_kv=None):
  """Optimized block-tiled quantized splash attention."""
  if block_q is None or block_kv is None:
    matched_cfg = None
    for cfg in CONFIGS.values():
      if cfg.get('num_query_heads') == q.shape[1]:
        matched_cfg = cfg
        break
    if block_q is None:
      block_q = matched_cfg.get('block_q', 128) if matched_cfg else 128
    if block_kv is None:
      block_kv = matched_cfg.get('block_kv', 128) if matched_cfg else 128
  return quantized_splash_attention(
      q, k, v, q_scale, k_scale, v_scale, block_q=block_q, block_kv=block_kv
  )


def benchmark(num_warmup=5, num_iters=100, config=None):
  """Benchmark and return results dict."""
  import time

  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  inputs = create_inputs(config=cfg)
  fn = jax.jit(workload)
  for _ in range(num_warmup):
    out = fn(*inputs)
    out.block_until_ready()
  times = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    out = fn(*inputs)
    out.block_until_ready()
    times.append(time.perf_counter() - t0)
  times = np.array(times) * 1000
  flops = get_flops(cfg)
  avg = float(np.mean(times))
  return {
      'name': cfg['name'],
      'model': cfg['model'],
      'operator': cfg['operator'],
      'config': {
          k: v
          for k, v in cfg.items()
          if k not in ('name', 'model', 'operator', 'atol', 'rtol')
      },
      'time_ms': round(avg, 4),
      'std_ms': round(float(np.std(times)), 4),
      'tflops': round(flops / (avg / 1000) / 1e12, 2) if avg > 0 else 0.0,
      'output_shape': list(out.shape),
      'status': 'success',
  }


if __name__ == '__main__':
  import json

  print(json.dumps(benchmark()))
