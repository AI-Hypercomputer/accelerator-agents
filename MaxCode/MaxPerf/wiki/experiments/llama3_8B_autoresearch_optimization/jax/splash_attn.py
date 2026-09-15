# pylint: skip-file
# Copyright 2025 Google LLC
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

import functools

# Optional tokamax-shipped splash impl. Tokamax adds knobs not in upstream
# JAX's splash: `use_base2_exp`, `fuse_reciprocal`, `use_experimental_scheduler`,
# `dq_reduction_steps`. Toggle with env var USE_TOKAMAX_SPLASH=1.
import os as _os_top

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

_USE_TOKAMAX_SPLASH = _os_top.environ.get("USE_TOKAMAX_SPLASH", "0") == "1"
if _USE_TOKAMAX_SPLASH:
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_kernel as tokamax_splash_kernel,
  )
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_mask as tokamax_splash_mask,
  )


def tpu_splash_attention(
    mesh,
    q_sharding,
    # Input should be of shape (batch, length, heads, kv_dim)
    apply_shard_map,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    decoder_segment_ids: jax.Array | None,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  """TPU Flash Attention."""
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(
        decoder_segment_ids, decoder_segment_ids
    )

  # Autotuned config from exp 8 (171-config sweep across 3 shapes).
  # Universal winner: symmetric block_q == block_kv == 1024,
  # `dkv` blocks scale with seq_len (capped via the BlockSizes `min(...)`
  # clamps below), and the fused bwd kernel wins by 7-15 % on every shape.
  # See wiki/.../torchax/experiments/2026-04-25-exp8-splash-kernel-autotune-potential.md
  # exp 36: at seq=8192, exp 10's autotune found bq=2048 bkv=1024 is +0.16%
  # faster than symmetric 1024,1024. The asymmetric layout may also use less
  # workspace per call. min(global, seq_len) clamps to seq for shorter seqs.
  # Allow env-var override per block size for sweeping without rebuilds.
  import os as _os

  global_block_q = int(_os.environ.get("SPLASH_BQ", "2048"))
  global_block_kv = int(_os.environ.get("SPLASH_BKV", "1024"))
  global_block_kv_compute = int(_os.environ.get("SPLASH_BKV_COMPUTE", "1024"))
  global_block_q_dkv = int(_os.environ.get("SPLASH_BQ_DKV", "2048"))
  global_block_kv_dkv = int(_os.environ.get("SPLASH_BKV_DKV", "2048"))
  global_block_kv_dkv_compute = int(
      _os.environ.get("SPLASH_BKV_DKV_COMPUTE", "2048")
  )
  global_use_fused_bwd_kernel = _os.environ.get("SPLASH_FUSED_BWD", "1") == "1"
  # Unfused-bwd needs explicit dQ block sizes (default to bq/bkv).
  global_block_q_dq = (
      None
      if global_use_fused_bwd_kernel
      else int(_os.environ.get("SPLASH_BQ_DQ", str(global_block_q)))
  )
  global_block_kv_dq = (
      None
      if global_use_fused_bwd_kernel
      else int(_os.environ.get("SPLASH_BKV_DQ", str(global_block_kv)))
  )
  global_q_layout = "HEAD_DIM_MINOR"
  global_k_layout = "HEAD_DIM_MINOR"
  global_v_layout = "HEAD_DIM_MINOR"

  # Log effective config so we can read it back in xpk logs.
  print(
      f"[splash] bq={global_block_q} bkv={global_block_kv} "
      f"bq_dkv={global_block_q_dkv} bkv_dkv={global_block_kv_dkv} "
      f"fused_bwd={global_use_fused_bwd_kernel} "
      f"bq_dq={global_block_q_dq} bkv_dq={global_block_kv_dq}",
      flush=True,
  )

  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    if decoder_segment_ids is not None:
      assert (
          query.shape[2] == decoder_segment_ids.q.shape[1]
      ), "Sharding along sequence dimension not allowed in tpu kernel attention"
    if _USE_TOKAMAX_SPLASH:
      use_base2_exp = _os.environ.get("TOKAMAX_USE_BASE2_EXP", "1") == "1"
      fuse_reciprocal = _os.environ.get("TOKAMAX_FUSE_RECIPROCAL", "1") == "1"
      use_experimental_scheduler = (
          _os.environ.get("TOKAMAX_USE_EXP_SCHED", "0") == "1"
      )
      dq_reduction_steps = (
          int(_os.environ["TOKAMAX_DQ_REDUCTION_STEPS"])
          if _os.environ.get("TOKAMAX_DQ_REDUCTION_STEPS", "")
          else None
      )
      max_logit_const = (
          float(_os.environ["TOKAMAX_MAX_LOGIT_CONST"])
          if _os.environ.get("TOKAMAX_MAX_LOGIT_CONST", "")
          else None
      )
      cost_estimate_flops_fwd = (
          float(_os.environ["TOKAMAX_COST_FWD"])
          if _os.environ.get("TOKAMAX_COST_FWD", "")
          else None
      )
      cost_estimate_flops_bwd = (
          float(_os.environ["TOKAMAX_COST_BWD"])
          if _os.environ.get("TOKAMAX_COST_BWD", "")
          else None
      )
      from jax.experimental import pallas as _pl

      fwd_cost = (
          _pl.CostEstimate(
              flops=int(cost_estimate_flops_fwd),
              transcendentals=0,
              bytes_accessed=0,
          )
          if cost_estimate_flops_fwd is not None
          else None
      )
      bwd_cost = (
          _pl.CostEstimate(
              flops=int(cost_estimate_flops_bwd),
              transcendentals=0,
              bytes_accessed=0,
          )
          if cost_estimate_flops_bwd is not None
          else None
      )
      sa_config = tokamax_splash_kernel.SplashConfig(
          block_q=min(global_block_q, query.shape[2]),
          block_kv=min(global_block_kv, key.shape[2]),
          block_kv_compute=min(global_block_kv_compute, key.shape[2]),
          block_q_dkv=min(global_block_q_dkv, query.shape[2]),
          block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
          block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
          use_fused_bwd_kernel=True,  # tokamax only supports fused bwd
          q_layout=tokamax_splash_kernel.QKVLayout[global_q_layout],
          k_layout=tokamax_splash_kernel.QKVLayout[global_k_layout],
          v_layout=tokamax_splash_kernel.QKVLayout[global_v_layout],
          attn_logits_soft_cap=attn_logits_soft_cap,
          use_base2_exp=use_base2_exp,
          fuse_reciprocal=fuse_reciprocal,
          use_experimental_scheduler=use_experimental_scheduler,
          dq_reduction_steps=dq_reduction_steps,
          max_logit_const=max_logit_const,
          fwd_cost_estimate=fwd_cost,
          bwd_cost_estimate=bwd_cost,
      )
      print(
          f"[tokamax-splash] config: bq={sa_config.block_q}"
          f" bkv={sa_config.block_kv} bq_dkv={sa_config.block_q_dkv}"
          f" bkv_dkv={sa_config.block_kv_dkv} base2={use_base2_exp}"
          f" fuse_recip={fuse_reciprocal}"
          f" exp_sched={use_experimental_scheduler}",
          flush=True,
      )
      # tokamax splash takes a single mask and broadcasts to all heads internally.
      single_head_mask = tokamax_splash_mask.CausalMask(
          shape=(query.shape[2], query.shape[2])
      )
      _q_seq_shards = int(_os.environ.get("TOKAMAX_Q_SEQ_SHARDS", "1"))
      splash_kernel = tokamax_splash_kernel.make_splash_mha(
          mask=single_head_mask, q_seq_shards=_q_seq_shards, config=sa_config
      )
    else:
      block_sizes = splash_attention_kernel.BlockSizes(
          block_q=min(global_block_q, query.shape[2]),
          block_kv=min(global_block_kv, key.shape[2]),
          block_kv_compute=min(global_block_kv_compute, key.shape[2]),
          block_q_dkv=min(global_block_q_dkv, query.shape[2]),
          block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
          block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
          block_q_dq=None
          if global_use_fused_bwd_kernel
          else min(global_block_q_dq, query.shape[2]),
          block_kv_dq=None
          if global_use_fused_bwd_kernel
          else min(global_block_kv_dq, query.shape[2]),
          use_fused_bwd_kernel=global_use_fused_bwd_kernel,
          q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
          k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
          v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
      )
      mask = splash_attention_mask.CausalMask(
          shape=(query.shape[2], query.shape[2])
      )
      multi_head_mask = splash_attention_mask.MultiHeadMask(
          masks=(mask,) * query.shape[1]
      )
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=block_sizes,
          attn_logits_soft_cap=attn_logits_soft_cap,
      )

    return jax.vmap(splash_kernel)(
        query, key, value, segment_ids=decoder_segment_ids
    )

  if apply_shard_map:
    wrap_flash_attention = shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(
            q_sharding,
            q_sharding,
            q_sharding,
            None,
        ),
        out_specs=q_sharding,
        check_rep=False,
    )

  x = wrap_flash_attention(query, key, value, decoder_segment_ids)
  return x


if __name__ == "__main__":
  main()
