# pylint: skip-file
"""Kernel-only splash attention benchmark — jax-experimental vs tokamax.

Times splash attention fwd+bwd at the Llama 3 8B JAX trainer's per-chip
shape (B, Hq, L, hd). Two kernel families are swept:

1. `jax.experimental.pallas.ops.tpu.splash_attention` — upstream JAX splash.
   Knobs: BlockSizes (block_q, block_kv, block_q_dkv, block_kv_dkv,
   use_fused_bwd_kernel).
2. `tokamax._src.ops.experimental.tpu.splash_attention` — tokamax-shipped
   splash. Same block knobs **plus** `use_base2_exp`, `fuse_reciprocal`,
   `max_logit_const`, `dq_reduction_steps`, `use_experimental_scheduler`.

Per-shard cost is identical to the in-trainer cost when shard_map wraps
this kernel with q_seq_shards=1, so single-device timing is faithful.

Output: per-config CSV-formatted lines on stdout (prefixed `[CSV]`) and a
final ranked table. Intended to be run once on a GKE pod and harvested
from `kubectl logs`.

Default sweep:
  jax-experimental:   bq × bkv ∈ {1024, 2048} × {1024, 2048, 4096} = 6 configs.
  tokamax:            same 6 block configs × {base2 ∈ {0,1}} × {fuse_recip
                      ∈ {0,1}} × {mlc ∈ {None, 30}} = 48 configs.

  Total ~54 configs × ~3s/config (~3 min wall clock).

Llama 3 8B per-chip shape (post FSDP fsdp=8, bs=5 globally=40):
  B=5, Hq=32, Hkv=8, L=8192, hd=128
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from typing import Any

import jax
# Upstream jax-experimental splash
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as jax_splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as jax_mask,
)
import jax.numpy as jnp
import numpy as np

# Tokamax-shipped splash
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as tk_splash,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as tk_mask,
)


# Llama 3 8B per-chip GQA shape.
HQ = 32
HKV = 8
HD = 128


def make_qkv(batch: int, seq_len: int, dtype: jnp.dtype, device):
  """(B, Hq, L, hd) for Q; (B, Hkv, L, hd) for K, V."""
  rng = np.random.default_rng(0)
  q = (rng.standard_normal((batch, HQ, seq_len, HD)) * 0.1).astype(np.float32)
  k = (rng.standard_normal((batch, HKV, seq_len, HD)) * 0.1).astype(np.float32)
  v = (rng.standard_normal((batch, HKV, seq_len, HD)) * 0.1).astype(np.float32)
  return tuple(
      jax.device_put(jnp.asarray(a, dtype=dtype), device) for a in (q, k, v)
  )


def time_fn(fn, args, n_warmup: int, n_iters: int) -> float:
  """Returns mean ms/iter."""
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  start = time.perf_counter()
  for _ in range(n_iters):
    out = fn(*args)
  jax.block_until_ready(out)
  end = time.perf_counter()
  return (end - start) / n_iters * 1e3


def attn_flops(batch: int, seq_len: int, num_kv_groups: int) -> float:
  """Causal attention FLOPs for one fwd+bwd pass (×3) at this shape.

  fwd: 4*B*Hq*L^2*hd / 2 = 2*B*Hq*L^2*hd (causal mask)
  bwd: ~2× fwd → 4*B*Hq*L^2*hd
  total fwd+bwd: ~6*B*Hq*L^2*hd
  """
  return 6.0 * batch * HQ * seq_len * seq_len * HD


# -------------------- jax-experimental splash --------------------


@dataclasses.dataclass(frozen=True)
class JaxConfig:
  block_q: int
  block_kv: int
  block_q_dkv: int
  block_kv_dkv: int


def build_jax_kernel(seq_len: int, cfg: JaxConfig):
  bs = jax_splash.BlockSizes(
      block_q=cfg.block_q,
      block_kv=cfg.block_kv,
      block_kv_compute=cfg.block_kv,
      block_q_dkv=cfg.block_q_dkv,
      block_kv_dkv=cfg.block_kv_dkv,
      block_kv_dkv_compute=cfg.block_kv_dkv,
      block_q_dq=None,
      block_kv_dq=None,
      use_fused_bwd_kernel=True,
      q_layout=jax_splash.QKVLayout.HEAD_DIM_MINOR,
      k_layout=jax_splash.QKVLayout.HEAD_DIM_MINOR,
      v_layout=jax_splash.QKVLayout.HEAD_DIM_MINOR,
  )
  base = jax_mask.CausalMask(shape=(seq_len, seq_len))
  mask = jax_mask.MultiHeadMask(masks=(base,) * HQ)
  return jax_splash.make_splash_mha_single_device(mask, block_sizes=bs)


# -------------------- tokamax splash --------------------


@dataclasses.dataclass(frozen=True)
class TokamaxConfig:
  block_q: int
  block_kv: int
  block_q_dkv: int
  block_kv_dkv: int
  use_base2_exp: bool
  fuse_reciprocal: bool
  max_logit_const: float | None


def build_tokamax_kernel(seq_len: int, cfg: TokamaxConfig):
  sa = tk_splash.SplashConfig(
      block_q=cfg.block_q,
      block_kv=cfg.block_kv,
      block_kv_compute=cfg.block_kv,
      block_q_dkv=cfg.block_q_dkv,
      block_kv_dkv=cfg.block_kv_dkv,
      block_kv_dkv_compute=cfg.block_kv_dkv,
      use_fused_bwd_kernel=True,
      q_layout=tk_splash.QKVLayout.HEAD_DIM_MINOR,
      k_layout=tk_splash.QKVLayout.HEAD_DIM_MINOR,
      v_layout=tk_splash.QKVLayout.HEAD_DIM_MINOR,
      use_base2_exp=cfg.use_base2_exp,
      fuse_reciprocal=cfg.fuse_reciprocal,
      max_logit_const=cfg.max_logit_const,
  )
  single_head_mask = tk_mask.CausalMask(shape=(seq_len, seq_len))
  return tk_splash.make_splash_mha(
      mask=single_head_mask, q_seq_shards=1, config=sa
  )


# -------------------- timing harness --------------------


def time_fwd_bwd_jax(kernel, qkv, n_warmup, n_iters):
  """jax-experimental splash expects per-head MHA layout (Hq=Hkv after repeat).

  We `_repeat_kv` here for an apples-to-apples vs tokamax (which natively
  handles GQA broadcast).
  """
  q, k, v = qkv
  # broadcast K/V to MHA so jax-experimental sees same shape as tokamax internal
  k_rep = jnp.repeat(k, HQ // HKV, axis=1)
  v_rep = jnp.repeat(v, HQ // HKV, axis=1)
  qkv_mha = (q, k_rep, v_rep)

  vfwd = jax.vmap(kernel)

  def loss_fn(q_, k_, v_):
    return vfwd(q_, k_, v_).astype(jnp.float32).sum()

  grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
  return time_fn(grad_fn, qkv_mha, n_warmup, n_iters)


def time_fwd_bwd_tokamax(kernel, qkv, n_warmup, n_iters):
  """Tokamax splash takes (q, k, v) with q.shape[1]=Hq, k/v.shape[1]=Hkv;

  broadcasts internally.
  """
  vfwd = jax.vmap(kernel)

  def loss_fn(q_, k_, v_):
    return vfwd(q_, k_, v_).astype(jnp.float32).sum()

  grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
  return time_fn(grad_fn, qkv, n_warmup, n_iters)


def csv_row(*cells: Any) -> str:
  return "[CSV]," + ",".join("" if c is None else str(c) for c in cells)


# -------------------- main --------------------


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
      "--batch_size",
      type=int,
      default=5,
      help="Per-chip batch dim for the kernel call",
  )
  p.add_argument("--seq_len", type=int, default=8192)
  p.add_argument("--n_warmup", type=int, default=2)
  p.add_argument("--n_iters", type=int, default=10)
  p.add_argument("--dtype", default="bfloat16")
  args = p.parse_args()

  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
  # Single-host process — pick a local device. On multi-host JAX, devices()[0]
  # may belong to another process and `device_put` rejects the copy.
  device = jax.local_devices()[0]
  qkv = make_qkv(args.batch_size, args.seq_len, dtype, device)

  print(
      f"Llama 3 8B kernel benchmark — B={args.batch_size} Hq={HQ} Hkv={HKV} "
      f"L={args.seq_len} hd={HD} dtype={dtype}"
  )

  # CSV header
  print(
      csv_row(
          "kernel",
          "block_q",
          "block_kv",
          "block_q_dkv",
          "block_kv_dkv",
          "use_base2_exp",
          "fuse_reciprocal",
          "max_logit_const",
          "ms",
          "TFLOPs",
      )
  )

  # FLOPs estimate for fwd+bwd (3× fwd) at this shape per call.
  flops = attn_flops(args.batch_size, args.seq_len, HQ // HKV)

  results: list[tuple[str, dict, float]] = []

  # ============== Sweep 1: jax-experimental splash ==============
  block_pairs = [
      (1024, 1024),
      (1024, 2048),
      (2048, 1024),
      (2048, 2048),
      (2048, 4096),
      (4096, 2048),
  ]
  print(f"\n=== jax-experimental splash sweep ({len(block_pairs)} configs) ===")
  for bq, bkv in block_pairs:
    cfg = JaxConfig(bq, bkv, bq, bkv)
    try:
      kernel = build_jax_kernel(args.seq_len, cfg)
      ms = time_fwd_bwd_jax(kernel, qkv, args.n_warmup, args.n_iters)
      tflops = flops / (ms * 1e-3) / 1e12
      print(
          csv_row(
              "jax-experimental",
              bq,
              bkv,
              bq,
              bkv,
              "",
              "",
              "",
              f"{ms:.3f}",
              f"{tflops:.2f}",
          )
      )
      results.append((
          "jax-experimental",
          dict(block_q=bq, block_kv=bkv, block_q_dkv=bq, block_kv_dkv=bkv),
          ms,
      ))
    except Exception as e:
      print(
          f"[skip jax-experimental bq={bq} bkv={bkv}] {type(e).__name__}:"
          f" {str(e)[:120]}"
      )

  # ============== Sweep 2: tokamax-splash ==============
  # Smaller block-size grid + the 3 perf knobs.
  knobs = list(itertools.product([False, True], [False, True], [None, 30.0]))
  tk_block_pairs = [(1024, 2048), (2048, 1024), (2048, 2048), (2048, 4096)]
  n_tokamax = len(tk_block_pairs) * len(knobs)
  print(f"\n=== tokamax-splash sweep ({n_tokamax} configs) ===")
  for bq, bkv in tk_block_pairs:
    for base2, fuse_recip, mlc in knobs:
      cfg = TokamaxConfig(
          bq,
          bkv,
          bq,
          bkv,
          use_base2_exp=base2,
          fuse_reciprocal=fuse_recip,
          max_logit_const=mlc,
      )
      try:
        kernel = build_tokamax_kernel(args.seq_len, cfg)
        ms = time_fwd_bwd_tokamax(kernel, qkv, args.n_warmup, args.n_iters)
        tflops = flops / (ms * 1e-3) / 1e12
        print(
            csv_row(
                "tokamax",
                bq,
                bkv,
                bq,
                bkv,
                int(base2),
                int(fuse_recip),
                mlc,
                f"{ms:.3f}",
                f"{tflops:.2f}",
            )
        )
        results.append((
            "tokamax",
            dict(
                block_q=bq,
                block_kv=bkv,
                block_q_dkv=bq,
                block_kv_dkv=bkv,
                use_base2_exp=base2,
                fuse_reciprocal=fuse_recip,
                max_logit_const=mlc,
            ),
            ms,
        ))
      except Exception as e:
        print(
            f"[skip tokamax bq={bq} bkv={bkv} base2={base2} "
            f"fuse_recip={fuse_recip} mlc={mlc}] "
            f"{type(e).__name__}: {str(e)[:100]}"
        )

  # ============== Final ranked table ==============
  results.sort(key=lambda r: r[2])
  print("\n========================== TOP 15 ==========================")
  print(f"{'rank':>4}  {'kernel':<18}  {'ms':>8}  TFLOPs  config")
  for i, (kernel, cfg, ms) in enumerate(results[:15]):
    tflops = flops / (ms * 1e-3) / 1e12
    cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"{i+1:>4}  {kernel:<18}  {ms:>8.3f}  {tflops:>5.2f}  {cfg_str}")
  print("============================================================")


if __name__ == "__main__":
  main()
