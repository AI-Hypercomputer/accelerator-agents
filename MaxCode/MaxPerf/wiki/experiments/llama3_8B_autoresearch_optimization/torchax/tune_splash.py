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
"""Kernel-only autotune for splash attention on Llama 3 8B shapes.

Sweeps BlockSizes x layouts x use_fused_bwd_kernel for the exact (B, H, L, hd)
the Llama 3 8B torchax trainer feeds into splash, on a single device. The
multi-host FSDP wrapping (shard_map with head_shards=1, q_seq_shards=1) does
not change what the kernel itself sees per-shard, so single-device timing is
faithful to the per-chip cost in production.

Sweep design:
  Phase 1 - forward only. Grid: block_q x block_kv x (q_layout in
            {HEAD_DIM_MINOR, SEQ_MINOR}). k/v_layout pinned to HEAD_DIM_MINOR
            (gemma4 exp24 found q_layout dominates).
  Phase 2 - full fwd+bwd, gradient via jax.grad. Take top-3 fwd configs from
            phase 1, sweep block_q_dkv x block_kv_dkv x block_q_dq x
            block_kv_dq x use_fused_bwd_kernel. Fused path zeros the dq blocks.

Output: per-config CSV-formatted lines on stdout (prefixed with [CSV]) plus a
final ranked table. kubectl logs is the source of truth; copy-paste the [CSV]
lines into a spreadsheet for analysis.

Usage (XPK, both hosts run the same command, both print, results are identical
within noise so just read rank 0's logs):
  python tune_splash.py --seq_len 1024 --batch_size 4 --mode fwd_bwd
"""

import argparse
import dataclasses
import itertools
import math
import time
import traceback
from typing import Any

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
import jax.numpy as jnp
import numpy as np

# Llama 3 8B post-`repeat_kv` MHA shape. The torchax trainer's HF SDPA path
# expands GQA to full-MHA before reaching `_custom_attention`, so the kernel
# always sees Hq == Hkv == 32.
HQ = 32
HD = 128


@dataclasses.dataclass(frozen=True)
class FwdConfig:
  block_q: int
  block_kv: int
  q_layout: int  # QKVLayout enum value
  # k/v layouts pinned to HEAD_DIM_MINOR by default; q_layout dominates.

  def block_sizes(self) -> splash.BlockSizes:
    return splash.BlockSizes(
        block_q=self.block_q,
        block_kv=self.block_kv,
        block_kv_compute=self.block_kv,
        q_layout=splash.QKVLayout(self.q_layout),
        k_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        v_layout=splash.QKVLayout.HEAD_DIM_MINOR,
    )


@dataclasses.dataclass(frozen=True)
class FullConfig:
  fwd: FwdConfig
  block_q_dkv: int
  block_kv_dkv: int
  block_q_dq: int | None  # None when use_fused_bwd_kernel
  block_kv_dq: int | None
  use_fused_bwd_kernel: bool

  def block_sizes(self) -> splash.BlockSizes:
    return splash.BlockSizes(
        block_q=self.fwd.block_q,
        block_kv=self.fwd.block_kv,
        block_kv_compute=self.fwd.block_kv,
        block_q_dkv=self.block_q_dkv,
        block_kv_dkv=self.block_kv_dkv,
        block_kv_dkv_compute=self.block_kv_dkv,
        block_q_dq=None if self.use_fused_bwd_kernel else self.block_q_dq,
        block_kv_dq=None if self.use_fused_bwd_kernel else self.block_kv_dq,
        use_fused_bwd_kernel=self.use_fused_bwd_kernel,
        q_layout=splash.QKVLayout(self.fwd.q_layout),
        k_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        v_layout=splash.QKVLayout.HEAD_DIM_MINOR,
    )


def make_qkv(batch: int, seq_len: int, dtype: jnp.dtype, device) -> tuple:
  rng = np.random.default_rng(0)
  shape = (batch, HQ, seq_len, HD)
  arrs = [
      jnp.asarray(
          (rng.standard_normal(shape) * 0.1).astype(np.float32), dtype=dtype
      )
      for _ in range(3)
  ]
  return tuple(jax.device_put(a, device) for a in arrs)


def build_kernel(seq_len: int, block_sizes: splash.BlockSizes):
  base = mask_lib.CausalMask(shape=(seq_len, seq_len))
  mask = mask_lib.MultiHeadMask(masks=(base,) * HQ)
  return splash.make_splash_mha_single_device(mask, block_sizes=block_sizes)


def time_fn(fn, args, n_warmup: int, n_iters: int) -> float:
  for _ in range(n_warmup):
    out = fn(*args)
  jax.block_until_ready(out)
  start = time.perf_counter()
  for _ in range(n_iters):
    out = fn(*args)
  jax.block_until_ready(out)
  end = time.perf_counter()
  return (end - start) / n_iters * 1e3  # ms


def time_fwd(kernel, qkv, n_warmup: int, n_iters: int) -> float:
  fwd = jax.jit(jax.vmap(kernel))
  return time_fn(fwd, qkv, n_warmup, n_iters)


def time_fwd_bwd(kernel, qkv, n_warmup: int, n_iters: int) -> float:
  vfwd = jax.vmap(kernel)

  def loss_fn(q_, k_, v_):
    out = vfwd(q_, k_, v_)
    # cast to fp32 for a stable scalar; matches training-loss reduction
    return out.astype(jnp.float32).sum()

  grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
  return time_fn(grad_fn, qkv, n_warmup, n_iters)


def fmt_layout(qly: int) -> str:
  return splash.QKVLayout(qly).name


def csv_row(*cells: Any) -> str:
  return "[CSV]," + ",".join(str(c) for c in cells)


def time_one(
    *,
    kernel,
    qkv,
    mode: str,
    n_warmup: int,
    n_iters: int,
) -> float | None:
  try:
    if mode == "fwd":
      return time_fwd(kernel, qkv, n_warmup, n_iters)
    elif mode == "fwd_bwd":
      return time_fwd_bwd(kernel, qkv, n_warmup, n_iters)
    else:
      raise ValueError(mode)
  except Exception as e:
    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
    print(f"  [skip] {msg[:160]}")
    return None


def phase1_fwd(
    seq_len: int,
    batch: int,
    qkv,
    n_warmup: int,
    n_iters: int,
) -> list[tuple[FwdConfig, float]]:
  tiles = [b for b in [256, 512, 1024, 2048] if b <= seq_len]
  layouts = [splash.QKVLayout.HEAD_DIM_MINOR, splash.QKVLayout.SEQ_MINOR]
  print(
      csv_row(
          "phase",
          "config",
          "block_q",
          "block_kv",
          "q_layout",
          "k_layout",
          "v_layout",
          "fused_bwd",
          "block_q_dkv",
          "block_kv_dkv",
          "block_q_dq",
          "block_kv_dq",
          "ms",
          "TFLOP/s",
      )
  )
  configs = [
      FwdConfig(bq, bkv, int(qly))
      for bq, bkv, qly in itertools.product(tiles, tiles, layouts)
  ]
  flops = batch * HQ * seq_len * seq_len * HD * 4  # qk + sv each 2*B*H*L*L*hd
  print(
      f"\n=== Phase 1 (fwd-only, {len(configs)} configs, "
      f"shape=({batch},{HQ},{seq_len},{HD})) ==="
  )
  results: list[tuple[FwdConfig, float]] = []
  for i, cfg in enumerate(configs):
    print(
        f"\n[fwd {i+1}/{len(configs)}] block_q={cfg.block_q} "
        f"block_kv={cfg.block_kv} q_layout={fmt_layout(cfg.q_layout)}"
    )
    t0 = time.perf_counter()
    kernel = build_kernel(seq_len, cfg.block_sizes())
    ms = time_one(
        kernel=kernel,
        qkv=qkv,
        mode="fwd",
        n_warmup=n_warmup,
        n_iters=n_iters,
    )
    t1 = time.perf_counter()
    if ms is None:
      tflops = float("nan")
    else:
      tflops = flops / (ms * 1e-3) / 1e12
      results.append((cfg, ms))
    print(
        f"  -> {ms if ms else 'skip':>8.3f} ms  "
        f"({tflops:5.1f} TFLOP/s; total {(t1 - t0):4.1f}s)"
        if ms
        else "  -> skip"
    )
    print(
        csv_row(
            "fwd",
            i,
            cfg.block_q,
            cfg.block_kv,
            fmt_layout(cfg.q_layout),
            "HEAD_DIM_MINOR",
            "HEAD_DIM_MINOR",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            f"{ms:.4f}" if ms else "skip",
            f"{tflops:.2f}" if ms else "skip",
        )
    )
  results.sort(key=lambda x: x[1])
  return results


def phase2_full(
    seq_len: int,
    batch: int,
    qkv,
    top_fwd: list[FwdConfig],
    n_warmup: int,
    n_iters: int,
) -> list[tuple[FullConfig, float]]:
  tiles_dkv = [b for b in [512, 1024, 2048] if b <= seq_len]
  tiles_dq = [b for b in [512, 1024, 2048] if b <= seq_len]
  configs: list[FullConfig] = []
  for fwd_cfg in top_fwd:
    # Fused-bwd: dq fields are None. Sweep dkv block grid.
    for bq_dkv, bkv_dkv in itertools.product(tiles_dkv, tiles_dkv):
      configs.append(
          FullConfig(
              fwd=fwd_cfg,
              block_q_dkv=bq_dkv,
              block_kv_dkv=bkv_dkv,
              block_q_dq=None,
              block_kv_dq=None,
              use_fused_bwd_kernel=True,
          )
      )
    # Non-fused-bwd: pin dq blocks to dkv blocks (separate dq grid is rarely
    # the win, and balloons the search). Phase-3 follow-up could decouple if
    # phase-2 selects a non-fused winner.
    for bq, bkv in itertools.product(tiles_dkv, tiles_dkv):
      configs.append(
          FullConfig(
              fwd=fwd_cfg,
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_q_dq=bq,
              block_kv_dq=bkv,
              use_fused_bwd_kernel=False,
          )
      )
  # FLOPs for fwd+bwd: ~3x fwd (one fwd + two bwd matmul-equivalents)
  flops_fwd = batch * HQ * seq_len * seq_len * HD * 4
  flops = flops_fwd * 3
  print(
      f"\n=== Phase 2 (fwd+bwd, {len(configs)} configs, "
      f"top-{len(top_fwd)} fwd seeds) ==="
  )
  results: list[tuple[FullConfig, float]] = []
  for i, cfg in enumerate(configs):
    fused = "T" if cfg.use_fused_bwd_kernel else "F"
    print(
        f"\n[full {i+1}/{len(configs)}] "
        f"bq={cfg.fwd.block_q} bkv={cfg.fwd.block_kv} "
        f"qly={fmt_layout(cfg.fwd.q_layout)} fused={fused} "
        f"dkv=({cfg.block_q_dkv},{cfg.block_kv_dkv}) "
        f"dq=({cfg.block_q_dq},{cfg.block_kv_dq})"
    )
    t0 = time.perf_counter()
    try:
      kernel = build_kernel(seq_len, cfg.block_sizes())
    except Exception as e:
      print(f"  [skip build] {str(e)[:160]}")
      ms = None
    else:
      ms = time_one(
          kernel=kernel,
          qkv=qkv,
          mode="fwd_bwd",
          n_warmup=n_warmup,
          n_iters=n_iters,
      )
    t1 = time.perf_counter()
    if ms is None:
      tflops = float("nan")
    else:
      tflops = flops / (ms * 1e-3) / 1e12
      results.append((cfg, ms))
    print(
        f"  -> {ms if ms else 'skip':>8.3f} ms  "
        f"({tflops:5.1f} TFLOP/s; total {(t1 - t0):4.1f}s)"
        if ms
        else "  -> skip"
    )
    print(
        csv_row(
            "full",
            i,
            cfg.fwd.block_q,
            cfg.fwd.block_kv,
            fmt_layout(cfg.fwd.q_layout),
            "HEAD_DIM_MINOR",
            "HEAD_DIM_MINOR",
            fused,
            cfg.block_q_dkv,
            cfg.block_kv_dkv,
            cfg.block_q_dq if cfg.block_q_dq is not None else "n/a",
            cfg.block_kv_dq if cfg.block_kv_dq is not None else "n/a",
            f"{ms:.4f}" if ms else "skip",
            f"{tflops:.2f}" if ms else "skip",
        )
    )
  results.sort(key=lambda x: x[1])
  return results


def production_default_full(seq_len: int) -> FullConfig:
  """The current production splash config from torchax/splash_attn.py.

  Production wraps each block size with `min(global_*, seq_len)` so that at
  shorter sequences it clamps without raising. We replicate that clamping here.
  """
  return FullConfig(
      fwd=FwdConfig(
          min(1024, seq_len),
          min(512, seq_len),
          int(splash.QKVLayout.HEAD_DIM_MINOR),
      ),
      block_q_dkv=min(2048, seq_len),
      block_kv_dkv=min(512, seq_len),
      block_q_dq=min(2048, seq_len),
      block_kv_dq=min(512, seq_len),
      use_fused_bwd_kernel=False,
  )


def maxtext_recipe_full(seq_len: int) -> FullConfig:
  """The MaxText Llama 3.1-8B recipe block config (for reference)."""
  return FullConfig(
      fwd=FwdConfig(
          min(1024, seq_len),
          min(1024, seq_len),
          int(splash.QKVLayout.HEAD_DIM_MINOR),
      ),
      block_q_dkv=min(1024, seq_len),
      block_kv_dkv=min(1024, seq_len),
      block_q_dq=min(1024, seq_len),
      block_kv_dq=min(1024, seq_len),
      use_fused_bwd_kernel=True,
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq_len", type=int, default=1024)
  ap.add_argument("--batch_size", type=int, default=4, help="per-shard batch")
  ap.add_argument("--dtype", type=str, default="bfloat16")
  ap.add_argument(
      "--mode",
      choices=["fwd", "fwd_bwd"],
      default="fwd_bwd",
      help="fwd: phase1 only; fwd_bwd: both phases",
  )
  ap.add_argument(
      "--top_k",
      type=int,
      default=3,
      help="top-k fwd configs to take into phase2",
  )
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--iters", type=int, default=30)
  args = ap.parse_args()

  print(f"\n=== splash kernel autotune ===")
  print(f"jax devices: {jax.devices()}")
  print(f"local devices: {jax.local_devices()}")
  print(
      f"process_index={jax.process_index()} /"
      f" process_count={jax.process_count()}"
  )
  device = jax.local_devices()[0]
  print(f"timing on: {device}")
  # Both ranks independently run the same sweep on their own local chip 0.
  # Each gets its own log stream — read rank-0's logs as the canonical result;
  # rank-1's serves as a cross-check (should agree within ~1%).
  print(
      f"shape (B, Hq, L, hd) = ({args.batch_size}, {HQ}, {args.seq_len}, {HD})"
  )
  print(f"dtype: {args.dtype}, warmup={args.warmup}, iters={args.iters}")

  dtype = jnp.dtype(args.dtype)
  qkv = make_qkv(args.batch_size, args.seq_len, dtype, device)

  # Anchor: production-default config baseline.
  print("\n=== anchor: production default ===")
  prod = production_default_full(args.seq_len)
  print(f"  blocks: {prod.block_sizes()}")
  kernel = build_kernel(args.seq_len, prod.block_sizes())
  prod_fwd_ms = time_one(
      kernel=kernel,
      qkv=qkv,
      mode="fwd",
      n_warmup=args.warmup,
      n_iters=args.iters,
  )
  prod_fb_ms = time_one(
      kernel=kernel,
      qkv=qkv,
      mode="fwd_bwd",
      n_warmup=args.warmup,
      n_iters=args.iters,
  )
  print(f"  production fwd     = {prod_fwd_ms:.3f} ms")
  print(f"  production fwd+bwd = {prod_fb_ms:.3f} ms")

  # MaxText recipe reference.
  mtx = maxtext_recipe_full(args.seq_len)
  print(f"\n=== reference: MaxText Llama 3.1-8B recipe ===")
  print(f"  blocks: {mtx.block_sizes()}")
  kernel = build_kernel(args.seq_len, mtx.block_sizes())
  mtx_fwd_ms = time_one(
      kernel=kernel,
      qkv=qkv,
      mode="fwd",
      n_warmup=args.warmup,
      n_iters=args.iters,
  )
  mtx_fb_ms = time_one(
      kernel=kernel,
      qkv=qkv,
      mode="fwd_bwd",
      n_warmup=args.warmup,
      n_iters=args.iters,
  )
  print(f"  maxtext fwd     = {mtx_fwd_ms:.3f} ms")
  print(f"  maxtext fwd+bwd = {mtx_fb_ms:.3f} ms")

  # Phase 1 — forward sweep.
  fwd_results = phase1_fwd(
      args.seq_len, args.batch_size, qkv, args.warmup, args.iters
  )
  print("\n--- Phase 1 ranked (best fwd) ---")
  for i, (cfg, ms) in enumerate(fwd_results[:10]):
    delta = (
        (prod_fwd_ms - ms) / prod_fwd_ms * 100 if prod_fwd_ms else float("nan")
    )
    print(
        f"  {i + 1:2d}. {ms:7.3f} ms  "
        f"bq={cfg.block_q:4d} bkv={cfg.block_kv:4d} "
        f"qly={fmt_layout(cfg.q_layout):14s} "
        f"({delta:+.2f}% vs prod fwd)"
    )

  if args.mode == "fwd":
    print("\n=== mode=fwd, stopping after phase 1 ===")
    return

  # Phase 2 — full fwd+bwd sweep on top-k fwd seeds.
  top_fwd = [c for c, _ in fwd_results[: args.top_k]]
  full_results = phase2_full(
      args.seq_len, args.batch_size, qkv, top_fwd, args.warmup, args.iters
  )
  print("\n--- Phase 2 ranked (best fwd+bwd) ---")
  for i, (cfg, ms) in enumerate(full_results[:15]):
    delta = (prod_fb_ms - ms) / prod_fb_ms * 100 if prod_fb_ms else float("nan")
    fused = "T" if cfg.use_fused_bwd_kernel else "F"
    print(
        f"  {i + 1:2d}. {ms:8.3f} ms  fwd(bq={cfg.fwd.block_q:4d} "
        f"bkv={cfg.fwd.block_kv:4d} qly={fmt_layout(cfg.fwd.q_layout):14s}) "
        f"fused={fused} "
        f"dkv=({cfg.block_q_dkv:4d},{cfg.block_kv_dkv:4d}) "
        f"dq=({cfg.block_q_dq if cfg.block_q_dq else 'None':>4},"
        f"{cfg.block_kv_dq if cfg.block_kv_dq else 'None':>4})"
        f"  ({delta:+.2f}% vs prod fwd+bwd)"
    )

  print(
      "\n=== summary ===\n"
      f"  production fwd     : {prod_fwd_ms:.3f} ms\n"
      f"  production fwd+bwd : {prod_fb_ms:.3f} ms\n"
      f"  maxtext   fwd      : {mtx_fwd_ms:.3f} ms ("
      f"{(prod_fwd_ms - mtx_fwd_ms) / prod_fwd_ms * 100:+.2f}% vs prod)\n"
      f"  maxtext   fwd+bwd  : {mtx_fb_ms:.3f} ms ("
      f"{(prod_fb_ms - mtx_fb_ms) / prod_fb_ms * 100:+.2f}% vs prod)"
  )
  if fwd_results:
    best_cfg, best_ms = fwd_results[0]
    print(
        f"  best fwd           : {best_ms:.3f} ms ("
        f"{(prod_fwd_ms - best_ms) / prod_fwd_ms * 100:+.2f}% vs prod) "
        f"bq={best_cfg.block_q} bkv={best_cfg.block_kv} "
        f"qly={fmt_layout(best_cfg.q_layout)}"
    )
  if full_results:
    best_cfg, best_ms = full_results[0]
    fused = "T" if best_cfg.use_fused_bwd_kernel else "F"
    print(
        f"  best fwd+bwd       : {best_ms:.3f} ms ("
        f"{(prod_fb_ms - best_ms) / prod_fb_ms * 100:+.2f}% vs prod) "
        f"bq={best_cfg.fwd.block_q} bkv={best_cfg.fwd.block_kv} "
        f"qly={fmt_layout(best_cfg.fwd.q_layout)} fused={fused} "
        f"dkv=({best_cfg.block_q_dkv},{best_cfg.block_kv_dkv}) "
        f"dq=({best_cfg.block_q_dq},{best_cfg.block_kv_dq})"
    )


if __name__ == "__main__":
  main()
