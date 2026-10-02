"""TPU equivalence test — HF PyTorch CPU reference vs sharded JAX TPU run.

This complements `test_equivalence.py`. The CPU test proves the Flax NNX model
math and HF weight loader match PyTorch in eager mode. This test exercises the
optimized execution path we care about on TPU: mesh placement, parameter
sharding, input sharding, and a jitted loss+gradient step.

Run on a TPU VM / XPK worker:

    python test_tpu_equivalence.py

The test intentionally refuses to run on CPU unless `--allow_cpu` is passed.
It uses a tiny Qwen3 config so the PyTorch reference can run on host CPU.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM as HFQwen3ForCausalLM

# The TPU training images often default JAX to TPU. For the explicit debug mode,
# select CPU before importing JAX so local smoke tests do not try TPU topology.
if "--allow_cpu" in sys.argv and "JAX_PLATFORMS" not in os.environ:
  os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

THIS_DIR = "/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_cx/wiki/experiments/qwen3_cx_autoresearch_optimization/jax"
sys.path.insert(0, THIS_DIR)

from model import Qwen3ForCausalLM as FlaxQwen3ForCausalLM
from model import get_param, load_hf_state_dict
from model.sharding import apply_sharding, build_plan, input_sharding


def tiny_config() -> Qwen3Config:
  return Qwen3Config(
      vocab_size=256,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      max_position_embeddings=64,
      rope_theta=1_000_000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
      attention_dropout=0.0,
      attn_implementation="eager",
  )


def dtype_from_name(name: str) -> jnp.dtype:
  if name == "bf16":
    return jnp.bfloat16
  if name == "fp32":
    return jnp.float32
  raise ValueError(f"unsupported dtype {name!r}; use bf16 or fp32")


def cross_entropy_jax(logits: jax.Array, labels: jax.Array) -> jax.Array:
  v = logits.shape[-1]
  logp = jax.nn.log_softmax(logits.reshape(-1, v).astype(jnp.float32), axis=-1)
  picked = jnp.take_along_axis(
      logp, labels.reshape(-1)[:, None], axis=-1
  ).squeeze(-1)
  return -picked.mean()


def make_mesh(tp_parallelism: int):
  n_global = jax.device_count()
  if n_global % tp_parallelism != 0:
    raise ValueError(
        f"device_count={n_global} is not divisible by tp={tp_parallelism}"
    )
  fsdp = n_global // tp_parallelism
  axis_type = jax.sharding.AxisType
  mesh = jax.make_mesh(
      (fsdp, tp_parallelism),
      ("fsdp", "tp"),
      axis_types=(axis_type.Auto, axis_type.Auto),
  )
  return mesh, fsdp


def summarize(name: str, actual, expected, atol: float, rtol: float) -> bool:
  actual = np.asarray(actual, dtype=np.float32)
  expected = np.asarray(expected, dtype=np.float32)
  diff = np.abs(actual - expected)
  max_abs = float(diff.max()) if diff.size else 0.0
  denom = float(np.abs(expected).max()) or 1.0
  ok = np.allclose(actual, expected, atol=atol, rtol=rtol)
  print(
      f"  {'OK ' if ok else 'XX '} {name:<48} "
      f"max|delta|={max_abs:.3e} max_rel={max_abs / denom:.3e}",
      flush=True,
  )
  return ok


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument(
      "--allow_cpu",
      action="store_true",
      help="allow running without a TPU; intended only for debugging",
  )
  p.add_argument(
      "--dtype",
      choices=("bf16", "fp32"),
      default="bf16",
      help="JAX TPU weight/compute dtype",
  )
  p.add_argument(
      "--batch_size",
      type=int,
      default=None,
      help="global batch; default is one example per global device",
  )
  p.add_argument("--seqlen", type=int, default=16)
  p.add_argument("--tp_parallelism", type=int, default=1)
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--fwd_atol", type=float, default=None)
  p.add_argument("--fwd_rtol", type=float, default=None)
  p.add_argument("--bwd_atol", type=float, default=None)
  p.add_argument("--bwd_rtol", type=float, default=None)
  return p.parse_args()


def main() -> int:
  args = parse_args()
  try:
    backend = jax.default_backend()
    devices = jax.devices()
  except Exception as exc:
    if not args.allow_cpu:
      print(
          "Unable to initialize a JAX backend for TPU equivalence. Run this"
          " test on a TPU runtime, or pass --allow_cpu only for local"
          " debugging.",
          flush=True,
      )
      print(f"Backend initialization error: {exc}", flush=True)
      return 2
    raise
  print(
      f"[jax] backend={backend} device_count={jax.device_count()}"
      f" devices={devices}",
      flush=True,
  )
  if backend == "cpu" and not args.allow_cpu:
    print(
        "Refusing to run TPU equivalence test on CPU. Use a TPU runtime or pass"
        " --allow_cpu.",
        flush=True,
    )
    return 2

  cfg = tiny_config()
  batch_size = args.batch_size or jax.device_count()
  if batch_size % jax.device_count() != 0:
    raise ValueError(
        f"global batch_size={batch_size} must be divisible by"
        f" device_count={jax.device_count()}"
    )

  torch.manual_seed(0)
  ref = HFQwen3ForCausalLM(cfg).to(torch.float32).eval()

  gen = torch.Generator().manual_seed(args.seed)
  input_ids = torch.randint(
      0, cfg.vocab_size, (batch_size, args.seqlen), generator=gen
  )
  labels = torch.randint(
      0, cfg.vocab_size, (batch_size, args.seqlen), generator=gen
  )

  ref.zero_grad(set_to_none=True)
  logits_ref = ref(input_ids).logits
  loss_ref = F.cross_entropy(
      logits_ref.reshape(-1, cfg.vocab_size), labels.reshape(-1)
  )
  loss_ref.backward()
  grads_ref = {
      name: param.grad.detach().float().cpu().numpy()
      for name, param in ref.named_parameters()
      if param.grad is not None
  }
  print(
      f"[torch-cpu] loss={loss_ref.item():.6f}"
      f" params_with_grad={len(grads_ref)}",
      flush=True,
  )

  jax_dtype = dtype_from_name(args.dtype)
  mesh, fsdp = make_mesh(args.tp_parallelism)
  print(f"[jax] mesh fsdp={fsdp} tp={args.tp_parallelism}: {mesh}", flush=True)

  model = FlaxQwen3ForCausalLM(
      cfg,
      weights_dtype=jax_dtype,
      compute_dtype=jax_dtype,
      rngs=nnx.Rngs(0),
  )
  n_loaded, _, unexpected = load_hf_state_dict(
      model, ref.state_dict(), dtype=jax_dtype
  )
  if unexpected:
    raise RuntimeError(
        f"unexpected HF keys while loading JAX model: {unexpected[:8]}"
    )
  print(
      f"[jax] loaded {n_loaded} HF tensors into Flax model as {args.dtype}",
      flush=True,
  )

  plan = build_plan(model, mesh)
  for note in plan.notes:
    print(f"[sharding] {note}", flush=True)
  print(
      f"[sharding] matched={len(plan.buckets['matched'])} "
      f"replicated={len(plan.buckets['replicated'])}",
      flush=True,
  )
  apply_sharding(model, plan)

  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, input_ids_j, labels_j):
    m = nnx.merge(graphdef, params, rest)
    logits = m(input_ids_j)
    return cross_entropy_jax(logits, labels_j), logits

  value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
  in_shard = input_sharding(mesh)
  input_ids_j = jax.device_put(
      np.asarray(input_ids.numpy(), dtype=np.int32), in_shard
  )
  labels_j = jax.device_put(
      np.asarray(labels.numpy(), dtype=np.int32), in_shard
  )

  mesh_cm = (
      jax.sharding.use_mesh(mesh) if hasattr(jax.sharding, "use_mesh") else mesh
  )
  with mesh_cm:
    (loss_jax, logits_jax), grads = value_and_grad(
        params, input_ids_j, labels_j
    )
  jax.block_until_ready(loss_jax)
  print(f"[jax-tpu] loss={float(loss_jax):.6f}", flush=True)

  if args.dtype == "bf16":
    fwd_atol = 3e-2 if args.fwd_atol is None else args.fwd_atol
    fwd_rtol = 3e-2 if args.fwd_rtol is None else args.fwd_rtol
    bwd_atol = 5e-2 if args.bwd_atol is None else args.bwd_atol
    bwd_rtol = 5e-2 if args.bwd_rtol is None else args.bwd_rtol
  else:
    fwd_atol = 2e-3 if args.fwd_atol is None else args.fwd_atol
    fwd_rtol = 2e-3 if args.fwd_rtol is None else args.fwd_rtol
    bwd_atol = 2e-2 if args.bwd_atol is None else args.bwd_atol
    bwd_rtol = 2e-2 if args.bwd_rtol is None else args.bwd_rtol

  ok = True
  print("\nFORWARD / LOSS:", flush=True)
  ok &= summarize(
      "logits", logits_jax, logits_ref.detach().numpy(), fwd_atol, fwd_rtol
  )
  ok &= summarize(
      "loss",
      np.array(float(loss_jax)),
      np.array(loss_ref.item()),
      fwd_atol,
      fwd_rtol,
  )

  print("\nBACKWARD GRADIENTS:", flush=True)
  grad_model = nnx.merge(graphdef, grads, rest)
  for name, grad_ref in grads_ref.items():
    grad_jax = get_param(grad_model, name)[...]
    ok &= summarize(name, grad_jax, grad_ref, bwd_atol, bwd_rtol)

  print(
      "\n"
      + (
          "==== TPU EQUIVALENCE PASS ===="
          if ok
          else "==== TPU EQUIVALENCE FAIL ===="
      ),
      flush=True,
  )
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
