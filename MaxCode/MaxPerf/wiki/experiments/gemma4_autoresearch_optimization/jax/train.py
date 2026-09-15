#!/usr/bin/env python
# pylint: skip-file
"""Gemma 4 E4B fine-tune trainer — **native JAX** (Flax NNX).

Mirrors the torchax trainer's CLI, summary block, sharding strategy, and
baseline behaviour so `awk '/step  2/,/step 19/'` comparisons keep working.
The model is the port in `model/modeling_gemma4.py`; weights are loaded
from HuggingFace safetensors into the NNX tree via `model/weight_loader.py`.

Differences from the torchax path:
  - no torch / torchax dependency at run time; only jax + flax + optax.
  - attention is a plain XLA SDPA matmul path. Splash Pallas is a
    follow-up — see the `_attn_xla_sdpa` comment in the model file.
  - weights are loaded in bf16 directly into sharded jax.Arrays.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional

DEFAULTS = {
    "model_id": "google/gemma-4-E4B",
    "dataset": "wikitext-2-raw-v1",
    # Exp 52+: new regime default is seq_len=8192 with fp32 master weights
    # + bf16 compute (mixed precision). Pass --dtype bf16 to fall back to
    # the legacy single-dtype (bf16 everywhere) path.
    "seq_len": 8192,
    "batch_size": 4,
    "steps": 20,
    "learning_rate": 1e-5,
    "warmup_steps": 2,
    "strategy": "fsdp",
    "dp": 1,
    "tp": 1,
    "fsdp": 0,
    "dtype": None,  # legacy single-dtype shortcut; None = use split below
    "weights_dtype": "fp32",
    "compute_dtype": "bf16",
    "log_every": 1,
    "grad_accum": 1,
    "seed": 42,
}


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      prog="gemma4-jax-trainer",
      description="Gemma 4 E4B fine-tune (native JAX / Flax NNX).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--model_id", default=DEFAULTS["model_id"])
  p.add_argument("--dataset", default=DEFAULTS["dataset"])
  p.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
  p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
  p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
  p.add_argument(
      "--learning_rate", type=float, default=DEFAULTS["learning_rate"]
  )
  p.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
  p.add_argument(
      "--strategy", choices=["fsdp", "tp"], default=DEFAULTS["strategy"]
  )
  p.add_argument("--fsdp", type=int, default=DEFAULTS["fsdp"])
  p.add_argument("--dp", type=int, default=DEFAULTS["dp"])
  p.add_argument("--tp", type=int, default=DEFAULTS["tp"])
  # Exp 52+: mixed-precision flags (fp32 master + bf16 compute).
  # --dtype is kept as a legacy shortcut that sets BOTH --weights-dtype
  # and --compute-dtype to the given value. Passing --dtype explicitly
  # overrides the --weights-dtype / --compute-dtype defaults.
  p.add_argument(
      "--dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["dtype"],
      help=(
          "Legacy single-dtype shortcut. If set, overrides both "
          "--weights-dtype and --compute-dtype. Default is unset: use "
          "the split flags below (fp32 master + bf16 compute)."
      ),
  )
  p.add_argument(
      "--weights-dtype",
      dest="weights_dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["weights_dtype"],
      help=(
          "Storage dtype for model params & optimizer state. Default fp32 "
          "(master weights). Set to bf16 to match the legacy pre-exp-52 mode."
      ),
  )
  p.add_argument(
      "--compute-dtype",
      dest="compute_dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["compute_dtype"],
      help="Compute dtype for matmul/conv activations. Default bf16.",
  )
  p.add_argument("--checkpoint_dir", default=None)
  p.add_argument("--profile_dir", default=None)
  p.add_argument("--profile_steps", type=int, nargs="*", default=[])
  p.add_argument("--log_every", type=int, default=DEFAULTS["log_every"])
  p.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
  p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
  p.add_argument(
      "--config",
      default=None,
      help="YAML file with default values (CLI overrides file).",
  )
  return p


def _load_yaml_defaults(path: str) -> dict:
  import yaml

  with open(path) as f:
    return yaml.safe_load(f) or {}


def main(argv: Optional[list] = None) -> int:
  args = _build_argparser().parse_args(argv)
  if args.config:
    cfg = _load_yaml_defaults(args.config)
    for k, v in cfg.items():
      if getattr(args, k, None) == DEFAULTS.get(k):
        setattr(args, k, v)

  random.seed(args.seed)
  os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

  import numpy as np
  import jax
  import jax.numpy as jnp
  import optax
  from flax import nnx
  from jax.sharding import NamedSharding, PartitionSpec as P
  from transformers import AutoConfig, AutoTokenizer

  # Module-local imports.
  this_dir = Path(__file__).resolve().parent
  sys.path.insert(0, str(this_dir))
  from model.modeling_gemma4 import Gemma4ForCausalLM
  from model.weight_loader import load_hf_weights
  from model.sharding import (
      get_mesh,
      get_param_sharding,
      apply_sharding,
      input_sharding,
      replicated as replicated_sharding,
      _iter_params,
  )
  from data import IGNORE_INDEX, make_dataloader

  np.random.seed(args.seed)

  # Resolve weights_dtype / compute_dtype. --dtype is a legacy single-dtype
  # shortcut that, when set, overrides the split flags.
  def _to_jnp_dtype(name: str):
    return jnp.bfloat16 if name == "bf16" else jnp.float32

  if args.dtype is not None:
    weights_dtype_name = args.dtype
    compute_dtype_name = args.dtype
    print(
        f"[dtype] legacy --dtype {args.dtype} -> weights={args.dtype}"
        f" compute={args.dtype}"
    )
  else:
    weights_dtype_name = args.weights_dtype
    compute_dtype_name = args.compute_dtype
    print(
        f"[dtype] weights={weights_dtype_name} compute={compute_dtype_name} "
        f"({'mixed-precision AMP' if weights_dtype_name != compute_dtype_name else 'single-dtype'})"
    )

  weights_dtype = _to_jnp_dtype(weights_dtype_name)
  compute_dtype = _to_jnp_dtype(compute_dtype_name)
  # Back-compat variable name used downstream (sharding etc. read this
  # as the "primary" dtype). Use weights_dtype so sharding treats
  # storage correctly.
  jnp_dtype = weights_dtype

  # Mesh --------------------------------------------------------------------
  if args.strategy == "fsdp":
    fsdp_size = args.fsdp or jax.device_count()
    mesh = get_mesh("fsdp", fsdp=fsdp_size)
    print(f"[mesh] strategy=fsdp fsdp={fsdp_size} devices={jax.device_count()}")
  else:
    mesh = get_mesh("tp", dp=args.dp, tp=args.tp)
    print(
        f"[mesh] strategy=tp dp={args.dp} tp={args.tp}"
        f" devices={jax.device_count()}"
    )

  # Register the mesh with the pallas-attention module so the splash
  # kernel's shard_map sees a concrete Mesh (Mosaic custom-calls cannot be
  # auto-partitioned). No-op when JAX_ATTENTION_IMPL != "splash".
  attn_impl = os.environ.get("JAX_ATTENTION_IMPL", "xla").lower()
  if attn_impl == "splash":
    from model.pallas_attention import set_mesh as _set_splash_mesh

    _set_splash_mesh(mesh)
    print(f"[attn] JAX_ATTENTION_IMPL=splash — pallas splash_attention enabled")
  else:
    print(f"[attn] JAX_ATTENTION_IMPL={attn_impl} — XLA SDPA (baseline)")

  # Load HF config (text-only sub-config). ----------------------------------
  print(
      f"[load] {args.model_id} (weights={weights_dtype_name}"
      f" compute={compute_dtype_name})"
  )
  t0 = time.perf_counter()
  hf_config = AutoConfig.from_pretrained(args.model_id)
  text_cfg = (
      hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
  )
  tokenizer = AutoTokenizer.from_pretrained(args.model_id)

  # Build the NNX model with random init (will be overwritten by loader).
  # Under fp32-master, init directly into fp32 would put the full
  # (vocab=262144 × n_layers=42 × 256) fp32 PLE embedding — ~11 GiB — on
  # a single device before sharding can apply, blowing past the 31 GiB
  # per-chip HBM at 3.55 GiB free (model init runs inline on device 0
  # for nnx). Fix: init the *storage* in the compute_dtype (bf16) —
  # which is at most 5.25 GiB for the biggest tensor and fits — then
  # apply sharding, then have the weight loader overwrite each param
  # value with a fp32 array that's scatter-sharded directly (see
  # ``shardings=`` kwarg below). Post-load, params are fp32 + sharded.
  rngs = nnx.Rngs(args.seed)
  _init_weights_dtype = compute_dtype  # always bf16 for init-time storage
  model = Gemma4ForCausalLM(
      text_cfg,
      weights_dtype=_init_weights_dtype,
      compute_dtype=compute_dtype,
      rngs=rngs,
  )
  # If we're really fp32-master, fix up the per-module weights_dtype
  # metadata so any later introspection sees the true target. (The
  # running param values will become fp32 after the loader pass.)
  if weights_dtype != _init_weights_dtype:

    def _fixup_dtype_meta(m):
      for name in ("weights_dtype",):
        if hasattr(m, name):
          setattr(m, name, weights_dtype)
      for _, v in vars(m).items():
        if isinstance(v, nnx.Module):
          _fixup_dtype_meta(v)
        elif isinstance(v, list):
          for elem in v:
            if isinstance(elem, nnx.Module):
              _fixup_dtype_meta(elem)

    _fixup_dtype_meta(model)

  # Pre-build splash kernels (one per (sliding_window, head_dim) combo).
  # This materializes MaskInfo at top-level so the lru_cache entries aren't
  # captured inside the top-level jitted_step's trace (which would leak
  # tracers on subsequent step-1 retrace). See pallas_attention.py docstring.
  if attn_impl == "splash":
    from model.pallas_attention import _build_splash_kernel as _pre_build
    from model.pallas_attention import _splash_config_key

    n_q = text_cfg.num_attention_heads
    sw = text_cfg.sliding_window
    sliding_hd = text_cfg.head_dim
    full_hd = text_cfg.global_head_dim or text_cfg.head_dim
    cfg_key = _splash_config_key()
    _pre_build(args.seq_len, n_q, sw, sliding_hd, cfg_key)
    _pre_build(args.seq_len, n_q, None, full_hd, cfg_key)
    print(
        f"[attn] pre-built splash kernels for seq={args.seq_len} "
        f"num_q_heads={n_q} sliding={sw}@hd={sliding_hd} full@hd={full_hd}"
    )
    # Log the active splash config for reproducibility (exp 48+)
    import os as _os

    _splash_env_snapshot = {
        k: _os.environ.get(k)
        for k in (
            "SPLASH_BLOCK_Q",
            "SPLASH_BLOCK_KV",
            "SPLASH_BLOCK_KV_COMPUTE",
            "SPLASH_BLOCK_Q_DKV",
            "SPLASH_BLOCK_KV_DKV",
            "SPLASH_BLOCK_KV_DKV_COMPUTE",
            "SPLASH_USE_FUSED_BWD",
            "SPLASH_QKV_LAYOUT",
        )
        if _os.environ.get(k) is not None
    }
    if _splash_env_snapshot:
      print(f"[attn] splash overrides: {_splash_env_snapshot}")
  # Sharding first — exp 52+ reorder. Plan is shape-based so it can run
  # on the still-bf16-init'd params; applying it ahead of the loader means
  # fp32-master tensors are scatter-sharded at load time and never
  # materialize the full [vocab, L*D_ple] fp32 tensor (~11 GiB) on any
  # single device.
  plan = get_param_sharding(model, mesh)
  for note in plan.notes:
    print(f"[sharding] {note}")
  counts = {k: len(v) for k, v in plan.buckets.items()}
  print(f"[sharding] bucket counts: {counts}")
  apply_sharding(model, plan)

  # Weight load (with sharding lookup so each param lands sharded).
  stats = load_hf_weights(
      model,
      args.model_id,
      weights_dtype=weights_dtype,
      shardings=plan.shardings,
      verbose=True,
  )
  print(
      f"[load] weights: assigned={stats['assigned']} "
      f"skipped_modality={stats['skipped_modality']} "
      f"skipped_shared_kv={stats['skipped_shared_kv']} "
      f"missing={stats['missing']} "
      "(missing should be 0 or 2 — vision/audio embed connectors)"
  )
  print(f"[load] done in {time.perf_counter() - t0:.1f}s")

  # Split the NNX model into (graphdef, state) so jit sees a pytree.
  # graphdef is static; state carries the Param values.
  graphdef, state = nnx.split(model)
  # Keep a reference to the text config for helpful prints.
  vocab_size = text_cfg.vocab_size

  # Optimizer ---------------------------------------------------------------
  lr_schedule = optax.warmup_constant_schedule(
      init_value=0.0,
      peak_value=args.learning_rate,
      warmup_steps=args.warmup_steps,
  )
  optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
  opt_state = optimizer.init(state)

  # Data --------------------------------------------------------------------
  if args.strategy == "fsdp":
    fsdp_size = args.fsdp or jax.device_count()
    global_bs = args.batch_size * fsdp_size
    print(
        f"[data] dataset=wikitext/{args.dataset} seq_len={args.seq_len}"
        f" per_chip_batch={args.batch_size} global_batch={global_bs}"
        f" (fsdp={fsdp_size})"
    )
  else:
    global_bs = args.batch_size * args.dp
    print(
        f"[data] dataset=wikitext/{args.dataset} seq_len={args.seq_len}"
        f" per_chip_batch={args.batch_size} global_batch={global_bs}"
        f" (dp={args.dp})"
    )
  data_iter = make_dataloader(
      seq_len=args.seq_len,
      batch_size=global_bs,
      tokenizer=tokenizer,
      dataset_config=args.dataset,
  )

  in_shard = input_sharding(mesh)

  # CE dtype gate (exp 37). Default bf16: logits stay bf16, log_softmax runs
  # in bf16, no intermediate fp32 [B*S, V] materialization (V=262144 ~ 1.5 GiB
  # at b=3 s=1024). Gemma 4's final_logit_softcapping=30.0 bounds logits to
  # ±30, well within bf16's range — verified numerically stable by torchax
  # exp 12 (https://.../torchax/experiments/2026-04-23-exp12-bf16-ce-accepted.md).
  # Set JAX_CE_DTYPE=fp32 to restore the upcast path for comparison.
  ce_dtype_env = os.environ.get("JAX_CE_DTYPE", "bf16").lower()
  if ce_dtype_env == "fp32":
    ce_dtype = jnp.float32
    print(f"[ce] JAX_CE_DTYPE=fp32 — logits upcast to fp32 before log_softmax")
  else:
    ce_dtype = jnp.bfloat16
    print(
        f"[ce] JAX_CE_DTYPE=bf16 — log_softmax in bf16 (default; ~1.5 GiB"
        f" saved)"
    )

  # CE impl gate (exp 47). Default = materialized bf16 path (exp 36 best).
  # `JAX_CE_IMPL=levanter` loads marin/levanter's Pallas TPU fused CE kernel
  # (softcap + log_softmax + NLL, no [B,S,V] logits materialization).
  ce_impl_env = os.environ.get("JAX_CE_IMPL", "default").lower()
  levanter_ce_fn = None
  if ce_impl_env == "levanter":
    # Disable the on-miss autotune path (needs a `rigging` GCS writer we
    # don't ship); we pass explicit block sizes for our shape instead.
    os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
    from model.kernels.fused_ce import load_kernel

    levanter_ce_fn = load_kernel()
    from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes as _LevBlocks
    # Hand-picked block sizes for Gemma 4 E4B on v6e (V=262144, H=2560).
    # Default (1024, 512, 1024) overruns 32 MiB VMEM; (128, 256, 512) fits
    # and keeps the kernel streaming (parity-verified bf16 diff ~0.05).
    # TPU-label-layout invariant: b_block_size must be a multiple of 1024
    # when the per-shard batch B >= 1024 (validated by the kernel's
    # `_validate_inputs`). Our per-device flat batch is B*S = 3*1024 = 3072,
    # so we use 1024.
    _levanter_block_sizes = _LevBlocks(
        b_block_size=1024,
        h_block_size=256,
        v_block_size=512,
    )
    softcap_value = float(text_cfg.final_logit_softcapping or 0.0) or None
    print(
        "[ce] JAX_CE_IMPL=levanter — fused Pallas CE w/"
        f" softcap={softcap_value}"
        f" block_sizes=(b={_levanter_block_sizes.b_block_size},"
        f" h={_levanter_block_sizes.h_block_size},"
        f" v={_levanter_block_sizes.v_block_size})"
    )
  else:
    _levanter_block_sizes = None
    softcap_value = None
    print(
        f"[ce] JAX_CE_IMPL={ce_impl_env} — materialized log_softmax path (exp"
        " 36 default)"
    )

  use_levanter_ce = levanter_ce_fn is not None

  # Shard_map wrapper is required because Mosaic custom calls cannot be
  # auto-partitioned (same pattern as splash_attention in pallas_attention.py).
  # In-specs pin hidden and labels to the per-chip FSDP batch shard and
  # replicate the lm_head weight (transposed to [H, V]) — XLA will insert
  # the all-gather on w_hv, same cost the materialized lm_head matmul
  # already pays in the exp-36 path.
  if use_levanter_ce:
    from jax.sharding import PartitionSpec as P

    def _levanter_ce_sharded(flat_hidden, safe_labels, mask, w_hv):
      def _kernel_call(fh, sl, msk, w):
        local_sum = levanter_ce_fn(
            fh,
            sl,
            w,
            reduction="sum",
            weight=msk,
            logit_soft_cap=softcap_value,
            implementation="pallas_tpu",
            dtype=jnp.bfloat16,
            block_sizes=_levanter_block_sizes,
        )
        # Sum partial loss_sums across the FSDP shards so the caller's
        # scalar division-by-mask-sum matches the global mean.
        return jax.lax.psum(local_sum, axis_name="fsdp")

      return jax.shard_map(
          _kernel_call,
          mesh=mesh,
          in_specs=(
              P("fsdp", None),  # flat_hidden: [B*S, H] sharded on B
              P("fsdp"),  # safe_labels: [B*S] sharded on B
              P("fsdp"),  # mask: [B*S] sharded on B
              P(None, None),  # w_hv: [H, V] replicated (all-gathered)
          ),
          out_specs=P(),  # scalar loss — psum makes it replicated.
          check_vma=False,
      )(flat_hidden, safe_labels, mask, w_hv)

  # Pure-jax forward+loss, closing over graphdef + ce_dtype (static). -------
  def forward_loss(state, input_ids, labels):
    model = nnx.merge(graphdef, state)
    if use_levanter_ce:
      # Exp 47 path: skip lm_head+softcap in the model; fused kernel
      # recomputes hidden @ W.T streaming with softcap inline.
      hidden = model(input_ids, return_hidden=True)  # (B, T, D) bf16
      W = model.lm_head_weight()  # (V, H) bf16 tied to embed
      B_, S_, D_ = hidden.shape
      flat_hidden = hidden.reshape(B_ * S_, D_)
      flat_labels = labels.reshape(-1)
      mask = (flat_labels != IGNORE_INDEX).astype(jnp.float32)
      safe_labels = jnp.where(
          flat_labels == IGNORE_INDEX, jnp.zeros_like(flat_labels), flat_labels
      )
      # Transpose [V, H] -> [H, V] for levanter's kernel layout.
      w_hv = W.T
      loss_sum = _levanter_ce_sharded(flat_hidden, safe_labels, mask, w_hv)
      loss = loss_sum / jnp.maximum(mask.sum(), 1.0)
      return loss.astype(jnp.float32)
    logits = model(input_ids)  # (B, T, V) — bf16 from the model forward
    # CE with ignore_index=-100. When ce_dtype=bf16 (default), log_softmax
    # runs entirely in bf16 — no fp32 [B*S, V] intermediate. Gemma 4's
    # softcap=30 keeps the softmax stable. The accumulator for the final
    # reduction is promoted to fp32 so cumulative error over B*S tokens
    # doesn't shift the loss value reported back to the optimizer schedule.
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab).astype(ce_dtype)
    flat_labels = labels.reshape(-1)
    mask = (flat_labels != IGNORE_INDEX).astype(flat_logits.dtype)
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    safe_labels = jnp.where(
        flat_labels == IGNORE_INDEX, jnp.zeros_like(flat_labels), flat_labels
    )
    picked = jnp.take_along_axis(
        log_probs, safe_labels[:, None], axis=-1
    ).squeeze(-1)
    # Promote the tiny scalar reduction to fp32 so the reported loss is
    # stable across token counts (negligible cost: one scalar div).
    loss = -(
        picked.astype(jnp.float32) * mask.astype(jnp.float32)
    ).sum() / jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
    return loss.astype(jnp.float32)

  from jax import checkpoint_policies as _ckpt_policies
  # Exp 52+: remat policy is env-selectable so the seq=8192 regime can
  # climb its HBM wall without a code patch per variant. Default
  # `dots_with_no_batch_dims` matches exp 36; `nothing_saveable` is
  # maximum-remat (recomputes everything on backward — highest TPS
  # penalty, lowest peak HBM). `everything_saveable` is minimum-remat
  # (fastest step, largest HBM).
  _remat_policy_name = os.environ.get(
      "JAX_REMAT_POLICY", "dots_with_no_batch_dims"
  ).lower()
  _policy_map = {
      "dots_with_no_batch_dims": (
          _ckpt_policies.checkpoint_dots_with_no_batch_dims
      ),
      "nothing_saveable": _ckpt_policies.nothing_saveable,
      "everything_saveable": _ckpt_policies.everything_saveable,
      "dots": _ckpt_policies.checkpoint_dots,
      # Host-offload dot-with-no-batch-dims: stashes saved dot outputs to
      # host RAM during forward and streams back during backward. Trades
      # ICI/PCIe bandwidth for HBM headroom. Useful at the fp32-master
      # seq=8192 ceiling where the model is ~3 GiB over per-chip HBM.
      "offload_dot_with_no_batch_dims": (
          _ckpt_policies.offload_dot_with_no_batch_dims(
              "device",
              "pinned_host",
          )
      ),
  }
  _remat_policy = _policy_map.get(
      _remat_policy_name, _ckpt_policies.checkpoint_dots_with_no_batch_dims
  )
  print(f"[remat] JAX_REMAT_POLICY={_remat_policy_name}")

  # Under JAX_SCAN_LAYERS=1 the scan bodies carry their own per-iter
  # `jax.checkpoint(..., policy=checkpoint_dots_with_no_batch_dims)`;
  # wrapping forward_loss in an *outer* checkpoint as well forces
  # double-remat (nested `jax.checkpoint`s replay forward twice on
  # backward). Under the exp-36 Python for-loop path, this outer
  # checkpoint is the ONLY checkpoint and is required — the for-loop
  # would otherwise store every layer's full activation stack.
  if os.environ.get("JAX_SCAN_LAYERS") == "1":
    grad_fn = jax.value_and_grad(forward_loss)
  else:
    checkpointed = jax.checkpoint(forward_loss, policy=_remat_policy)
    grad_fn = jax.value_and_grad(checkpointed)

  def train_step(state, opt_state, input_ids, labels):
    with jax.named_scope("train_step"):
      with jax.named_scope("forward_backward"):
        loss, grads = grad_fn(state, input_ids, labels)
      with jax.named_scope("optimizer"):
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
    return loss, state, opt_state

  jitted_step = jax.jit(train_step, donate_argnums=(0, 1))

  # Training loop -----------------------------------------------------------
  print(f"[train] steps={args.steps} grad_accum={args.grad_accum}")
  prof_steps = set(args.profile_steps)
  prof_start = min(prof_steps) if prof_steps else None
  prof_stop = max(prof_steps) if prof_steps else None
  profile_dir = args.profile_dir
  if profile_dir:
    Path(profile_dir).mkdir(parents=True, exist_ok=True)

  step_times = []
  compile_time = None
  profile_active = False

  # Enter the mesh so that jit sees the right sharding environment.
  if hasattr(jax.sharding, "use_mesh"):
    _mesh_cm = jax.sharding.use_mesh(mesh)
    _mesh_cm.__enter__()
  else:
    mesh.__enter__()
  t_start = time.perf_counter()
  for step in range(args.steps):
    try:
      input_ids_np, labels_np = next(data_iter)
    except StopIteration:
      print(f"[data] exhausted at step {step}; stopping.")
      break
    input_ids = jax.device_put(input_ids_np, in_shard)
    labels = jax.device_put(labels_np, in_shard)

    if (
        profile_dir
        and prof_start is not None
        and step == prof_start
        and not profile_active
    ):
      print(f"[profile] start_trace -> {profile_dir}")
      jax.profiler.start_trace(profile_dir)
      profile_active = True

    with jax.profiler.TraceAnnotation(f"train_step_{step}"):
      t_step = time.perf_counter()
      loss, state, opt_state = jitted_step(state, opt_state, input_ids, labels)
      jax.block_until_ready(loss)
      dt = time.perf_counter() - t_step

    if step == 0:
      compile_time = dt
      print(f"[compile] step 0 (compile + first exec): {dt:.2f}s")
    else:
      step_times.append(dt)

    if step % args.log_every == 0:
      loss_f = float(jax.device_get(loss))
      print(f"[step {step:4d}] loss={loss_f:.4f}  dt={dt*1000:.1f}ms")

    if (
        profile_dir
        and prof_stop is not None
        and step == prof_stop
        and profile_active
    ):
      jax.block_until_ready(loss)
      jax.profiler.stop_trace()
      profile_active = False
      print(f"[profile] stop_trace (captured steps {prof_start}..{prof_stop})")

  if profile_active:
    jax.profiler.stop_trace()

  wall = time.perf_counter() - t_start

  # Report ------------------------------------------------------------------
  tokens_per_step = global_bs * args.seq_len
  if step_times:
    mean_dt = sum(step_times) / len(step_times)
    tps = tokens_per_step / mean_dt
  else:
    mean_dt = float("nan")
    tps = float("nan")

  print("\n================ summary ================")
  print(
      f"compile time (step 0) : {compile_time:.2f}s"
      if compile_time is not None
      else "compile time          : (no step ran)"
  )
  print(f"steps measured        : {len(step_times)}")
  print(f"mean step time        : {mean_dt*1000:.1f} ms")
  print(f"tokens per step       : {tokens_per_step}")
  print(f"tokens / sec          : {tps:.0f}")
  print(f"wall clock            : {wall:.1f}s")
  print("==========================================")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
