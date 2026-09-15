# pylint: skip-file
"""Llama 3 8B native-JAX (Flax NNX) trainer.

Mirrors the torchax sibling at `../torchax/train.py` flag-for-flag and
metric-for-metric so the two stacks can be A/B'd directly. Differences:

  - **No torchax / no torch** at run time. The model is the Flax NNX port
    in `model/modeling_llama3.py`; weights load straight from HF
    safetensors via `model/weight_loader.py`.
  - Optimizer is `optax.adamw` (mu_dtype=fp32 when `--master_dtype=fp32`).
  - The train_step is a plain `jax.jit`'d function with
    `donate_argnums=(0, 1)` (state, opt_state) so XLA can in-place update.
  - Tokamax CE goes through the same `shard_map` wrapper as the torchax
    sibling; `chunked_xla` requires the fp32 cast at the boundary
    (validated path; bf16 destroys precision — exp 66 INVALID).

Compile cache: set ``JAX_COMPILATION_CACHE_DIR`` to a persistent path so
cold-compile cost amortizes across runs.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import sys
import time
from typing import Optional

# Stub `jaxtyping.jaxtyped` to avoid typeguard AST-walk crashes on tokamax's
# `*B T H d` annotations under py3.13. Must run before any tokamax import.
try:
  import jaxtyping as _jt

  _jt.jaxtyped = lambda typechecker=None: (lambda fn: fn)
except ImportError:
  pass

# tokamax/_src/config.py reads sys.argv via `flags.FLAGS(sys.argv)` on first
# config-option access; this trainer uses `fire.Fire`, so absl flags would
# fail on `--model_id=...`. Pre-parse absl with only argv[0] so tokamax's
# `is_parsed()` short-circuits.
try:
  import sys as _sys
  from absl import flags as _absl_flags

  if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS([_sys.argv[0]])
except Exception:
  pass

import fire
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

# Make the local `splash_attn.py` and `data.py` importable when run as
# `python -m train`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
  sys.path.insert(0, str(_THIS_DIR))


def _to_jnp_dtype(name: str) -> jnp.dtype:
  return jnp.bfloat16 if name == "bf16" else jnp.float32


# MaxText-style named-checkpoint sets; mirror `decoders.py:393-435` so a
# single CLI flag selects the same policy by name as MaxText.
_SAVE_QKV_PROJ_NAMES = ("query_proj", "key_proj", "value_proj")
_SAVE_QKV_OUT_NAMES = (*_SAVE_QKV_PROJ_NAMES, "out_proj")
_SAVE_DOT_EXCEPT_MLP_NAMES = (
    *_SAVE_QKV_OUT_NAMES,
)  # MLP layers are recomputed
_OFFLOAD_QKV_NAMES = _SAVE_QKV_PROJ_NAMES
_MINIMAL_OFFLOAD_NAMES = (*_SAVE_QKV_OUT_NAMES, "mlpwi_0", "mlpwi_1", "mlpwo")


def _resolve_scan_policy(name: str):
  """Resolve a `scan_remat_policy` name to a jax.checkpoint policy.

  Recognises (a) every attribute of `jax.checkpoint_policies` (e.g.
  `nothing_saveable`, `dots_saveable`, `everything_saveable`) and
  (b) MaxText's named-set policies (`save_qkv_proj`, `save_out_proj`,
  `save_dot_except_mlp`, `qkv_proj_offloaded`, `minimal_offloaded`).
  """
  cp = jax.checkpoint_policies
  if name == "save_qkv_proj":
    return cp.save_only_these_names(*_SAVE_QKV_PROJ_NAMES)
  if name == "save_out_proj":
    return cp.save_only_these_names("out_proj")
  if name == "save_dot_except_mlp":
    return cp.save_only_these_names(*_SAVE_DOT_EXCEPT_MLP_NAMES)
  if name == "qkv_proj_offloaded":
    return cp.save_and_offload_only_these_names(
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=list(_OFFLOAD_QKV_NAMES),
        offload_src="device",
        offload_dst="pinned_host",
    )
  if name == "minimal_offloaded":
    return cp.save_and_offload_only_these_names(
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=list(_MINIMAL_OFFLOAD_NAMES),
        offload_src="device",
        offload_dst="pinned_host",
    )
  return getattr(cp, name)


def main(
    model_id: str = "meta-llama/Meta-Llama-3-8B",
    batch_size: int = 4,
    seqlen: int = 1024,
    train_steps: int = 15,
    tp_parallelism: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    weights_dtype: str = "bf16",
    compute_dtype: str = "match",
    use_real_data: bool = True,
    load_pretrained_weights: bool = False,  # False = random init via Flax NNX
    # (matches torchax trainer for perf
    # benchmarking; True = load HF
    # safetensors via weight_loader)
    use_splash: bool = False,
    use_scan: bool = False,
    use_per_layer_remat: bool = False,
    master_dtype: str = "match",
    scan_remat_policy: str = "nothing_saveable",
    use_tokamax_ce: bool = False,
    tokamax_ce_impl: str = "mosaic_tpu",
    tokamax_ce_autotune: bool = False,
    profile_dir: Optional[str] = None,
    profile_step: int = 5,
):
  """Native-JAX Llama 3 8B trainer. Same CLI as the torchax sibling.

  Args:
    model_id: HF repo id for weights.
    batch_size: per-chip batch.
    seqlen: sequence length.
    train_steps: number of step iterations (step 0 compiles).
    tp_parallelism: 1 = pure FSDP across all chips; >1 = 2D (fsdp, tp) mesh.
      learning_rate / weight_decay: optax.adamw args.
    weights_dtype: storage dtype for params + optimizer state. Default bf16; set
      "fp32" for AMP master.
    compute_dtype: dtype for matmul/conv activations. "match" inherits
      weights_dtype; "bf16" with weights_dtype="fp32" enables true
      mixed-precision.
    use_real_data: True = wikitext-2-raw-v1; False = random tokens (perf-only
      smoke test).
    use_splash: install splash attention via env-gated dispatch in the model.
      Trainer also calls `set_splash_mesh(mesh)`.
    use_scan: use LlamaForCausalLMScan (32 layers stacked into one scan body).
      Confirmed by torchax exp 60/78 to be the only fitting remat at bs=3
      seq=8192.
    use_per_layer_remat: wrap each LlamaDecoderLayer.__call__ in a per-layer
      `jax.checkpoint` (only effective with use_scan=False).
    master_dtype: "fp32" forces optimizer mu/nu to fp32 even when weights are
      bf16 (standard mixed-precision). "match" inherits weights_dtype.
    scan_remat_policy: jax.checkpoint_policies attribute name. Default
      nothing_saveable — recompute everything on backward (validated by torchax
      exp 60/78 as the only one that fits at bs=3 seq=8192).
    use_tokamax_ce: True = use tokamax.linear_softmax_cross_entropy_loss (Pallas
      streamed CE; ~256 MiB activation savings at seq=8192).
    tokamax_ce_impl: mosaic_tpu | chunked_xla | xla. chunked_xla is the
      validated faster path (exp 62b, +1.6% over mosaic_tpu) but requires fp32
      inputs (handled internally).
    tokamax_ce_autotune: True = autotune CE block sizes
      (autotuning_cache_miss_fallback="autotune"). profile_dir / profile_step:
      capture an xprof trace at the given step.
  """
  n_global = jax.device_count()
  n_local = jax.local_device_count()
  n_hosts = jax.process_count()
  print(
      f"[dist] global_devices={n_global} local_devices={n_local}"
      f" hosts={n_hosts}",
      flush=True,
  )

  # -------------------------------------------------------------------
  # Mesh.
  # -------------------------------------------------------------------
  fsdp = n_global // tp_parallelism
  AxisType = jax.sharding.AxisType
  mesh = jax.make_mesh(
      (fsdp, tp_parallelism),
      ("fsdp", "tp"),
      axis_types=(AxisType.Auto, AxisType.Auto),
  )
  print(f"[mesh] fsdp={fsdp} tp={tp_parallelism} mesh={mesh}", flush=True)

  # -------------------------------------------------------------------
  # Dtypes.
  # -------------------------------------------------------------------
  weights_dtype_jax = _to_jnp_dtype(weights_dtype)
  if compute_dtype == "match":
    compute_dtype_jax = weights_dtype_jax
    compute_dtype_name = weights_dtype
  else:
    compute_dtype_jax = _to_jnp_dtype(compute_dtype)
    compute_dtype_name = compute_dtype
  amp = weights_dtype_jax != compute_dtype_jax
  print(
      f"[dtype] weights={weights_dtype} compute={compute_dtype_name} "
      f"({'mixed-precision AMP' if amp else 'single-dtype'})",
      flush=True,
  )

  # -------------------------------------------------------------------
  # Build model on bf16 storage (or directly on weights_dtype). For the
  # AMP fp32-master path we init in compute_dtype (bf16) so the random
  # construction step doesn't try to materialize 32 GiB of fp32 weights
  # on device 0. The weight loader then overwrites each tensor with the
  # fp32 HF value, scatter-sharded to its target NamedSharding.
  # -------------------------------------------------------------------
  init_storage_dtype = compute_dtype_jax if amp else weights_dtype_jax

  from transformers import AutoConfig, AutoTokenizer

  print(f"[load] config from {model_id} ...", flush=True)
  config = AutoConfig.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id) if use_real_data else None

  from model.modeling_llama3 import (
      LlamaForCausalLM,
      LlamaForCausalLMScan,
      set_splash_mesh,
  )
  from model.weight_loader import load_hf_weights
  from model.sharding import (
      build_plan,
      apply_sharding,
      input_sharding,
      _iter_params,
  )

  rngs = nnx.Rngs(0)

  if use_scan:
    scan_policy = _resolve_scan_policy(scan_remat_policy)
    print(
        "[scan] LlamaForCausalLMScan (32 layers stacked), "
        f"checkpoint_policy={scan_remat_policy}",
        flush=True,
    )
    model = LlamaForCausalLMScan(
        config,
        weights_dtype=init_storage_dtype,
        compute_dtype=compute_dtype_jax,
        scan_remat_policy=scan_policy,
        rngs=rngs,
    )
  else:
    model = LlamaForCausalLM(
        config,
        weights_dtype=init_storage_dtype,
        compute_dtype=compute_dtype_jax,
        rngs=rngs,
    )

  # Param-count sanity print.
  n_params = sum(int(p.value.size) for _, p in _iter_params(model))
  print(
      f"[load] model has {n_params/1e9:.2f} B parameters (NNX-side)", flush=True
  )

  # Splash attention setup (registers the mesh for the kernel's shard_map).
  if use_splash:
    os.environ["JAX_ATTENTION_IMPL"] = "splash"
    set_splash_mesh(mesh)
    print(
        "[attn] splash kernel selected (JAX_ATTENTION_IMPL=splash); "
        "mesh registered with model.set_splash_mesh()",
        flush=True,
    )
  else:
    os.environ["JAX_ATTENTION_IMPL"] = "xla"
    print("[attn] XLA SDPA (JAX_ATTENTION_IMPL=xla)", flush=True)

  # Per-layer remat (unscanned only).
  if use_per_layer_remat and not use_scan:
    # Wrap each LlamaDecoderLayer.__call__ in a per-layer jax.checkpoint
    # so XLA schedules them serially. We do this by patching the
    # __call__ on each layer instance.
    _policy = _resolve_scan_policy(scan_remat_policy)
    for i, layer in enumerate(model.model.layers):
      orig_call = layer.__call__

      @functools.wraps(orig_call)
      def _ckpt_call(*args, _orig=orig_call, _pol=_policy, **kw):
        return jax.checkpoint(_orig, policy=_pol)(*args, **kw)

      layer.__call__ = _ckpt_call
    print(
        "[remat] per-layer gradient_checkpoint installed "
        f"(policy={scan_remat_policy}, "
        f"{config.num_hidden_layers} scopes)",
        flush=True,
    )

  # -------------------------------------------------------------------
  # Build sharding plan, apply, then load weights into shards directly.
  # -------------------------------------------------------------------
  plan = build_plan(model, mesh, use_scan=use_scan)
  for note in plan.notes:
    print(f"[sharding] {note}", flush=True)
  print(
      f"[sharding] matched={len(plan.buckets['matched'])} "
      f"replicated={len(plan.buckets['replicated'])}",
      flush=True,
  )
  apply_sharding(model, plan)

  if load_pretrained_weights:
    t0 = time.perf_counter()
    stats = load_hf_weights(
        model,
        model_id,
        weights_dtype=weights_dtype_jax,
        shardings=plan.shardings,
        use_scan=use_scan,
        verbose=True,
    )
    print(
        f"[load] weights assigned={stats['assigned']} "
        f"missing={stats['missing']} "
        f"stacked_groups={stats.get('stacked_groups', 0)} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
  else:
    # Random init (matches torchax trainer's `make_array_from_callback`
    # path — random per-shard init, never materializes pretrained weights
    # on host). Performance numbers are identical to pretrained weights;
    # we benchmark the COMPUTE not the weights.
    print(
        "[load] using Flax NNX random init (load_pretrained_weights=False)",
        flush=True,
    )

  # -------------------------------------------------------------------
  # Optimizer.
  # -------------------------------------------------------------------
  master_dtype_jax = jnp.float32 if master_dtype == "fp32" else None
  optimizer = optax.adamw(
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      mu_dtype=master_dtype_jax,
  )
  print(
      f"[opt] adamw mu/nu_dtype={master_dtype} (weights_dtype={weights_dtype})",
      flush=True,
  )

  graphdef, state = nnx.split(model)
  opt_state = optimizer.init(state)

  # Replicate any opt_state leaves that didn't inherit a per-param
  # sharding (small scalars from optax).
  repl = NamedSharding(mesh, P())

  def _fix_leaf(leaf):
    if isinstance(leaf, jax.Array) and len(leaf.sharding.device_set) < n_global:
      return jax.device_put(leaf, repl)
    return leaf

  opt_state = jax.tree.map(_fix_leaf, opt_state)

  # -------------------------------------------------------------------
  # Data.
  # -------------------------------------------------------------------
  global_batch = batch_size * fsdp
  if use_real_data:
    from data import make_dataloader

    data_iter = make_dataloader(
        seq_len=seqlen,
        batch_size=global_batch,
        tokenizer=tokenizer,
    )
    print(
        f"[data] wikitext-2-raw-v1 global_batch={global_batch} seqlen={seqlen}",
        flush=True,
    )
  else:
    from data import fake_dataloader

    data_iter = fake_dataloader(
        train_steps + 5,
        seqlen,
        global_batch,
        vocab_size=config.vocab_size,
    )
    print(
        f"[data] fake (random ints) global_batch={global_batch} "
        f"seqlen={seqlen}",
        flush=True,
    )

  # -------------------------------------------------------------------
  # Loss + train_step.
  # -------------------------------------------------------------------
  # tokamax CE setup.
  levanter_block_sizes = None
  tokamax_mod = None
  if use_tokamax_ce:
    try:
      import tokamax as tokamax_mod
    except Exception as e:
      raise RuntimeError(
          "use_tokamax_ce=True but tokamax is not importable. "
          f"Original error: {e}"
      )
    # We need the lm_head weight in the loss path. Toggle the model's
    # skip_lm_head so the forward returns hidden states; the loss path
    # reads `model.lm_head.weight` from the live state inside the jit.
    if not use_scan:
      # Either path works; we just need skip_lm_head=True.
      pass
    # The flag is read inside __call__; flipping it here propagates to
    # the model object since graphdef/state were already split. Keep
    # both in sync.
    model.skip_lm_head = True
    graphdef, state = nnx.split(model)
    opt_state = jax.tree.map(_fix_leaf, optimizer.init(state))
    print(
        "[ce] tokamax.linear_softmax_cross_entropy_loss "
        f"impl={tokamax_ce_impl} autotune={tokamax_ce_autotune}",
        flush=True,
    )
  else:
    print("[ce] plain softmax_cross_entropy (logits materialized)", flush=True)

  def _ce_tokamax(hidden, labels, lm_head_w):
    """Tokamax fused CE — mirrors the torchax sibling exactly.

    fp32 cast at the boundary; shard_map; psum across fsdp.
    """
    B, L, H = hidden.shape
    BL = B * L
    h_flat = hidden.reshape(BL, H)
    l_flat = labels.reshape(BL)
    # lm_head weight is (V, H) — kernel wants (H, V).
    w_HV = lm_head_w.T
    # fp32 boundary cast (REQUIRED for chunked_xla; otherwise lse
    # accumulates in bf16 and loss collapses to bf16 quantization —
    # torchax exp 66 INVALID).
    h32 = h_flat.astype(jnp.float32)
    w32 = w_HV.astype(jnp.float32)

    from jax.experimental.shard_map import shard_map as _shard_map

    def _ce_local(h, l, w):
      if tokamax_ce_autotune:
        with tokamax_mod.config.autotuning_cache_miss_fallback("autotune"):
          s = tokamax_mod.linear_softmax_cross_entropy_loss(
              h, l, w, reduction="sum", implementation=tokamax_ce_impl
          )
      else:
        s = tokamax_mod.linear_softmax_cross_entropy_loss(
            h, l, w, reduction="sum", implementation=tokamax_ce_impl
        )
      return jax.lax.psum(s, axis_name="fsdp")

    ce_sm = _shard_map(
        _ce_local,
        mesh=mesh,
        in_specs=(P("fsdp", None), P("fsdp"), P()),
        out_specs=P(),
        check_rep=False,
    )
    return ce_sm(h32, l_flat, w32) / float(BL)

  def _ce_softmax(logits, labels):
    """Plain softmax cross-entropy mean. logits (B, L, V); labels (B, L)."""
    v = logits.shape[-1]
    flat_logits = logits.reshape(-1, v)
    flat_labels = labels.reshape(-1)
    # one_hot in compute_dtype is OK; reduce in fp32 for stability.
    log_probs = jax.nn.log_softmax(flat_logits.astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(
        log_probs, flat_labels[:, None], axis=-1
    ).squeeze(-1)
    return -picked.mean()

  def loss_fn(state, input_ids, labels):
    m = nnx.merge(graphdef, state)
    if use_tokamax_ce:
      hidden = m(input_ids, return_hidden=True)
      w = m.lm_head_weight()  # (V, H)
      loss = _ce_tokamax(hidden, labels, w)
    else:
      logits = m(input_ids)
      loss = _ce_softmax(logits, labels)
    return loss.astype(jnp.float32)

  grad_fn = jax.value_and_grad(loss_fn)

  def train_step(state, opt_state, input_ids, labels):
    with jax.named_scope("train_step"):
      with jax.named_scope("forward_backward"):
        loss, grads = grad_fn(state, input_ids, labels)
      with jax.named_scope("optimizer"):
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
    return loss, state, opt_state

  jitted_step = jax.jit(train_step, donate_argnums=(0, 1))

  in_shard = input_sharding(mesh)

  # -------------------------------------------------------------------
  # Training loop.
  # -------------------------------------------------------------------
  print(
      f"[train] starting train_steps={train_steps} "
      f"per_chip_batch={batch_size} global_batch={global_batch}",
      flush=True,
  )
  warmup_steps = 2
  total_tokens_after_warmup = 0
  total_time_after_warmup = 0.0
  n_measured_steps = 0
  profile_active = False
  if profile_dir:
    Path(profile_dir).mkdir(parents=True, exist_ok=True)

  if hasattr(jax.sharding, "use_mesh"):
    _mesh_cm = jax.sharding.use_mesh(mesh)
    _mesh_cm.__enter__()
  else:
    mesh.__enter__()

  for i in range(train_steps):
    try:
      input_ids_np, labels_np = next(data_iter)
    except StopIteration:
      print(f"[data] exhausted at step {i}; stopping.", flush=True)
      break
    tokens_this_step = input_ids_np.shape[0] * input_ids_np.shape[1]
    input_ids = jax.device_put(input_ids_np, in_shard)
    labels = jax.device_put(labels_np, in_shard)

    if profile_dir and i == profile_step and not profile_active:
      print(f"[profile] start_trace -> {profile_dir}", flush=True)
      jax.profiler.start_trace(profile_dir)
      profile_active = True

    t0 = time.perf_counter()
    loss, state, opt_state = jitted_step(state, opt_state, input_ids, labels)
    jax.block_until_ready(loss)
    dt = time.perf_counter() - t0

    if profile_dir and i == profile_step and profile_active:
      jax.profiler.stop_trace()
      profile_active = False
      print(f"[profile] stop_trace (captured step {profile_step})", flush=True)

    loss_f = float(jax.device_get(loss))
    tps = tokens_this_step / dt
    print(
        f"[step {i:2d}/{train_steps}] loss={loss_f:.4f} "
        f"step_time={dt*1000:.1f}ms throughput={tps:.0f} tok/s",
        flush=True,
    )

    if i >= warmup_steps:
      total_tokens_after_warmup += tokens_this_step
      total_time_after_warmup += dt
      n_measured_steps += 1

  if profile_active:
    jax.profiler.stop_trace()

  # -------------------------------------------------------------------
  # Summary block — same MFU formula as torchax/train.py lines 596-632.
  # -------------------------------------------------------------------
  if total_time_after_warmup > 0 and n_measured_steps > 0:
    avg_tps = total_tokens_after_warmup / total_time_after_warmup
    per_chip = avg_tps / n_global
    avg_step_time = total_time_after_warmup / n_measured_steps
    peak = 918e12  # v6e bf16 peak ≈ 918 TFLOPS / chip.
    # MaxText-style train-step TFLOPs (forward + 2× backward = ×3) for
    # a dense GQA Llama. Same formula as the torchax sibling.
    cfg = config
    B = global_batch / n_global
    L = seqlen
    D = cfg.hidden_size
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", D // Hq)
    Mlp = cfg.intermediate_size
    V = cfg.vocab_size
    nL = cfg.num_hidden_layers
    qkv_flops = 2 * B * L * D * (Hq + 2 * Hkv) * hd
    proj_flops = 2 * B * L * D * Hq * hd
    ffn_flops = 2 * B * L * Mlp * D * (2 + 1)
    embed_flops = 2 * B * L * D * V
    causal_attn_flops = 4 * B * L * L * Hq * hd / 2  # /2 for causal mask.
    learnable_tflops = (
        (ffn_flops * nL + (qkv_flops + proj_flops) * nL + embed_flops)
        * 3
        / 1e12
    )
    attention_tflops = causal_attn_flops * nL * 3 / 1e12
    total_tflops_per_step = learnable_tflops + attention_tflops
    mfu = (total_tflops_per_step * 1e12) / (avg_step_time * peak)
    print("\n================ summary ================", flush=True)
    print(f"global_batch          : {global_batch}", flush=True)
    print(f"seqlen                : {seqlen}", flush=True)
    print(f"steps measured        : {n_measured_steps}", flush=True)
    print(
        f"avg throughput        : {avg_tps:.0f} tok/s ({per_chip:.0f}/chip)",
        flush=True,
    )
    print(f"approx MFU            : {mfu*100:.1f}% (v6e bf16 peak)", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
  fire.Fire(main)
