"""Minimal Qwen3 native-JAX (Flax NNX) trainer.

Mirrors `../torchax/train.py` metric-for-metric (same MFU formula, same flags
where they overlap) so the two lanes A/B directly. NO torch / torchax at run
time — the model is the Flax NNX port in `model/modeling_qwen3.py`.

Deliberately minimal: no tokamax CE, no splash, no scan, no AMP — random-init
bf16 weights, FSDP sharding, optax AdamW, a single `jax.jit`'d train step
(`value_and_grad` over the NNX param state), synthetic data, plain softmax CE.
Each optimization lands later as its own attributable experiment.

Run (from this folder):
    python -u train.py --use_real_data False --seqlen 2048 --batch_size 1 \
        --train_steps 20 \
        --profile_dir gs://<bucket>/autoresearch/qwen3_cc/<run> \
        --profile_start_step 12 --profile_steps 3
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Optional

# tokamax's config resolves lazily via `flags.FLAGS(sys.argv)` on first
# config-option access (e.g. inside the mosaic_tpu CE op). This trainer uses
# fire.Fire, so sys.argv carries `--model_id=...` etc., which tokamax's absl
# parser doesn't recognize → UnrecognizedFlagError at trace time. Pre-parse absl
# with only argv[0] so tokamax's `is_parsed()` short-circuits. (Mirrors the
# llama3 jax/torchax trainers; required for any tokamax CE run — see v011.)
try:
  from absl import flags as _absl_flags

  if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS([sys.argv[0]])
except Exception:
  pass

import fire
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
  sys.path.insert(0, str(_THIS_DIR))


def _to_jnp_dtype(name: str) -> jnp.dtype:
  return jnp.bfloat16 if name == "bf16" else jnp.float32


# CE with a stable custom gradient, ported verbatim from MaxText
# (MaxText/max_utils.py:cross_entropy_with_logits, itself from T5X). The
# custom_vjp hands XLA an explicit fused backward (softmax - onehot) instead of
# letting autodiff differentiate through log_softmax — a lighter, single-pass
# backward graph over the [B,L,V] logits.
@jax.custom_vjp
def cross_entropy_with_logits(logits, targets, z_loss):
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits, targets, z_loss=0.0):
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,
      log_softmax,
      log_z,
  )


def _cross_entropy_with_logits_bwd(res, g):
  g = g[0]
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  deriv = (
      jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp
      - targets
  )
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (
      jnp.asarray(g_logits, logits.dtype),
      jnp.asarray(g_targets, targets.dtype),
      jnp.array(0.0),
  )


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd
)


def main(
    model_id: str = "Qwen/Qwen3-8B",
    batch_size: int = 1,  # per-fsdp-shard batch; global = batch × fsdp
    seqlen: int = 2048,
    train_steps: int = 20,
    tp_parallelism: int = 1,  # 1 = pure FSDP across all chips
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    weights_dtype: str = "bf16",
    use_remat: bool = False,  # per-layer jax.checkpoint (cuts activation HBM)
    offload_remat: bool = False,  # remat to host DRAM (pinned_host) instead of recompute
    use_scan: bool = False,  # lax.scan over the stacked decoder layer (1 compiled body)
    use_splash: bool = False,  # GQA-native splash attention (no N² scores)
    use_tokamax_ce: bool = False,  # streamed CE at lm_head (drops [B,L,V] logits)
    tokamax_ce_impl: str = "mosaic_tpu",  # mosaic_tpu | xla (this tokamax build's
    # valid impls; chunked_xla is NOT available
    # here — it crashed v010, see that exp page)
    use_maxtext_ce: bool = False,  # MaxText/T5X custom_vjp CE over full logits (no kernel)
    shard_acts: bool = False,  # pin activation layouts (MaxText with_logical_constraint)
    use_real_data: bool = False,  # False = synthetic tokens (perf baseline)
    # --- profiling (all CLI flags) ---
    profile_dir: Optional[str] = None,
    profile_gcs_dir: Optional[str] = None,
    profile_start_step: int = 8,
    profile_steps: int = 3,
    xprof_url_base: str = "http://localhost:8791",
):
  n_global = jax.device_count()
  n_local = jax.local_device_count()
  n_hosts = jax.process_count()
  print(
      f"[dist] global_devices={n_global} local_devices={n_local}"
      f" hosts={n_hosts}",
      flush=True,
  )

  fsdp = n_global // tp_parallelism
  AxisType = jax.sharding.AxisType
  mesh = jax.make_mesh(
      (fsdp, tp_parallelism),
      ("fsdp", "tp"),
      axis_types=(AxisType.Auto, AxisType.Auto),
  )
  print(f"[mesh] fsdp={fsdp} tp={tp_parallelism} mesh={mesh}", flush=True)

  wdtype = _to_jnp_dtype(weights_dtype)

  from transformers import AutoConfig

  print(f"[load] config from {model_id} ...", flush=True)
  config = AutoConfig.from_pretrained(model_id)

  from model import Qwen3ForCausalLM
  from model.sharding import build_plan, apply_sharding, input_sharding, _iter_params

  model = Qwen3ForCausalLM(
      config,
      weights_dtype=wdtype,
      compute_dtype=wdtype,
      rngs=nnx.Rngs(0),
      use_remat=use_remat,
      offload_remat=offload_remat,
      use_scan=use_scan,
  )
  if use_scan:
    print(
        "[scan] lax.scan over stacked decoder layer (1 compiled body) ON",
        flush=True,
    )
  if use_remat:
    _policy_name = (
        "save_and_offload(proj+mlpwi → pinned_host)"
        if offload_remat
        else "nothing_saveable"
    )
    print(f"[remat] per-layer jax.checkpoint ({_policy_name}) ON", flush=True)
  if use_splash:
    os.environ["JAX_ATTENTION_IMPL"] = "splash"
    from model import set_splash_mesh

    set_splash_mesh(mesh)
    print("[attn] splash kernel ON (JAX_ATTENTION_IMPL=splash)", flush=True)
  if use_maxtext_ce:
    print(
        "[ce] MaxText/T5X custom_vjp cross_entropy_with_logits (z_loss=0) ON",
        flush=True,
    )
  if shard_acts:
    from model import set_shard_acts, set_splash_mesh

    set_splash_mesh(mesh)  # _sac reuses the splash mesh global; ensure it's set
    set_shard_acts(True)
    print(
        "[shard] activation sharding constraints ON (P(fsdp, ...) at layer"
        " boundaries)",
        flush=True,
    )
  n_params = sum(int(p.value.size) for _, p in _iter_params(model))
  print(
      f"[load] Qwen3 has {n_params/1e9:.2f} B parameters (NNX-side), random"
      " init",
      flush=True,
  )

  # Sharding plan → place each param on its NamedSharding.
  plan = build_plan(model, mesh)
  for note in plan.notes:
    print(f"[sharding] {note}", flush=True)
  print(
      f"[sharding] matched={len(plan.buckets['matched'])} "
      f"replicated={len(plan.buckets['replicated'])}",
      flush=True,
  )
  apply_sharding(model, plan)

  # Optimizer.
  optimizer = optax.adamw(
      learning_rate=learning_rate, weight_decay=weight_decay
  )
  print(f"[opt] adamw lr={learning_rate} wd={weight_decay}", flush=True)

  # Split Params only — the RoPE `inv_freq` (nnx.data) stays in `rest` so it is
  # NOT differentiated/updated (it's a constant). graphdef + rest reconstruct
  # the model inside the loss.
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)
  opt_state = optimizer.init(params)

  # Replicate any opt_state scalar leaves that didn't inherit a per-param sharding.
  repl = NamedSharding(mesh, P())

  def _fix_leaf(leaf):
    if isinstance(leaf, jax.Array) and len(leaf.sharding.device_set) < n_global:
      return jax.device_put(leaf, repl)
    return leaf

  opt_state = jax.tree.map(_fix_leaf, opt_state)

  # Data.
  global_batch = batch_size * fsdp
  if use_real_data:
    from transformers import AutoTokenizer
    from data import make_dataloader

    tok = AutoTokenizer.from_pretrained(model_id)
    data_iter = make_dataloader(
        seq_len=seqlen, batch_size=global_batch, tokenizer=tok
    )
    print(
        f"[data] wikitext-2-raw-v1 global_batch={global_batch} seqlen={seqlen}",
        flush=True,
    )
  else:
    from data import fake_dataloader

    data_iter = fake_dataloader(
        train_steps + 5, seqlen, global_batch, vocab_size=config.vocab_size
    )
    print(
        f"[data] fake (random ints) global_batch={global_batch}"
        f" seqlen={seqlen}",
        flush=True,
    )

  # Loss + train step.
  def _ce(logits, labels):
    v = logits.shape[-1]
    logp = jax.nn.log_softmax(
        logits.reshape(-1, v).astype(jnp.float32), axis=-1
    )
    picked = jnp.take_along_axis(
        logp, labels.reshape(-1)[:, None], axis=-1
    ).squeeze(-1)
    return -picked.mean()

  def _ce_tokamax(hidden, labels, lm_head_w):
    # Streamed cross-entropy over V via tokamax (Pallas) under a shard_map —
    # never materializes [B,L,V] logits. Mirrors the llama3 jax sibling.
    # fp32 boundary cast is REQUIRED for chunked_xla (else lse accumulates in
    # bf16 and the loss collapses to bf16 quantization).
    import tokamax
    from jax.experimental.shard_map import shard_map as _shard_map

    B, L, H = hidden.shape
    BL = B * L
    l_flat = labels.reshape(BL)
    # f32 hidden + weight: REQUIRED — the tokamax mosaic_tpu CE backward kernel
    # allocates an f32 scratch and rejects a bf16 weight (v029 trace crash:
    # "Invalid dtype for swap: Ref=f32, Value=bf16"). Also needed for chunked_xla
    # lse accumulation. The f32[H,V] weight is the bs3 HBM wall — addressed by
    # scan's smaller program footprint, not by weight dtype.
    h_ce = hidden.reshape(BL, H).astype(jnp.float32)
    w_ce = lm_head_w.T.astype(jnp.float32)  # (V,H) -> (H,V)

    def _ce_local(h, l, w):
      s = tokamax.linear_softmax_cross_entropy_loss(
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
    return ce_sm(h_ce, l_flat, w_ce) / float(BL)

  def _ce_maxtext(logits, labels):
    v = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, v, dtype=jnp.float32)
    xent, _ = cross_entropy_with_logits(
        logits.astype(jnp.float32), one_hot, 0.0
    )
    return xent.mean()

  def loss_fn(params, input_ids, labels):
    m = nnx.merge(graphdef, params, rest)
    if use_tokamax_ce:
      hidden = m(input_ids, return_hidden=True)
      return _ce_tokamax(hidden, labels, m.lm_head_weight()).astype(jnp.float32)
    if use_maxtext_ce:
      return _ce_maxtext(m(input_ids), labels).astype(jnp.float32)
    return _ce(m(input_ids), labels).astype(jnp.float32)

  grad_fn = jax.value_and_grad(loss_fn)

  def train_step(params, opt_state, input_ids, labels):
    with jax.named_scope("forward_backward"):
      loss, grads = grad_fn(params, input_ids, labels)
    with jax.named_scope("optimizer"):
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
    return loss, params, opt_state

  jitted_step = jax.jit(train_step, donate_argnums=(0, 1))
  in_shard = input_sharding(mesh)

  from profiling import TraceController

  prof = TraceController(
      local_dir=profile_dir,
      gcs_dir=profile_gcs_dir,
      start_step=profile_start_step,
      num_steps=profile_steps,
      xprof_url_base=xprof_url_base,
  )

  print(
      f"[train] starting train_steps={train_steps} per_chip_batch={batch_size} "
      f"global_batch={global_batch}",
      flush=True,
  )
  warmup_steps = 2
  total_tokens = 0
  total_time = 0.0
  n_measured = 0

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

    prof.maybe_start(i)
    t0 = time.perf_counter()
    loss, params, opt_state = jitted_step(params, opt_state, input_ids, labels)
    jax.block_until_ready(loss)
    dt = time.perf_counter() - t0
    prof.maybe_stop(i)

    tps = tokens_this_step / dt
    print(
        f"[step {i:2d}/{train_steps}] loss={float(loss):.4f} "
        f"step_time={dt*1000:.1f}ms throughput={tps:.0f} tok/s",
        flush=True,
    )

    if i >= warmup_steps:
      total_tokens += tokens_this_step
      total_time += dt
      n_measured += 1

  prof.finalize()

  if total_time > 0 and n_measured > 0:
    avg_tps = total_tokens / total_time
    per_chip = avg_tps / n_global
    avg_step_time = total_time / n_measured
    peak = 918e12  # v6e bf16 peak ≈ 918 TFLOPS / chip.
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
    causal_attn_flops = 4 * B * L * L * Hq * hd / 2
    learnable_tflops = (
        (ffn_flops * nL + (qkv_flops + proj_flops) * nL + embed_flops)
        * 3
        / 1e12
    )
    attention_tflops = causal_attn_flops * nL * 3 / 1e12
    total_tflops = learnable_tflops + attention_tflops
    mfu = (total_tflops * 1e12) / (avg_step_time * peak)
    print("\n================ summary ================", flush=True)
    print(f"global_batch          : {global_batch}", flush=True)
    print(f"seqlen                : {seqlen}", flush=True)
    print(f"steps measured        : {n_measured}", flush=True)
    print(
        f"avg throughput        : {avg_tps:.0f} tok/s ({per_chip:.0f}/chip)",
        flush=True,
    )
    print(f"approx MFU            : {mfu*100:.1f}% (v6e bf16 peak)", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
  fire.Fire(main)
