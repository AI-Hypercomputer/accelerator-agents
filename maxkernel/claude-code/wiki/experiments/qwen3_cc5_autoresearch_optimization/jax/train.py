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
        --profile_dir gs://<bucket>/autoresearch/qwen3_cc5/<run> \
        --profile_start_step 12 --profile_steps 3
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Optional

import fire
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import optax

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
  sys.path.insert(0, str(_THIS_DIR))

# Stub `jaxtyping.jaxtyped` to avoid typeguard AST-walk crashes on tokamax's
# `*B T H d` annotations. Must run before any tokamax import.
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
  from absl import flags as _absl_flags

  if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS([sys.argv[0]])
except Exception:
  pass


def _to_jnp_dtype(name: str) -> jnp.dtype:
  return jnp.bfloat16 if name == "bf16" else jnp.float32


def main(
    model_id: str = "Qwen/Qwen3-8B",
    batch_size: int = 1,  # per-fsdp-shard batch; global = batch × fsdp
    seqlen: int = 2048,
    train_steps: int = 20,
    tp_parallelism: int = 1,  # 1 = pure FSDP across all chips
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    weights_dtype: str = "bf16",
    use_real_data: bool = False,  # False = synthetic tokens (perf baseline)
    use_splash: bool = False,  # splash attention kernel (TPU only)
    use_tokamax_ce: bool = False,  # tokamax streamed CE, bf16 inputs (no [BL,V] logits in HBM)
    tokamax_ce_impl: str = "mosaic_tpu",
    remat_policy: str = "",  # "" | "save_attn" | "selective" | "full"
    use_scan: bool = False,  # scan-over-layers (stacked params, O(1) compile)
    # --- profiling (all CLI flags) ---
    profile_dir: Optional[str] = None,
    profile_gcs_dir: Optional[str] = None,
    profile_start_step: int = 8,
    profile_steps: int = 3,
    xprof_url_base: str = "http://localhost:8791",
):
  if remat_policy:
    os.environ["JAX_REMAT_POLICY"] = remat_policy
    print(f"[remat] per-layer jax.checkpoint policy={remat_policy}", flush=True)
  if use_scan:
    os.environ["JAX_USE_SCAN"] = "1"
    print("[scan] scan-over-layers enabled (stacked params)", flush=True)

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

  if use_splash:
    os.environ["JAX_ATTENTION_IMPL"] = "splash"
    from model.modeling_qwen3 import set_splash_mesh

    set_splash_mesh(mesh)
    print(
        "[attn] splash attention enabled (env JAX_ATTENTION_IMPL=splash)",
        flush=True,
    )

  model = Qwen3ForCausalLM(
      config,
      weights_dtype=wdtype,
      compute_dtype=wdtype,
      rngs=nnx.Rngs(0),
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

  tokamax_mod = None
  if use_tokamax_ce:
    import tokamax as tokamax_mod
    # skip_lm_head is a plain attr — it lands in graphdef, so it must be
    # set BEFORE the nnx.split below.
    model.skip_lm_head = True
    print(
        f"[ce] tokamax.linear_softmax_cross_entropy_loss impl={tokamax_ce_impl}"
        " (bf16 inputs)",
        flush=True,
    )
  else:
    print("[ce] plain softmax_cross_entropy (logits materialized)", flush=True)

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
    """Tokamax fused CE, f32-x-only boundary.

    chunked_xla accumulates its lse in `x.dtype` (tokamax chunked_xla.py: `dtype
    = x.dtype`), so x MUST be f32 (32 MB at bs1) or the loss carries a ~ln2
    bf16-lse bias (v008b). w stays bf16 — casting it costs 2.49 GB and is never
    needed (v003). psum('fsdp') inside shard_map is the canonical
    Mosaic-under-GSPMD pattern.
    """
    B, L, H = hidden.shape
    BL = B * L
    h_flat = hidden.reshape(BL, H).astype(jnp.float32)
    l_flat = labels.reshape(BL)
    w_HV = lm_head_w.T  # (V, H) -> (H, V), stays bf16

    from jax.experimental.shard_map import shard_map

    def _ce_local(h, l, w):
      s = tokamax_mod.linear_softmax_cross_entropy_loss(
          h, l, w, reduction="sum", implementation=tokamax_ce_impl
      )
      return jax.lax.psum(s, axis_name="fsdp")

    ce_sm = shard_map(
        _ce_local,
        mesh=mesh,
        in_specs=(P("fsdp", None), P("fsdp"), P()),
        out_specs=P(),
        check_rep=False,
    )
    return ce_sm(h_flat, l_flat, w_HV) / float(BL)

  def loss_fn(params, input_ids, labels):
    m = nnx.merge(graphdef, params, rest)
    if use_tokamax_ce:
      hidden = m(input_ids, return_hidden=True)
      return _ce_tokamax(hidden, labels, m.lm_head_weight()).astype(jnp.float32)
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
