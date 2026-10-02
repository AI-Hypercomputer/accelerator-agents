"""Minimal Qwen3 training on TPU via torchax.

A deliberately small baseline trainer — the canonical torchax + torchtitan
pattern adapted to HF `transformers.Qwen3ForCausalLM` and real WikiText data.
It establishes the *baseline* number for the qwen3_cc5 autoresearch lane; the
optimization knobs (splash attention, scan-over-layers, tokamax CE, AMP master
weights, per-layer remat) are intentionally NOT here yet — each one lands later
as its own experiment so the win is attributable.

Canonical patterns retained (proven on multi-host TPU GKE):
- Meta-device model init + per-weight `make_array_from_callback` (never
  materializes a full weight on host).
- `axis_types=(Auto, Auto)` mesh — JAX infers output shardings.
- `torchax.train.make_train_step` for the train step (remat=nothing_saveable).
- `helper.compile_step_func` for explicit precompile + cost analysis.
- `sharded_device_put` handles single-host and multi-host correctly.

Compile cache: set `JAX_COMPILATION_CACHE_DIR` to a persistent path to amortize
cold-compile cost across runs.

Run (from this folder):
    python -m train --steps 20 --batch_size 1 --seqlen 8192 \
        --profile_dir $PROFILE_DIR --profile_step 10
"""

from __future__ import annotations

import functools
import time

import fire
import helper  # local sibling
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from model import AutoTokenizer, Qwen3ForCausalLM
from model.sharding import SHARDING_MAP, _process_sharding_name
import optax
import torch
import torch.nn.functional
import torchax
import torchax.interop
from torchax.interop import JittableModule, jax_view, torch_view
import torchax.train

# -----------------------------------------------------------------------------
# Multi/single-host device_put helper (canonical pattern).
# -----------------------------------------------------------------------------


def sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:
  """Place a host-local jax.Array on a multi-device mesh, correctly handling

  single-host AND multi-host.
  """
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)
  if jax.device_count() == jax.local_device_count():
    return jax.device_put(tensor, sharding)
  shape = tensor.shape
  x_split = [
      jax.device_put(tensor[i], device)
      for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


# -----------------------------------------------------------------------------
# Per-shard weight init via `jax.make_array_from_callback`.
# -----------------------------------------------------------------------------


def _make_weight_shard(weight_meta, slice_index):
  shard_meta = weight_meta[slice_index]
  seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (
      2**31 - 1
  )
  key = jax.random.PRNGKey(seed)
  dtype_map = {
      torch.bfloat16: jnp.bfloat16,
      torch.float16: jnp.float16,
      torch.float32: jnp.float32,
  }
  jax_dtype = dtype_map.get(shard_meta.dtype, jnp.bfloat16)
  return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype) * 0.02


def _materialize_buffers_replicated(model: torch.nn.Module, mesh) -> None:
  """Replace any leftover meta-device buffers with fully-replicated torchax

  tensors. For HF Qwen3, `inv_freq` (rotary embedding) is a non-persistent
  buffer — meta-init leaves it unmaterialized and `interop.jax_view` later
  crashes on it.
  """
  env = torchax.default_env()
  replicated = NamedSharding(mesh, P())

  def _walk(module, prefix=""):
    for name, buf in list(module.named_buffers(recurse=False)):
      full = f"{prefix}{name}"
      if buf.is_meta:
        # HF Qwen3 keeps both `inv_freq` (live) and `original_inv_freq`
        # (kept for dynamic RoPE rescaling). For the default (no scaling)
        # they're identical — regenerate both from rope_theta. Zeroing
        # `original_inv_freq` silently breaks any dynamic-scaling path.
        if name in ("inv_freq", "original_inv_freq") and hasattr(
            module, "config"
        ):
          half = buf.shape[0]
          theta = getattr(module.config, "rope_theta", 1000000.0)
          inv_freq = 1.0 / (
              theta ** (torch.arange(0, half, dtype=torch.float32) / half)
          )
          buf = inv_freq.to(
              buf.dtype if buf.dtype != torch.float32 else torch.float32
          )
        else:
          print(
              f"[buffer] unknown meta buffer {full} {tuple(buf.shape)} "
              f"{buf.dtype} — initializing zeros",
              flush=True,
          )
          buf = torch.zeros(buf.shape, dtype=buf.dtype)
      jax_arr = jax.device_put(jnp.asarray(buf.numpy()), replicated)
      setattr(module, name, env.j2t_iso(jax_arr))

    for child_name, child in module.named_children():
      _walk(child, prefix=f"{prefix}{child_name}.")

  _walk(model)


def create_sharded_weights(model, mesh, sharding_map=None):
  """Walk model.state_dict() and instantiate each weight directly on its shard

  via `jax.make_array_from_callback`. Skips entries with no sharding spec.
  """
  if sharding_map is None:
    sharding_map = SHARDING_MAP
  res = {}
  env = torchax.default_env()
  skipped = []
  for name, weight_meta in model.state_dict().items():
    spec = sharding_map.get(_process_sharding_name(name))
    if spec is None:
      skipped.append(name)
      continue
    sharding = NamedSharding(mesh, P(*spec))
    print(
        f"[shard] {name} {tuple(weight_meta.shape)} {weight_meta.dtype} ->"
        f" {spec}",
        flush=True,
    )
    res[name] = env.j2t_iso(
        jax.make_array_from_callback(
            weight_meta.shape,
            sharding,
            functools.partial(_make_weight_shard, weight_meta),
        )
    )
  if skipped:
    print(
        f"[shard] skipped {len(skipped)} entries with no sharding spec "
        f"(buffers, etc.): {skipped[:5]}{'...' if len(skipped) > 5 else ''}",
        flush=True,
    )
  return res


def _load_config(model_id: str):
  """Load `Qwen3Config` for HF model_id without downloading weights."""
  from model import Qwen3Config

  return Qwen3Config.from_pretrained(model_id)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main(
    model_id: str = "Qwen/Qwen3-8B",
    batch_size: int = 1,  # per-fsdp-shard batch; global = batch_size * fsdp
    seqlen: int = 8192,
    train_steps: int = 20,
    tp_parallelism: int = 1,  # 1 => pure FSDP across all chips
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    weights_dtype: str = "bf16",  # bf16 | fp32
    use_real_data: bool = True,  # True=wikitext, False=random tokens (perf smoke)
    # --- profiling (all CLI flags) -------------------------------------------
    profile_dir: (
        str | None
    ) = None,  # where jax.profiler writes; local OR gs:// path.
    profile_gcs_dir: (
        str | None
    ) = None,  # gs://<bucket>/autoresearch/<run-name> to publish
    # a LOCAL capture to (rsync after run) for xprof-mcp.
    profile_start_step: int = 8,  # first step to capture.
    profile_steps: int = 3,  # number of consecutive steps to capture.
    xprof_url_base: str = "http://localhost:8791",  # for the printed click-through URL.
):
  torchax.enable_globally()
  torchax.enable_performance_mode()

  n_global = jax.device_count()
  n_local = jax.local_device_count()
  n_hosts = jax.process_count()
  print(
      f"[dist] global_devices={n_global} local_devices={n_local}"
      f" hosts={n_hosts}",
      flush=True,
  )

  # Mesh — Auto axis types so JAX infers output shardings.
  fsdp = n_global // tp_parallelism
  AxisType = jax.sharding.AxisType
  mesh = jax.make_mesh(
      (fsdp, tp_parallelism),
      ("fsdp", "tp"),
      axis_types=(AxisType.Auto, AxisType.Auto),
  )
  print(f"[mesh] fsdp={fsdp} tp={tp_parallelism} mesh={mesh}", flush=True)

  # Build the model on `meta` so weights aren't allocated on CPU.
  torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[weights_dtype]
  torch.set_default_dtype(torch_dtype)
  print(
      f"[load] model_id={model_id} weights_dtype={weights_dtype} (meta init)",
      flush=True,
  )
  with torch.device("meta"):
    model = Qwen3ForCausalLM(_load_config(model_id))
  n_params = sum(p.numel() for p in model.parameters())
  print(f"[load] model has {n_params/1e9:.2f} B parameters", flush=True)

  # Sharded weight init, then materialize non-sharded buffers (RoPE inv_freq).
  state_dict = create_sharded_weights(model, mesh)
  model.load_state_dict(state_dict, assign=True, strict=False)
  _materialize_buffers_replicated(model, mesh)

  # Tokenizer + data loader.
  tokenizer = AutoTokenizer.from_pretrained(model_id) if use_real_data else None
  global_batch = batch_size * fsdp
  if use_real_data:
    from data import make_dataloader

    loader = make_dataloader(
        seq_len=seqlen, batch_size=global_batch, tokenizer=tokenizer
    )
    print(
        f"[data] wikitext-2-raw-v1, global_batch={global_batch},"
        f" seqlen={seqlen}",
        flush=True,
    )
  else:
    from data import fake_dataloader

    loader = fake_dataloader(
        train_steps + 5,
        seqlen,
        global_batch,
        vocab_size=model.config.vocab_size,
    )
    print(
        f"[data] fake (random ints), global_batch={global_batch},"
        f" seqlen={seqlen}",
        flush=True,
    )

  # Trainer wrapping.
  jittable_mod = JittableModule(model)

  def model_fn(weights, buffers, args):
    out = jittable_mod.functional_call("forward", weights, buffers, args)
    if isinstance(out, torch.Tensor):
      return out
    return out.logits if hasattr(out, "logits") else out[0]

  def loss_fn(logits, labels):
    v = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, v), labels.reshape(-1)
    )

  optimizer = optax.adamw(
      learning_rate=learning_rate, weight_decay=weight_decay
  )
  print(f"[opt] adamw lr={learning_rate} wd={weight_decay}", flush=True)

  # optimizer.init produces mu/nu (inherit param sharding) plus small scalars
  # that default to one device. Walk the pytree and replicate any leaf that
  # doesn't already span the mesh, so compile_step_func's out_shardings match.
  _replicated = NamedSharding(mesh, P())
  _opt_state_raw = optimizer.init(jax_view(jittable_mod.params))

  def _fix_leaf_sharding(leaf):
    if isinstance(leaf, jax.Array) and len(leaf.sharding.device_set) < n_global:
      return jax.device_put(leaf, _replicated)
    return leaf

  opt_state = torch_view(jax.tree.map(_fix_leaf_sharding, _opt_state_raw))

  train_step = torchax.train.make_train_step(
      model_fn,
      loss_fn,
      optimizer,
      remat_policy=jax.checkpoint_policies.nothing_saveable,
  )

  x_sharding = NamedSharding(mesh, P("fsdp"))

  # Profiler — captures steps [profile_start_step, +profile_steps) to
  # `profile_dir` (local or gs://); optionally rsyncs a local capture to
  # `profile_gcs_dir` for xprof-mcp. See profiling.py.
  from profiling import TraceController

  prof = TraceController(
      local_dir=profile_dir,
      gcs_dir=profile_gcs_dir,
      start_step=profile_start_step,
      num_steps=profile_steps,
      xprof_url_base=xprof_url_base,
  )

  print("[train] starting...", flush=True)
  warmup_steps = 2
  total_tokens_after_warmup = 0
  total_time_after_warmup = 0.0
  n_measured_steps = 0

  with mesh:
    for i, (inputs, labels) in enumerate(loader):
      if i >= train_steps:
        break
      tokens_this_step = inputs.shape[0] * inputs.shape[1]

      inputs = inputs.to("jax")
      labels = labels.to("jax")
      inputs.apply_jax_(sharded_device_put, x_sharding)
      labels.apply_jax_(sharded_device_put, x_sharding)

      if i == 0:
        train_step = helper.compile_step_func(
            train_step,
            jittable_mod.params,
            jittable_mod.buffers,
            opt_state,
            inputs,
            labels,
            mesh,
        )

      prof.maybe_start(i)

      t0 = time.perf_counter()
      loss, jittable_mod.params, opt_state = train_step(
          jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )
      torchax.interop.call_jax(
          jax.block_until_ready, (loss, jittable_mod.params)
      )
      dt = time.perf_counter() - t0

      prof.maybe_stop(i)

      tps = tokens_this_step / dt
      print(
          f"[step {i:2d}/{train_steps}] loss={loss.item():.4f} "
          f"step_time={dt*1000:.1f}ms throughput={tps:.0f} tok/s",
          flush=True,
      )

      if i >= warmup_steps:
        total_tokens_after_warmup += tokens_this_step
        total_time_after_warmup += dt
        n_measured_steps += 1

  prof.finalize()

  if total_time_after_warmup > 0:
    avg_tps = total_tokens_after_warmup / total_time_after_warmup
    per_chip = avg_tps / n_global
    avg_step_time = total_time_after_warmup / n_measured_steps
    peak = 918e12  # v6e bf16 peak ≈ 918 TFLOPS / chip.
    # MaxText-style train-step TFLOPs (forward + 2× backward = ×3) for a
    # dense GQA + SwiGLU model (Qwen3 is dense; QK-norm flops negligible).
    cfg = model.config
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
    ffn_flops = 2 * B * L * Mlp * D * (2 + 1)  # SwiGLU: gate+up in, down out.
    embed_flops = 2 * B * L * D * V  # lm_head only.
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
