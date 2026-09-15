# pylint: skip-file
"""Llama 3 8B training on TPU via torchax.

Built on the canonical torchax + torchtitan example pattern from
`raw/code/torchax/examples/train_llama_torchtitan/`, adapted to use HF
`transformers.LlamaForCausalLM` (the model on `meta-llama/Meta-Llama-3-8B`)
and real WikiText data.

Key canonical patterns retained (proven to work on multi-host TPU GKE):
- Meta-device model init + per-weight `make_array_from_callback` (never
  materializes a full weight on host — see `model.create_sharded_weights`).
- `axis_types=(Auto, Auto)` mesh — JAX infers output shardings.
- `torchax.train.make_train_step` for the train step (interop wraps optax).
- `helper.compile_step_func` for explicit precompile + cost analysis.
- `sharded_device_put` handles single-host (`jax.device_put`) and multi-host
  (`make_array_from_single_device_arrays` per local device) correctly.
- No explicit `jax.distributed.initialize()` — JAX auto-detects on TPU GKE.

Compile cache: set `JAX_COMPILATION_CACHE_DIR` (env var) to a persistent path
to amortize cold-compile cost across runs. Cold compile of llama3-8b is
~3 min on v6e-4; cache-hit drops to ~10 s.
"""

from __future__ import annotations

import functools
import os
import time

# Stub `jaxtyping.jaxtyped` to avoid typeguard AST-walk crash on tokamax's
# `*B T H d` annotations under py3.13. Must run before any tokamax import.
try:
  import jaxtyping as _jt

  _jt.jaxtyped = lambda typechecker=None: (lambda fn: fn)
except ImportError:
  pass

# tokamax/_src/config.py reads sys.argv via `flags.FLAGS(sys.argv)` on first
# config-option access, but this trainer uses `fire.Fire` (so sys.argv carries
# `--model_id=...` etc., which absl flags doesn't recognize). Pre-parse absl
# with only argv[0] so tokamax's `is_parsed()` short-circuits.
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
import torch
import torch.nn.functional
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import torchax
import torchax.interop
import torchax.train
from torchax.interop import JittableModule, jax_view, torch_view

import helper  # local sibling
from model import LlamaForCausalLM, AutoTokenizer
from model.sharding import SHARDING_MAP, _process_sharding_name

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
  """Replace any leftover meta-device or raw-torch buffers in `model` with

  fully-replicated torchax tensors. For HF Llama, `inv_freq` (in each layer's
  rotary embedding) is a non-persistent buffer — meta-init leaves it
  unmaterialized and `interop.jax_view` later crashes on it.
  """
  env = torchax.default_env()
  replicated = NamedSharding(mesh, P())

  def _walk(module, prefix=""):
    for name, buf in list(module.named_buffers(recurse=False)):
      full = f"{prefix}{name}"
      if buf.is_meta:
        # Recompute on CPU using the same constructor logic — for
        # `inv_freq` we just regenerate from base; for unknown buffers
        # we initialize zeros (warn).
        if name == "inv_freq" and hasattr(module, "config"):
          # HF rotary inv_freq: 1.0 / (rope_theta ** (i/half_dim))
          half = buf.shape[0]
          theta = getattr(module.config, "rope_theta", 10000.0)
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

      # buf is now a real CPU torch.Tensor; convert to replicated jax.
      jax_arr = jax.device_put(jnp.asarray(buf.numpy()), replicated)
      setattr(module, name, env.j2t_iso(jax_arr))

    for child_name, child in module.named_children():
      _walk(child, prefix=f"{prefix}{child_name}.")

  _walk(model)


def create_sharded_weights(model, mesh, sharding_map=None):
  """Walk model.state_dict() and instantiate each weight directly on its

  shard via `jax.make_array_from_callback`. Skips entries with no sharding
  spec — caller is responsible for any unmatched buffers.
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


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main(
    model_id: str = "meta-llama/Meta-Llama-3-8B",
    batch_size: int = 4,
    seqlen: int = 1024,
    train_steps: int = 15,
    tp_parallelism: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    weights_dtype: str = "bf16",  # bf16|fp32. Storage dtype of model params and
    # optimizer mu/nu (per `master_dtype`). For true
    # AMP master, set weights_dtype=fp32 and
    # compute_dtype=bf16 — weights are stored fp32
    # but cast to bf16 inside `model_fn` for compute.
    compute_dtype: str = "match",  # bf16|fp32|match. "match" = compute uses
    # weights_dtype (no autocast). "bf16" with
    # weights_dtype=fp32 enables true mixed-precision:
    # fp32 master weights, bf16 forward/backward.
    use_real_data: bool = True,  # True=wikitext, False=random tokens (perf smoke only)
    use_splash: bool = False,  # True = override torch.nn.functional.scaled_dot_product_attention
    # with the canonical TPU splash-attention kernel.
    use_scan: bool = False,  # True = use model.scan.LlamaForCausalLMScan which
    # stacks the 32 LlamaDecoderLayers into a single
    # scan body via torchax.train.ScannedModule. XLA
    # compile-time HBM analysis sees one body's worth
    # of buffers instead of the 32-unrolled sum, often
    # freeing several GiB on high-density shapes.
    use_per_layer_remat: bool = False,  # True = wrap each LlamaDecoderLayer.forward in
    # jax.checkpoint(policy=nothing_saveable). 32 distinct
    # checkpoint scopes force XLA to schedule layer-by-layer
    # — should reduce compile-time HBM peak vs the outer
    # gradient_checkpoint exp 11 used (which collapses to
    # one giant scope and XLA schedules everything live).
    master_dtype: str = "match",  # "match" = mu/nu inherit weights_dtype (default).
    # "fp32" = force mu/nu to fp32 (AMP master state)
    # while weights stay in `weights_dtype`. The
    # standard mixed-precision pattern: fp32 master
    # for the optimizer step, smaller dtype for the
    # forward/backward.
    scan_remat_policy: str = "nothing_saveable",  # nothing_saveable | dots_saveable
    # | dots_with_no_batch_dims_saveable | save_anything_except_these_names
    # Policy for the scan body's gradient checkpoint.
    # Default `nothing_saveable` recomputes everything on bwd
    # (max memory savings); `dots_saveable` saves matmul
    # outputs (less bwd compute, some memory cost).
    use_tokamax_ce: bool = False,  # True = use tokamax.linear_softmax_cross_entropy_loss
    # which streams logsumexp over V via Pallas
    # (mosaic_tpu impl). Skip lm_head materialization
    # → ~256 MiB/chip activation savings at seq=8192.
    tokamax_ce_impl: str = "mosaic_tpu",  # mosaic_tpu | chunked_xla | xla. The
    # `chunked_xla` impl chunks (B*L, V) into pieces
    # using XLA matmuls (no Pallas); `mosaic_tpu` is the
    # streamed-V Pallas kernel (current frontier);
    # `xla` materializes full logits (OOMs at this shape).
    tokamax_ce_autotune: bool = False,  # True = autotune CE kernel block sizes
    # (cache-miss fallback = "autotune"). Runs all
    # configs in _get_autotuning_configs and picks
    # fastest. Adds compile time on first call.
    profile_dir: str | None = None,
    profile_step: int = 5,
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

  # Mesh — Auto axis types so JAX infers output shardings (newer JAX defaults
  # to Explicit, which breaks gather ops in HF Llama's forward).
  fsdp = n_global // tp_parallelism
  AxisType = jax.sharding.AxisType
  mesh = jax.make_mesh(
      (fsdp, tp_parallelism),
      ("fsdp", "tp"),
      axis_types=(AxisType.Auto, AxisType.Auto),
  )
  print(f"[mesh] fsdp={fsdp} tp={tp_parallelism} mesh={mesh}", flush=True)

  # Build the model on `meta` so weights aren't allocated on CPU. Each weight
  # gets per-shard random init in `create_sharded_weights`.
  torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[weights_dtype]
  torch.set_default_dtype(torch_dtype)
  print(
      f"[load] model_id={model_id} weights_dtype={weights_dtype} (meta init)",
      flush=True,
  )
  with torch.device("meta"):
    if use_scan:
      from model.scan import LlamaForCausalLMScan

      _ck_policy = getattr(jax.checkpoint_policies, scan_remat_policy)
      model = LlamaForCausalLMScan(
          _load_config(model_id), checkpoint_policy=_ck_policy
      )
      print(
          "[scan] LlamaForCausalLMScan installed (32 layers stacked into "
          f"1 scan body), checkpoint_policy={scan_remat_policy}",
          flush=True,
      )
    else:
      model = (
          LlamaForCausalLM.from_pretrained(
              model_id,
              torch_dtype=torch_dtype,
              low_cpu_mem_usage=True,
          )
          if False
          else LlamaForCausalLM(_load_config(model_id))
      )
  n_params = sum(p.numel() for p in model.parameters())
  print(f"[load] model has {n_params/1e9:.2f} B parameters", flush=True)

  # Sharded weight init.
  if use_scan:
    from model.sharding import SCAN_SHARDING_MAP

    state_dict = create_sharded_weights(
        model, mesh, sharding_map=SCAN_SHARDING_MAP
    )
  else:
    state_dict = create_sharded_weights(model, mesh)
  model.load_state_dict(state_dict, assign=True, strict=False)

  # Materialize any non-sharded buffers (HF Llama RoPE inv_freq, etc.) as
  # replicated torchax tensors. Without this, `interop.jax_view` crashes
  # on raw torch.Tensors when wrapping the train_step.
  _materialize_buffers_replicated(model, mesh)

  # Splash attention override (canonical pattern from torchtitan example).
  # HF Llama calls torch.nn.functional.scaled_dot_product_attention from its
  # SdpaAttention path — overriding the op redirects every layer to the
  # TPU splash kernel. GQA handled natively by `make_splash_mha`
  # (num_kv_heads inferred from input shape).
  if use_splash:
    import torch.nn.functional as F
    import splash_attn

    attn_partition = P("fsdp", "tp", None, None)
    _splash = jax.jit(
        functools.partial(
            splash_attn.tpu_splash_attention, mesh, attn_partition, True
        )
    )

    def _custom_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
      # query/key/value: (batch, num_heads, seq, head_dim) torchax tensors.
      jq, jk, jv = jax_view((query, key, value))
      res = _splash(jq, jk, jv, None)
      return torch_view(res)

    torchax.default_env().override_op_definition(
        F.scaled_dot_product_attention, _custom_attention
    )
    print(
        "[attn] splash kernel installed (GQA-native, mesh-shard"
        " P('fsdp','tp',_,_))",
        flush=True,
    )

  # Per-layer remat — patches LlamaModel.forward to wrap each
  # decoder_layer call in `interop.gradient_checkpoint`. Each layer becomes
  # its own checkpoint scope so XLA schedules them serially. The full
  # outer-loss `gradient_checkpoint` (exp 11) collapsed everything to one
  # scope and didn't reduce compile-time HBM peak.
  if use_per_layer_remat and not use_scan:
    import types
    from transformers.modeling_outputs import BaseModelOutputWithPast

    try:
      from transformers.masking_utils import create_causal_mask
    except ImportError:
      from transformers.models.llama.modeling_llama import create_causal_mask
    _remat_policy = jax.checkpoint_policies.nothing_saveable

    def _patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
      if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
      if position_ids is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)
      causal_mask = create_causal_mask(
          config=self.config,
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
          past_key_values=past_key_values,
          position_ids=position_ids,
      )
      hidden_states = inputs_embeds
      position_embeddings = self.rotary_emb(
          hidden_states, position_ids=position_ids
      )

      def _make_layer_call(layer):
        def call(h, mask, cos, sin):
          return layer(h, attention_mask=mask, position_embeddings=(cos, sin))

        return torchax.interop.gradient_checkpoint(
            call, kwargs={"policy": _remat_policy}
        )

      cos, sin = position_embeddings
      for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = _make_layer_call(decoder_layer)(
            hidden_states, causal_mask, cos, sin
        )
      hidden_states = self.norm(hidden_states)
      return BaseModelOutputWithPast(
          last_hidden_state=hidden_states, past_key_values=past_key_values
      )

    model.model.forward = types.MethodType(_patched_model_forward, model.model)
    print(
        "[remat] per-layer gradient_checkpoint installed "
        f"(policy=nothing_saveable, {len(model.model.layers)} scopes)",
        flush=True,
    )

  # Tokenizer (only needed if use_real_data).
  tokenizer = AutoTokenizer.from_pretrained(model_id) if use_real_data else None

  # Data loader.
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

    loader = fake_dataloader(train_steps + 5, seqlen, global_batch)
    print(
        f"[data] fake (random ints), global_batch={global_batch},"
        f" seqlen={seqlen}",
        flush=True,
    )

  # Trainer wrapping.
  env = torchax.default_env()
  jittable_mod = JittableModule(model)

  # AMP master-weight pattern: weights stored at `weights_dtype` (typically
  # fp32), forward/backward compute uses `compute_dtype` (typically bf16).
  # Cast each weight to compute_dtype inside model_fn — JAX's `astype` vjp
  # downcasts the bf16 grad back to fp32 on the way out, so the optimizer
  # sees fp32 grads matching its fp32 mu/nu and fp32 master weights.
  _torch_compute_dtype = None
  if compute_dtype != "match" and compute_dtype != weights_dtype:
    _torch_compute_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        compute_dtype
    ]
    print(
        f"[amp] true mixed-precision: storage={weights_dtype}"
        f" compute={compute_dtype} (weights cast to {compute_dtype} inside"
        " model_fn)",
        flush=True,
    )

  def _maybe_cast_weights(weights):
    if _torch_compute_dtype is None:
      return weights
    return {k: v.to(_torch_compute_dtype) for k, v in weights.items()}

  if use_tokamax_ce:
    # Skip lm_head — return hidden_states only. Loss path applies tokamax
    # `linear_softmax_cross_entropy_loss` (Pallas mosaic_tpu) over flat
    # B*L without materializing [B*L, V] logits.
    try:
      import tokamax as _tokamax
    except Exception as e:
      raise RuntimeError(
          "use_tokamax_ce=True requires tokamax importable; tried "
          "jaxtyping stub at module import. Fail: "
          + str(e)
      )
    # For LlamaForCausalLMScan we need `model.model_no_lm_head` semantics.
    # Easiest: monkey-patch lm_head to identity at use time.
    if not use_scan:
      raise RuntimeError("use_tokamax_ce currently requires use_scan=True")
    # Just toggle the model's `skip_lm_head` flag — `lm_head.weight`
    # remains the canonical key in jittable_mod.params.
    model.skip_lm_head = True  # type: ignore[attr-defined]
    print(
        "[ce] tokamax.linear_softmax_cross_entropy_loss enabled "
        "(skip_lm_head=True; lm_head.weight will be referenced from "
        "weights dict)",
        flush=True,
    )

    # Approach: compute the entire loss inside the function we wrap with
    # value_and_grad, so we can directly close over `weights` to read
    # lm_head.weight (the un-projected logit head). model_fn is a no-op
    # passthrough that returns hidden_states; the real CE happens here.
    def model_fn(weights, buffers, args):
      return jittable_mod.functional_call(
          "forward", _maybe_cast_weights(weights), buffers, args
      )  # (B, L, H)

    from jax.experimental.shard_map import shard_map as _shard_map

    def loss_fn(hidden, labels, weights):
      B, L, H = hidden.shape  # global B*L (logical shape pre-shard)
      BL = B * L
      h_flat = hidden.reshape(BL, H)
      l_flat = labels.reshape(BL)
      # canonical key: lm_head.weight (since model.lm_head is unchanged
      # but model.skip_lm_head=True bypasses the projection in forward).
      if "lm_head.weight" in weights:
        w_VH = weights["lm_head.weight"]
      else:
        raise KeyError(
            f"lm_head.weight not in weights. Available: {list(weights.keys())}"
        )
      w_HV = w_VH.transpose(0, 1)
      jh, jl, jw = torchax.interop.jax_view((h_flat, l_flat, w_HV))
      # Cast inputs to fp32 for BOTH impls. mosaic_tpu needs it for the
      # hardcoded fp32 grad output (the cast-vjp downcasts the bf16
      # grad on the way back). chunked_xla also needs it: its kernel
      # accumulates `lse` and `loss_sum` in `x.dtype`, so bf16 input
      # collapses the loss to bf16 quantization (~0.06 resolution at
      # magnitude ~11) — exp 66 series had loss=11.0000 plateaus. Cost
      # of the cast is ~negligible (one extra kernel; both inputs are
      # already in HBM at this point).
      jh32 = jh.astype(jnp.float32)
      jw32 = jw.astype(jnp.float32)

      # All Pallas + XLA tokamax CE impls go through the shard_map
      # wrapper. Even chunked_xla loses ~36 % per-chip without it
      # (exp 64 — JAX auto-partition picks a far worse pattern).
      if False:
        jloss = _tokamax.linear_softmax_cross_entropy_loss(
            jh32, jl, jw32, reduction="mean", implementation=tokamax_ce_impl
        )
      else:
        # mosaic_tpu / xla via shard_map (Pallas kernels can't auto-partition).
        def _ce_local(h, l, w):
          if tokamax_ce_autotune:
            with _tokamax.config.autotuning_cache_miss_fallback("autotune"):
              local_sum = _tokamax.linear_softmax_cross_entropy_loss(
                  h, l, w, reduction="sum", implementation=tokamax_ce_impl
              )
          else:
            local_sum = _tokamax.linear_softmax_cross_entropy_loss(
                h, l, w, reduction="sum", implementation=tokamax_ce_impl
            )
          return jax.lax.psum(local_sum, axis_name="fsdp")

        ce_sm = _shard_map(
            _ce_local,
            mesh=mesh,
            in_specs=(P("fsdp", None), P("fsdp"), P()),
            out_specs=P(),
            check_rep=False,
        )
        global_sum = ce_sm(jh32, jl, jw32)
        jloss = global_sum / float(BL)
      return torchax.interop.torch_view(jloss)

  else:

    def model_fn(weights, buffers, args):
      out = jittable_mod.functional_call(
          "forward", _maybe_cast_weights(weights), buffers, args
      )
      # HF returns CausalLMOutputWithPast; under torchax, attribute access works.
      # LlamaForCausalLMScan returns the logits tensor directly.
      if isinstance(out, torch.Tensor):
        return out
      return out.logits if hasattr(out, "logits") else out[0]

    def loss_fn(logits, labels):
      # Causal LM cross-entropy (mean over non-ignored tokens).
      v = logits.shape[-1]
      return torch.nn.functional.cross_entropy(
          logits.reshape(-1, v), labels.reshape(-1)
      )

  # AMP: optionally force mu/nu to fp32 even when weights are bf16 (the
  # standard mixed-precision pattern — fp32 master for the optimizer
  # update, smaller dtype for forward/backward). `nu` follows `mu_dtype`
  # in optax's scale_by_adam.
  _master_dtype_jax = jnp.float32 if master_dtype == "fp32" else None
  optimizer = optax.adamw(
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      mu_dtype=_master_dtype_jax,
  )
  print(
      f"[opt] adamw mu/nu_dtype = {master_dtype} "
      f"(weights_dtype = {weights_dtype})",
      flush=True,
  )
  # `optimizer.init` produces:
  #  - `mu`/`nu`: `zeros_like(params)` → inherit the per-weight sharding.
  #  - small scalars (`count`, `learning_rate` schedule state): default to
  #    the host's first device only, which fails `compile_step_func`'s
  #    out_shardings match. Walk the pytree and replicate every leaf that
  #    doesn't already span the mesh.
  _replicated = NamedSharding(mesh, P())
  _opt_state_raw = optimizer.init(jax_view(jittable_mod.params))

  def _fix_leaf_sharding(leaf):
    if isinstance(leaf, jax.Array) and len(leaf.sharding.device_set) < n_global:
      return jax.device_put(leaf, _replicated)
    return leaf

  opt_state = torch_view(jax.tree.map(_fix_leaf_sharding, _opt_state_raw))

  if use_tokamax_ce:
    # Local make_train_step that threads `weights` through loss_fn (canonical
    # torchax.train.make_train_step only passes (res, label) to loss_fn). The
    # tokamax-CE path needs `weights["lm_head.weight"]` inside the loss.
    def _make_train_step_with_weights(model_fn_in, loss_fn_in, optax_optimizer):
      env_local = torchax.default_env()

      def _loss(weights, buffers, args, label):
        with env_local, jax.named_scope("compute_loss"):
          res = model_fn_in(weights, buffers, args)
          return loss_fn_in(res, label, weights)

      grad_fn = torchax.interop.jax_value_and_grad(_loss)

      def step(weights, buffers, opt_state, args, label):
        with jax.named_scope("compute_gradient"):
          loss_v, gradient = grad_fn(weights, buffers, args, label)
        with jax.named_scope("optimizer_updates"):
          updates, opt_state = torchax.interop.call_jax(
              optax_optimizer.update, gradient, opt_state, weights
          )
          weights = torchax.interop.call_jax(
              optax.apply_updates, weights, updates
          )
        return loss_v, weights, opt_state

      return step

    train_step = _make_train_step_with_weights(model_fn, loss_fn, optimizer)
  else:
    train_step = torchax.train.make_train_step(
        model_fn,
        loss_fn,
        optimizer,
        remat_policy=jax.checkpoint_policies.nothing_saveable,
    )

  x_sharding = NamedSharding(mesh, P("fsdp"))

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
        # Explicit precompile via canonical helper (donate_argnums + cost analysis).
        train_step = helper.compile_step_func(
            train_step,
            jittable_mod.params,
            jittable_mod.buffers,
            opt_state,
            inputs,
            labels,
            mesh,
        )

      if profile_dir and i == profile_step:
        jax.profiler.start_trace(profile_dir)

      t0 = time.perf_counter()
      loss, jittable_mod.params, opt_state = train_step(
          jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )
      torchax.interop.call_jax(
          jax.block_until_ready, (loss, jittable_mod.params)
      )
      dt = time.perf_counter() - t0

      if profile_dir and i == profile_step:
        jax.profiler.stop_trace()

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

  if total_time_after_warmup > 0:
    avg_tps = total_tokens_after_warmup / total_time_after_warmup
    per_chip = avg_tps / n_global
    avg_step_time = total_time_after_warmup / n_measured_steps
    peak = 918e12  # v6e bf16 peak ≈ 918 TFLOPS / chip.
    # MaxText-style train-step TFLOPs (forward + 2× backward = ×3) for a
    # dense GQA Llama. Dropped: MoE / MLA / engram / vision / DPO / MTP.
    # Source: maxtext/src/maxtext/utils/maxtext_utils.py
    # `calculate_tflops_training_per_device`.
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
    # SwiGLU has 2 input projections (gate + up) and 1 output (down).
    ffn_flops = 2 * B * L * Mlp * D * (2 + 1)
    embed_flops = (
        2 * B * L * D * V
    )  # only the lm_head; embedding lookup is gather.
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


def _load_config(model_id: str):
  """Load `LlamaConfig` for HF model_id without downloading weights."""
  from model import LlamaConfig

  return LlamaConfig.from_pretrained(model_id)


if __name__ == "__main__":
  fire.Fire(main)
