#!/usr/bin/env python
# pylint: skip-file
# ==========================================================================
# UNTESTED SCAFFOLD.
#
# This trainer was written in one shot against the torchax codebase page
# (`wiki/codebases/torchax.md`, commit 8f957d1) and the jax-huggingface Part
# 2 recipe (`wiki/sources/2026-jax-huggingface-part-2.md`). It has not been
# executed: no Gemma 4 weights were downloaded, no JAX was installed in the
# authoring env, no step was run.
#
# Expect:
#   - HF pytree registrations may be stale vs whatever `transformers`
#     version you actually install; patch the registrations in `main()` if
#     `CausalLMOutputWithPast` / cache classes have moved.
#   - Parameter sharding depends on state_dict names matching the Gemma 4
#     convention — verify with a quick `print(next(iter(model.state_dict())))`
#     before the first real run.
#   - `num_kv_heads = 2` on Gemma 4 E4B does not divide tp=8 — K/V are
#     replicated. See `model/sharding.py`.
#   - First profile run should check captured-constants HLO (see
#     `2026-jax-huggingface-part-3.md`); if weights get inlined we will
#     need to switch to the `functional_call` pattern.
# ==========================================================================
"""Gemma 4 E4B fine-tune trainer — torchax baseline.

Loads Gemma 4 in bf16 via HuggingFace, moves weights to JAX via torchax,
applies the NeMo-Megatron TP sharding from `model/sharding.py`, trains for
`--steps` steps on a wikitext packed dataset, and optionally captures an
xprof trace around `--profile_steps`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional

# These imports are inside main() so `--help` works without a full JAX env.


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

DEFAULTS = {
    "model_id": "google/gemma-4-E4B",
    "dataset": "wikitext-2-raw-v1",
    # Exp 52+: new default seq_len=8192 (was 2048) to match the JAX stack.
    "seq_len": 8192,
    "batch_size": 4,
    "steps": 20,
    "learning_rate": 1e-5,
    "warmup_steps": 2,
    "strategy": (
        "fsdp"
    ),  # default: FSDP on all chips; "tp" selects Megatron-style TP
    "dp": 1,
    "tp": 1,
    "fsdp": 0,  # 0 == jax.device_count()
    # Legacy single-dtype shortcut. On torchax the split flags below are
    # recognized but not fully wired — torchax models dispatch through
    # HF PyTorch which takes a single torch_dtype. Passing
    # --weights-dtype fp32 --compute-dtype bf16 on torchax triggers a
    # warning and falls back to bf16 everywhere.
    "dtype": "bf16",
    "weights_dtype": "bf16",  # torchax defaults to legacy bf16
    "compute_dtype": "bf16",
    "log_every": 1,
    "grad_accum": 1,
    "seed": 42,
}


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      prog="gemma4-torchax-trainer",
      description="Gemma 4 E4B fine-tune (torchax) — performance baseline.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument(
      "--model_id", default=DEFAULTS["model_id"], help="HF model id to load."
  )
  p.add_argument(
      "--dataset",
      default=DEFAULTS["dataset"],
      help=(
          "wikitext dataset config (e.g. wikitext-2-raw-v1,"
          " wikitext-103-raw-v1)."
      ),
  )
  p.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
  p.add_argument(
      "--batch_size",
      type=int,
      default=DEFAULTS["batch_size"],
      help="Per-chip batch size; global batch = batch_size * dp.",
  )
  p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
  p.add_argument(
      "--learning_rate", type=float, default=DEFAULTS["learning_rate"]
  )
  p.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
  p.add_argument(
      "--strategy",
      choices=["fsdp", "tp"],
      default=DEFAULTS["strategy"],
      help=(
          "Sharding strategy. 'fsdp' shards every param's largest "
          "dim across all chips (default, best for fine-tuning). "
          "'tp' is Megatron-style tensor-parallel."
      ),
  )
  p.add_argument(
      "--fsdp",
      type=int,
      default=DEFAULTS["fsdp"],
      help=(
          "FSDP mesh size (0 = jax.device_count()). Only used "
          "when --strategy fsdp."
      ),
  )
  p.add_argument(
      "--dp",
      type=int,
      default=DEFAULTS["dp"],
      help="Data-parallel mesh axis size (TP strategy only).",
  )
  p.add_argument(
      "--tp",
      type=int,
      default=DEFAULTS["tp"],
      help="Tensor-parallel mesh axis size (TP strategy only).",
  )
  p.add_argument(
      "--dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["dtype"],
      help=(
          "Legacy single-dtype. Primary on torchax — model is "
          "loaded at this dtype via HF PyTorch."
      ),
  )
  # Exp 52+: split flags accepted for CLI parity with the JAX trainer.
  # On torchax the mixed-precision wiring is NOT implemented — the HF
  # PyTorch model takes a single torch_dtype. If you ask for fp32 master
  # + bf16 compute on torchax, a warning is printed and the run falls
  # back to the legacy --dtype value. Full support is JAX-only.
  p.add_argument(
      "--weights-dtype",
      dest="weights_dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["weights_dtype"],
      help=(
          "Storage dtype for model params. torchax: stub (warns + falls "
          "back to --dtype). Use the JAX trainer for full AMP support."
      ),
  )
  p.add_argument(
      "--compute-dtype",
      dest="compute_dtype",
      choices=["bf16", "fp32"],
      default=DEFAULTS["compute_dtype"],
      help=(
          "Compute dtype for matmul/conv. torchax: stub (warns + falls "
          "back to --dtype). Use the JAX trainer for full AMP support."
      ),
  )
  p.add_argument(
      "--checkpoint_dir",
      default=None,
      help="If set, save torchax checkpoint at end. Skipped if unset.",
  )
  p.add_argument(
      "--profile_dir",
      default=None,
      help="jax.profiler trace output dir. If unset: no trace.",
  )
  p.add_argument(
      "--profile_steps",
      type=int,
      nargs="*",
      default=[],
      help=(
          "Step indices (0-based) to capture; trace brackets "
          "min..max of this list. Example: --profile_steps 5 6 7"
      ),
  )
  p.add_argument("--log_every", type=int, default=DEFAULTS["log_every"])
  p.add_argument(
      "--grad_accum",
      type=int,
      default=DEFAULTS["grad_accum"],
      help="Gradient accumulation micro-steps per optimizer update.",
  )
  p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
  p.add_argument(
      "--config",
      default=None,
      help="YAML file with default values (CLI overrides file).",
  )
  return p


def _load_yaml_defaults(path: str) -> dict:
  try:
    import yaml  # type: ignore
  except ImportError as e:
    raise ImportError("--config requires pyyaml; `pip install pyyaml`.") from e
  with open(path) as f:
    return yaml.safe_load(f) or {}


# -----------------------------------------------------------------------------
# Pytree registrations — keep close to HF's API and away from import time.
# -----------------------------------------------------------------------------


def _register_hf_pytrees():
  """Make HF output / cache classes crossable by jax.jit.

  Copied from jax-huggingface Parts 1 + 3 pytree cookbook. If these classes
  moved between HF versions, patch here — there is an HF-API-drift note in
  `wiki/codebases/jax-huggingface.md` for exactly this.
  """
  from jax.tree_util import register_pytree_node
  from transformers import modeling_outputs
  from transformers import cache_utils

  def _out_flatten(v):
    return v.to_tuple(), None

  def _causal_unflatten(aux, children):
    return modeling_outputs.CausalLMOutputWithPast(*children)

  register_pytree_node(
      modeling_outputs.CausalLMOutputWithPast,
      _out_flatten,
      _causal_unflatten,
  )

  # Gemma 4 ConditionalGeneration returns its own output class. Register
  # it defensively — if the import path differs in future transformers
  # versions, fail open (the pytree traversal will error visibly).
  try:
    from transformers.models.gemma4 import modeling_gemma4

    for attr in ("Gemma4CausalLMOutputWithPast", "Gemma4ModelOutputWithPast"):
      cls = getattr(modeling_gemma4, attr, None)
      if cls is None:
        continue

      def _make(cls):
        def _unflatten(aux, children):
          return cls(*children)

        return _unflatten

      register_pytree_node(cls, _out_flatten, _make(cls))
  except Exception as e:
    print(f"[pytree] Gemma4 output registration skipped: {e!r}")

  # DynamicCache — present across all recent transformers versions.
  def _dyn_flatten(c):
    return (c.key_cache, c.value_cache), None

  def _dyn_unflatten(aux, children):
    c = cache_utils.DynamicCache()
    c.key_cache, c.value_cache = children
    return c

  register_pytree_node(
      cache_utils.DynamicCache,
      _dyn_flatten,
      _dyn_unflatten,
  )
  # StaticCache registration deferred until we exercise decode (not used
  # in training loss computation).


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
  args = _build_argparser().parse_args(argv)
  if args.config:
    cfg = _load_yaml_defaults(args.config)
    for k, v in cfg.items():
      if getattr(args, k, None) == DEFAULTS.get(k):
        setattr(args, k, v)

  # Seed before we import JAX so jax honors it.
  random.seed(args.seed)
  os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

  import numpy as np
  import jax
  import jax.numpy as jnp
  import optax
  import torch
  import torchax
  from torchax import interop
  from jax.sharding import NamedSharding, PartitionSpec as P

  # Module-local imports.
  this_dir = Path(__file__).resolve().parent
  sys.path.insert(0, str(this_dir))
  from model import Gemma4ForCausalLM, AutoTokenizer  # noqa: E402
  from model.sharding import (  # noqa: E402
      get_mesh,
      get_param_sharding,
      input_sharding,
      replicated as replicated_sharding,
  )
  from data import IGNORE_INDEX, make_dataloader  # noqa: E402

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # Precision / env setup ---------------------------------------------------
  # Exp 52+: CLI mirrors the JAX trainer's --weights-dtype / --compute-dtype,
  # but torchax lowers through HF PyTorch which takes a single torch_dtype.
  # If the caller requested a split (fp32 master + bf16 compute), warn and
  # fall back to --dtype. Full AMP support is JAX-only.
  if args.weights_dtype != args.compute_dtype:
    print(
        "[dtype] WARNING: torchax does not implement split"
        f" --weights-dtype={args.weights_dtype} /"
        f" --compute-dtype={args.compute_dtype}. Falling back to legacy"
        f" --dtype={args.dtype}. Use the JAX trainer for full fp32-master AMP"
        " support."
    )
  else:
    # No split requested → harmless, but print effective dtype.
    print(
        f"[dtype] torchax legacy single-dtype={args.dtype} (split flags"
        f" weights={args.weights_dtype} compute={args.compute_dtype} coincide)."
    )

  if args.dtype == "bf16":
    torchax.enable_performance_mode()  # bf16 matmul, no x64
  else:
    torchax.enable_accuracy_mode()
  _register_hf_pytrees()

  torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

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

  # Model load --------------------------------------------------------------
  # NOTE: we load on CPU in torch_dtype, then move to 'jax'. HF requires
  # a valid HF token with Gemma license acceptance for gated repos — the
  # model card on 2026-04-22 reports Apache-2.0 and NOT gated, but accept
  # the license on HF regardless.
  print(f"[load] {args.model_id} ({torch_dtype})")
  t0 = time.perf_counter()
  env = torchax.default_env()
  # The google/gemma-4-E4B HF checkpoint declares
  # `architectures: ['Gemma4ForConditionalGeneration']` and stores weights
  # under `model.language_model.*`. Loading straight into
  # `Gemma4ForCausalLM` silently re-initialises ~all weights (key miss).
  # So load ForConditionalGeneration (correct weights), then bypass the
  # multimodal orchestrator by monkey-patching `.forward` to call the
  # language-model backbone + lm_head directly. The multimodal forward is
  # not JIT-friendly: it does `input_ids[mask] = pad` with a dynamic bool
  # mask (NonConcreteBooleanIndexError under jax trace).
  from model import Gemma4ForConditionalGeneration
  from transformers.modeling_outputs import CausalLMOutputWithPast
  import types

  model = Gemma4ForConditionalGeneration.from_pretrained(
      args.model_id,
      dtype=torch_dtype,
  )

  def _text_forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=True,
      labels=None,
      **kwargs,
  ):
    lm_outputs = self.model.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    hidden = (
        lm_outputs.last_hidden_state
        if hasattr(lm_outputs, "last_hidden_state")
        else lm_outputs[0]
    )
    logits = self.lm_head(hidden)
    # Gemma 4 applies a final_logit_softcapping in its full forward path;
    # replicate it here so outputs match (and so bf16 logits don't blow up
    # for longer sequences — seen as NaN loss at seq>=2048 without this).
    sc = getattr(self.config.text_config, "final_logit_softcapping", None)
    if sc is not None and sc > 0:
      logits = sc * torch.tanh(logits / sc)
    return CausalLMOutputWithPast(logits=logits)

  model.forward = types.MethodType(_text_forward, model)
  tokenizer = AutoTokenizer.from_pretrained(args.model_id)
  print(f"[load] done in {time.perf_counter() - t0:.1f}s")

  # Exp 8 — swap the default XLA attention path for JAX Pallas splash
  # attention. Registers a new `ALL_ATTENTION_FUNCTIONS` entry and flips
  # `_attn_implementation` on every Gemma4Text layer's `self.config`
  # (shared across layers). Handles causal + sliding-window-512 via
  # splash mask builders; GQA 4:1 is native. See model/pallas_attention.py.
  from model.pallas_attention import register_splash_attention  # noqa: E402

  impl_key = register_splash_attention(mesh)
  # Gemma4Config is a composite: the real attn-impl flag lives on the text
  # sub-config used by Gemma4TextAttention. Set both defensively.
  if hasattr(model.config, "text_config"):
    model.config.text_config._attn_implementation = impl_key
  model.config._attn_implementation = impl_key
  print(f"[attention] using splash_pallas Pallas kernel")

  # Build sharding plan off the torch-side state dict keys --------------
  plan = get_param_sharding(model, mesh)
  for note in plan.notes:
    print(f"[sharding] {note}")
  counts = {k: len(v) for k, v in plan.buckets.items()}
  print(f"[sharding] bucket counts: {counts}")

  # Move weights to 'jax' device and apply sharding ---------------------
  with env:
    model.to("jax")
    sharded_state = {}
    raw_state = model.state_dict()
    for k, v in raw_state.items():
      sh = plan.shardings.get(k, replicated_sharding(mesh))
      # `.apply_jax_` is the in-place torchax op that calls a JAX
      # function on the backing jax.Array. See jax_hg_03.py.
      sharded_state[k] = v.apply_jax_(jax.device_put, sh)
    model.load_state_dict(sharded_state, assign=True, strict=False)

    # Wrap. JittableModule handles dedup for `tie_word_embeddings=True`.
    jmodel = interop.JittableModule(
        model,
        extra_jit_args={"static_argnames": ("use_cache", "return_dict")},
    )

  # Optimizer ---------------------------------------------------------------
  # optax.adamw is the jax-side optimizer. Following the pattern from
  # torchax/train.py:make_train_step — Optax on jax arrays is the cleanest
  # option since gradients from jax.value_and_grad are jax-native.
  lr_schedule = optax.warmup_constant_schedule(
      init_value=0.0,
      peak_value=args.learning_rate,
      warmup_steps=args.warmup_steps,
  )
  optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)

  # Extract jax-side pytree of weights + init optimizer state.
  # Re-apply shardings from the plan to every weight: the tied-weight dedup
  # inside JittableModule can leave a parameter on a single device even
  # though we applied sharding before load_state_dict. Re-sharding is a
  # no-op if the layout already matches and fixes stragglers (Gemma 4's
  # tied lm_head ↔ embed_tokens was the one we saw).
  _replicated = replicated_sharding(mesh)
  weights = interop.jax_view(jmodel.params)  # jax-native dict
  weights = {
      k: jax.device_put(v, plan.shardings.get(k, _replicated))
      for k, v in weights.items()
  }
  buffers = interop.jax_view(jmodel.buffers)
  opt_state = optimizer.init(weights)

  # Data --------------------------------------------------------------------
  # For FSDP: batch is sharded across all chips, so global batch = batch_size
  # * fsdp (each chip sees batch_size rows after sharding). For TP: batch is
  # sharded across dp only, so global = batch_size * dp.
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

  # Pure JAX train step -----------------------------------------------------
  # We stay on the jax side of the wrap/unwrap boundary for the whole step
  # (weights are jax, gradients are jax, optax is jax). The model forward
  # is reached via interop: the `JittableModule.functional_call` path is
  # torchax-aware.
  in_shard = input_sharding(mesh)
  repl = replicated_sharding(mesh)

  def forward_loss(weights, buffers, input_ids, labels):
    """Run the model functionally and compute the CE loss.

    `functional_call` threads `weights + buffers` into the torch side,
    `jax.jit` will compile this through torchax's dispatch.
    """
    # Back to torch-land for the forward — this is where the torchax op
    # table dispatches the model's ops to JAX.
    tw = interop.torch_view(weights)
    tb = interop.torch_view(buffers)
    tin = interop.torch_view(input_ids)
    tlabels = interop.torch_view(labels)  # (no-op if already kept)
    with torchax.default_env():
      out = jmodel.functional_call(
          "forward",
          tw,
          tb,
          tin,
          use_cache=False,
          return_dict=True,
      )
      logits = out.logits  # (B, T, V)
      # Shift is already applied in data.py — labels align with logits.
      # CE across vocab, mean over non-ignore.
      # Exp 10: keep log_softmax in bf16 (not upcasting to fp32). Drops
      # the ~4 GiB `[B, S, V=262144]` fp32 logits tensor that the
      # xprof-mcp TPU-optimization guide calls out. Gemma 4's
      # final_logit_softcapping=30.0 bounds the logits so bf16
      # log_softmax is numerically stable enough. We still promote the
      # final scalar loss to fp32 below.
      vocab = logits.shape[-1]
      flat_logits = logits.reshape(-1, vocab)
      flat_labels = tlabels.reshape(-1)
      mask = (flat_labels != IGNORE_INDEX).to(flat_logits.dtype)
      log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
      safe_labels = torch.where(
          flat_labels == IGNORE_INDEX,
          torch.zeros_like(flat_labels),
          flat_labels,
      )
      picked = log_probs.gather(-1, safe_labels[:, None]).squeeze(-1)
      loss = -(picked * mask).sum() / mask.sum().clamp_min(1.0)
    return interop.jax_view(loss)

  # Reverted to checkpoint (not offload) — offload_dot_with_no_batch_dims
  # hit the same compile-time peak as checkpoint (XLA's planner doesn't
  # account for the offload as freed HBM at compile-time). Stick with
  # the exp 5 winning policy while we figure out seq=2048 batch=2.
  from jax import checkpoint_policies as _ckpt_policies

  grad_fn = jax.value_and_grad(
      jax.checkpoint(
          forward_loss, policy=_ckpt_policies.checkpoint_dots_with_no_batch_dims
      )
  )

  def train_step(weights, buffers, opt_state, input_ids, labels):
    with jax.named_scope("train_step"):
      with jax.named_scope("forward_backward"):
        loss, grads = grad_fn(weights, buffers, input_ids, labels)
      with jax.named_scope("optimizer"):
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
    return loss, weights, opt_state

  # Attempted to pin output shardings to fix step-1 recompile, but the
  # tied lm_head↔embed_tokens sharding plumbing in torchax's JittableModule
  # doesn't cooperate: out_shardings for the tied key collapses to a
  # single-device sharding and jit rejects the mismatch. Left as a
  # documented refuted attempt; step-1 recompile remains open (~150 s).
  jitted_step = jax.jit(
      train_step,
      donate_argnums=(0, 2),  # donate weights + opt_state buffers
  )

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
  # Don't use `with` here — would force re-indenting the whole loop body.
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

    # Move to 'jax' with DP-sharded input layout.
    input_ids = jax.device_put(input_ids_np, in_shard)
    labels = jax.device_put(labels_np, in_shard)

    # Start / stop profile trace.
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
      loss, weights, opt_state = jitted_step(
          weights, buffers, opt_state, input_ids, labels
      )
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
    # In case profile_stop was past --steps.
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

  if args.checkpoint_dir:
    print(f"[ckpt] saving to {args.checkpoint_dir}")
    torchax.save_checkpoint(
        {"weights": weights, "opt_state": opt_state},
        args.checkpoint_dir,
    )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
