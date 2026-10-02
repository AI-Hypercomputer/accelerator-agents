---
title: equinox_with_torch — hybrid Equinox/torchax model (cross-framework interop)
type: concept
provenance: mixed
concept: equinox_with_torch
updated: 2026-07-03
status: fresh
---
# equinox_with_torch — hybrid Equinox/torchax model (cross-framework interop)
A minimal MNIST classifier that embeds a `torchax`-jittable plain-PyTorch `nn.Linear` submodule *inside* an Equinox (`eqx.Module`) model, training both the JAX-native and PyTorch-native parameters together through one `optax` optimizer and one `eqx.filter_jit`-compiled step — a working demonstration of the `torchax` interop layer's core promise: PyTorch and JAX modules can be composed and differentiated as a single computation graph.

## Entry points
- [`HybridModel.__init__`](../catalog/equinox_with_torch.md#HybridModel.__init__) — constructs two native `eqx.nn.Linear` layers ([`jax_linear1`](../catalog/equinox_with_torch.md#HybridModel.jax_linear1), [`jax_linear2`](../catalog/equinox_with_torch.md#HybridModel.jax_linear2)) plus one PyTorch [`TorchPart`](../catalog/equinox_with_torch.md#TorchPart) wrapped in `torchax`'s `JittableModule`, and is where the cross-framework bridging is set up once, at model-construction time.
- [`HybridModel.__call__`](../catalog/equinox_with_torch.md#HybridModel.__call__) — the forward pass; every training and inference call flows through here, alternating JAX-native layers with one call into the embedded PyTorch submodule.
- [`train_step`](../catalog/equinox_with_torch.md#train_step) — the `@eqx.filter_jit`-compiled training step; this is where the loss/gradient computation over the *whole* hybrid model (both parameter families) is actually forced through a single JIT boundary.

## Mechanism (step-by-step)
1. [`HybridModel.__init__`](../catalog/equinox_with_torch.md#HybridModel.__init__) builds `TorchPart(128, 128).to("jax")` — moving a plain `torch.nn.Module`'s parameters onto the `"jax"` `torchax` pseudo-device — then wraps it as a `JittableModule`; its parameters are converted to a JAX pytree via `jax_view` and stored on the Equinox module as [`torch_weights`](../catalog/equinox_with_torch.md#HybridModel.torch_weights), which is what makes them visible to Equinox's/Optax's gradient and parameter-update machinery as ordinary leaves.
2. A closure `_call_torch_functional(weights, x)` is built that calls `torch_part.functional_call('forward', weights, torch_part.buffers, x)` — i.e. the PyTorch submodule is invoked *functionally*, with weights passed in explicitly rather than read from `self`, which is what lets it participate in a functional (JAX-style) autodiff graph; this closure is stored as [`_func_call_torch`](../catalog/equinox_with_torch.md#HybridModel._func_call_torch).
3. [`HybridModel.__call__`](../catalog/equinox_with_torch.md#HybridModel.__call__) reshapes the input, runs [`jax_linear1`](../catalog/equinox_with_torch.md#HybridModel.jax_linear1) through `eqx.filter_vmap` (batching a single-example Equinox layer over the batch dimension), then calls `call_torch(self._func_call_torch, self.torch_weights, x)` — `call_torch` is the `torchax` interop primitive that lets a JAX-traced computation call into a PyTorch-semantics function and get a JAX array back out, before applying `jax.nn.relu` and the second linear layer.
4. [`loss_fn`](../catalog/equinox_with_torch.md#loss_fn) calls the model and computes `optax.softmax_cross_entropy_with_integer_labels` — a completely ordinary JAX loss function, oblivious to the fact that part of `model(x)`'s computation happened inside PyTorch.
5. [`train_step`](../catalog/equinox_with_torch.md#train_step) computes `(loss_val, grads) = eqx.filter_value_and_grad(loss_fn)(model, x, y)` — a single autodiff call that differentiates through *both* the Equinox linears and the embedded PyTorch linear (via its functional-call closure) — then applies the update via one shared [`optimizer`](../catalog/equinox_with_torch.md#optimizer) (`optax.adam`) and `eqx.apply_updates`.
6. The training loop converts each PyTorch `DataLoader` batch ([`x_torch`](../catalog/equinox_with_torch.md#x_torch)/[`y_torch`](../catalog/equinox_with_torch.md#y_torch), from [`train_loader`](../catalog/equinox_with_torch.md#train_loader) over [`train_dataset`](../catalog/equinox_with_torch.md#train_dataset)) to JAX arrays ([`x_jax`](../catalog/equinox_with_torch.md#x_jax)/[`y_jax`](../catalog/equinox_with_torch.md#y_jax)) via `.numpy()` before calling [`train_step`](../catalog/equinox_with_torch.md#train_step), and prints [`loss`](../catalog/equinox_with_torch.md#loss) every 100 batches, capped at 500 batches total.

## Key data structures
- [`HybridModel`](../catalog/equinox_with_torch.md#HybridModel) — an `eqx.Module` whose fields mix native JAX layers ([`jax_linear1`](../catalog/equinox_with_torch.md#HybridModel.jax_linear1), [`jax_linear2`](../catalog/equinox_with_torch.md#HybridModel.jax_linear2)) with a JAX-ified view of PyTorch parameters ([`torch_weights`](../catalog/equinox_with_torch.md#HybridModel.torch_weights)) and a non-parameter closure field ([`_func_call_torch`](../catalog/equinox_with_torch.md#HybridModel._func_call_torch)) — this shape is what [`torchax_filter`](../catalog/equinox_with_torch.md#torchax_filter) exists to navigate.
- [`torchax_filter`](../catalog/equinox_with_torch.md#torchax_filter) — used at `optimizer.init(eqx.filter(model, torchax_filter))`, it selects which leaves of `HybridModel` are trainable parameters (returns `True` for `eqx.is_array` leaves *or* `torch.Tensor` leaves) — this is the one place the two frameworks' "what counts as a parameter" notions have to be reconciled into a single Optax-compatible pytree filter.

## Dynamics (design intent)
> [!inferred] The comment inside `__init__` — "buffer part is usually the non-trainable part of the model, so here we pass in as closure and it wont change by training as equinox doesn't know it. If you want to train it then you can assign it to a attr like torch_weights, and take it as input in this function" — states the design tradeoff explicitly: only tensors registered as an `eqx.Module` field (like [`torch_weights`](../catalog/equinox_with_torch.md#HybridModel.torch_weights)) are visible to Equinox's gradient/update machinery; anything captured only in the closure (the PyTorch module's buffers) is frozen with respect to this training loop by construction, not by accident.

## Edge cases
- [`TorchPart`](../catalog/equinox_with_torch.md#TorchPart) is moved `.to("jax")` — a `torchax`-specific pseudo-device string — before being wrapped in `JittableModule`; using this on a `torchax`-unaware PyTorch build would fail, since `"jax"` is not a standard PyTorch device.

## Open questions
> [!inferred] Whether this pattern (Equinox-outer, torchax-inner) generalizes to larger PyTorch submodules or was only validated at the scale of one `nn.Linear(128,128)` is not addressed in the source — this file reads as a minimal proof-of-concept for the interop mechanism, not a scaling study.

## See also
- No other concept page in this repo's ingest exercises the `torchax`/`torch_xla2` interop layer at this level of directness; [llama_ref/model_with_collectives](llama_ref-model_with_collectives.md) and [llama_ref/model_with_scan](llama_ref-model_with_scan.md) use the same `torch_xla2.interop.call_jax`/`torch_view` primitives, but to call JAX collectives from PyTorch rather than to embed a PyTorch module inside a JAX-native one.
