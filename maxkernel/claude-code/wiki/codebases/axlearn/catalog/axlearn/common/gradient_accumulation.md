---
title: 'Module: axlearn/common/gradient_accumulation.py'
type: catalog
provenance: extracted
module: axlearn/common/gradient_accumulation.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.gradient_accumulation`/
symbols:
  with_minibatch_steps.decorator.fwd_helper: with_minibatch_steps().decorator().fwd_helper().
  with_minibatch_steps.decorator.fwd_helper.scan_body: with_minibatch_steps().decorator().fwd_helper().scan_body().
  with_minibatch_steps.decorator.func_fwd: with_minibatch_steps().decorator().func_fwd().
  with_minibatch_steps.decorator.original_func_positional_args: with_minibatch_steps().decorator().original_func_positional_args().
  with_minibatch_steps.decorator.func: with_minibatch_steps().decorator().func().
  with_minibatch_steps.decorator: with_minibatch_steps().decorator().
  with_minibatch_steps.decorator.sequential_vmap.wrapper: with_minibatch_steps().decorator().sequential_vmap().wrapper().
  with_minibatch_steps: with_minibatch_steps().
  _make_scan_minibatch_inputs: _make_scan_minibatch_inputs().
  _compute_minibatch_size: _compute_minibatch_size().
  with_minibatch_steps.decorator.sequential_vmap: with_minibatch_steps().decorator().sequential_vmap().
  with_minibatch_steps.decorator.func_bwd: with_minibatch_steps().decorator().func_bwd().
  with_minibatch_steps.decorator.fwd_helper.reshape_for_scan: with_minibatch_steps().decorator().fwd_helper().reshape_for_scan().
---
# Module: [`axlearn/common/gradient_accumulation.py`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py)

## Functions
- `_compute_minibatch_size(input_batch: Nested[Tensor], *, steps: int)` — [`L23`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L23) — Utility function to compute minibatch size from input batch.
- `_make_scan_minibatch_inputs(inputs: Nested[Tensor], *, forward_key: Tensor, param_noise_key: Tensor, minibatch_index: int)` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L56) — Creates minibatch inputs from inputs.
- `decorator(fn: ForwardFn)` — [`L145`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L145)
- `func(model_params: Nested[Tensor], inputs: Any)` — [`L310`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L310) — Wrap original function to pass in key-word args.
- `func_bwd(saved_fwd_state, grad_from_later_in_network)` — [`L325`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L325) — Defines backward pass for the custom vjp based gradient computation.
- `func_fwd(model_params: Nested[Tensor], inputs: Any)` — [`L314`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L314) — Defines forward pass for the custom vjp based gradient computation.
- `fwd_helper(model_params: Nested[Tensor], inputs: Any, compute_grad: bool)` — [`L156`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L156) — Helper function that scans a ForwardFn over minibatches.
- `original_func_positional_args(model_params: Nested[Tensor], inputs: Any)` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L150)
- `reshape_for_scan(x: Tensor)` — [`L174`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L174) — Helper function that adds a minibatch dimension while evenly dividing
- `scan_body(carry: tuple[Nested[Tensor], Nested[Tensor], Tensor, Tensor, MetricAccumulator], minibatch_index: int)` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L240) — Computes minibatch forward outputs and, optionally, gradients.
- `sequential_vmap(func: ForwardFn)` — [`L294`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L294) — Decorates a ForwardFn to process the input in minibatches.
- `with_minibatch_steps(*, steps: int, metric_accumulator: ConfigOr[MetricAccumulator], grad_dtype: Optional[jnp.dtype] = None)` — [`L95`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L95) — Decorate a ForwardFn to accumulate gradients over minibatch steps.
- `wrapper(model_params: Nested[Tensor], inputs: Any)` — [`L298`](../../../../../../raw/code/axlearn/axlearn/common/gradient_accumulation.py#L298)

