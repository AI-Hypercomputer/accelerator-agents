---
title: 'Module: ejkernel/callib/_pallas_call.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_pallas_call.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._pallas_call`/
symbols:
  buffered_pallas_call: buffered_pallas_call().
  buffered_pallas_call.pallas_call.pipeline: buffered_pallas_call().pallas_call().pipeline().
  buffered_pallas_call.pallas_call: buffered_pallas_call().pallas_call().
  _HBM_ANY: _HBM_ANY.
  buffered_pallas_call._augment_blockspec: buffered_pallas_call()._augment_blockspec().
  buffered_pallas_call.len_: buffered_pallas_call().len_().
  buffered_pallas_call._augment_blockspec.index_map_: buffered_pallas_call()._augment_blockspec().index_map_().
  buffered_pallas_call._bind_pipeline: buffered_pallas_call()._bind_pipeline().
  buffered_pallas_call.pallas_call.pipeline._pipeline: buffered_pallas_call().pallas_call().pipeline()._pipeline().
---
# Module: [`ejkernel/callib/_pallas_call.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py)

## Functions
- `_augment_blockspec(bs, smem_refs)` — [`L148`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L148) — Augment a BlockSpec by binding SMEM references into its index map.
- `_bind_pipeline(spec, count)` — [`L173`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L173) — Configure multi-buffering on a BlockSpec if needed.
- `_pipeline(*args)` — [`L240`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L240) — Invoke the user kernel with SMEM refs, data refs, and scratch refs.
- `buffered_pallas_call(kernel: Callable[..., Any], out_shape: jax.ShapeDtypeStruct, grid_spec: pltpu.PrefetchScalarGridSpec, compiler_params: pltpu.CompilerParams, input_buffer_count: Sequence[int] | None = None, **kw)` — [`L58`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L58) — Create a buffered Pallas call for TPU with custom prefetch and pipeline configuration.
- `index_map_(*idxs)` — [`L164`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L164) — Invoke the original index map with grid indices and SMEM refs.
- `len_(x)` — [`L135`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L135) — Return the length of a sequence, or 1 for a single element.
- `pallas_call(*args)` — [`L193`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L193) — Execute the buffered Pallas kernel with the given arguments.
- `pipeline(*args_refs)` — [`L210`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L210) — Orchestrate the pipelined kernel execution within Pallas.

## Module values
- `_HBM_ANY` — [`L53`](../../../../../../raw/code/ejkernel/ejkernel/callib/_pallas_call.py#L53)

