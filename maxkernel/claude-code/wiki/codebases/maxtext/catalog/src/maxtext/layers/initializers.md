---
title: 'Module: src/maxtext/layers/initializers.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/initializers.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.initializers`/
symbols:
  NdInitializer: NdInitializer.
  nd_dense_init: nd_dense_init().
  InitializerAxis: InitializerAxis.
  nd_dense_init.init_fn: nd_dense_init().init_fn().
  Initializer: Initializer.
  default_embed_init: default_embed_init.
  default_bias_init: default_bias_init.
  default_scalar_init: default_scalar_init.
  variable_to_logically_partitioned: variable_to_logically_partitioned().
---
# Module: [`src/maxtext/layers/initializers.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py)

## Functions
- `init_fn(key, shape, dtype, in_axis, out_axis)` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L55) — Initializes an array using variance scaling with specified axes.
- `nd_dense_init(scale, mode, distribution)` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L37) — Creates a variance-scaling initializer with dynamic in/out axes.
- `variable_to_logically_partitioned(variable: nnx.Variable)` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L63) — Wraps an NNX variable's value in `nn.LogicallyPartitioned`.

## Module values
- `Initializer` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L27)
- `InitializerAxis` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L28)
- `NdInitializer` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L29)
- `default_bias_init` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L33)
- `default_embed_init` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L31)
- `default_scalar_init` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/layers/initializers.py#L34)

