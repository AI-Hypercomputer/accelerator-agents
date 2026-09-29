---
title: 'Module: qwix/_src/model.py'
type: catalog
provenance: extracted
module: qwix/_src/model.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.model`/
symbols:
  quantize_model: quantize_model().
  _apply_interceptors: _apply_interceptors().
  quantize_linen_model: quantize_linen_model().
  quantize_nnx_model: quantize_nnx_model().
  quantize_linen_model._is_in_nn_module: quantize_linen_model()._is_in_nn_module().
  _input_transform: _input_transform().
  _output_transform_nnx: _output_transform_nnx().
  ModelType: ModelType.
---
# Module: [`qwix/_src/model.py`](../../../../../../raw/code/qwix/qwix/_src/model.py)

## Functions
- `_apply_interceptors(method: Any, provider: qconfig.QuantizationProvider, output_transform: Any, should_intercept: Any = lambda : True)` — [`L249`](../../../../../../raw/code/qwix/qwix/_src/model.py#L249) — Apply interceptors to a method. — documented in [qwix-_src-model](../../../concepts/qwix-_src-model.md)
- `_input_transform(provider: qconfig.QuantizationProvider, args, kwargs)` — [`L231`](../../../../../../raw/code/qwix/qwix/_src/model.py#L231)
- `_is_in_nn_module()` — [`L99`](../../../../../../raw/code/qwix/qwix/_src/model.py#L99)
- `_output_transform_nnx(provider: qconfig.QuantizationProvider, method_name: str, output: Any)` — [`L237`](../../../../../../raw/code/qwix/qwix/_src/model.py#L237)
- `quantize_linen_model(model: nn.Module, provider: qconfig.QuantizationProvider, methods: Collection[str])` — [`L92`](../../../../../../raw/code/qwix/qwix/_src/model.py#L92) — Quantize a linen model. — documented in [qwix-_src-model](../../../concepts/qwix-_src-model.md)
- `quantize_model(model: nn.Module, provider: qconfig.QuantizationProvider, *model_inputs: Any, methods: Collection[str] = ("__call__",), **model_inputs_kwargs: Any)` — [`L30`](../../../../../../raw/code/qwix/qwix/_src/model.py#L30) — documented in [qwix-_src-model](../../../concepts/qwix-_src-model.md)
- `quantize_nnx_model(model: nnx.Module, provider: qconfig.QuantizationProvider, *model_inputs: Any, call_method: str = "__call__", **model_inputs_kwargs: Any)` — [`L154`](../../../../../../raw/code/qwix/qwix/_src/model.py#L154) — Quantize an NNX model. — documented in [qwix-_src-model](../../../concepts/qwix-_src-model.md)

## Module values
- `ModelType` — [`L26`](../../../../../../raw/code/qwix/qwix/_src/model.py#L26)

