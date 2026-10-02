---
title: 'Module: export_test/export_test.py'
type: catalog
provenance: extracted
module: export_test/export_test.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `export_test.export_test`/
symbols:
  out: out.
  exported: exported.
  b: b.
  MyModel.forward: MyModel#forward().
  forward: forward.
  m: m.
  end: end.
  NNMLIR_LIB: NNMLIR_LIB.
  output: output.
  quantization_module_to_quantspec_pass: quantization_module_to_quantspec_pass().
  res: res.
  timeit: timeit().
  old_params: old_params.
  sample_input: sample_input.
  optim: optim.
  env: env.
  model: model.
  func_to_export: func_to_export().
  example_inputs_jax: example_inputs_jax.
  a: a.
  trip_softmax: trip_softmax().
  inputs: inputs.
  timeit.wrapper: timeit().wrapper().
  quant_spec_dummy: quant_spec_dummy().
  MyModel: MyModel#
  MyModel.fc1: MyModel#fc1.
  MyModel.fc2: MyModel#fc2.
  MyModel.fc3: MyModel#fc3.
  MyModel.loops: MyModel#loops.
  quantization_saturation_pass: quantization_saturation_pass().
  t: t.
  quantization_module_to_quantspec_pass.quantstub: quantization_module_to_quantspec_pass().quantstub().
  start: start.
  MyModel.__init__: MyModel#__init__().
  use_jax_for_the_last_step: use_jax_for_the_last_step.
  make_shape_struct: make_shape_struct().
---
# Module: [`export_test/export_test.py`](../../../../../raw/code/learning-machine/export_test/export_test.py)

## Classes
### `MyModel`  ·  implements/extends Module
- def: [`export_test/export_test.py:65`](../../../../../raw/code/learning-machine/export_test/export_test.py#L65)
- signature: `class MyModel(nn.Module):`
- members:
  - `forward(self, x)` — [`L73`](../../../../../raw/code/learning-machine/export_test/export_test.py#L73)
  - `fc1` — [`L68`](../../../../../raw/code/learning-machine/export_test/export_test.py#L68)
  - `fc2` — [`L69`](../../../../../raw/code/learning-machine/export_test/export_test.py#L69)
  - `fc3` — [`L70`](../../../../../raw/code/learning-machine/export_test/export_test.py#L70)
  - `loops` — [`L71`](../../../../../raw/code/learning-machine/export_test/export_test.py#L71)
- protocol/private: `__init__`[`L66`](../../../../../raw/code/learning-machine/export_test/export_test.py#L66)
- uses (calls/refs, reference-scoped): [`trip_softmax`](export_test.md#trip_softmax)
- used by: [`m`](export_test.md#m)

## Functions
- `func_to_export(x)` — [`L198`](../../../../../raw/code/learning-machine/export_test/export_test.py#L198)
- `make_shape_struct(x)` — [`L190`](../../../../../raw/code/learning-machine/export_test/export_test.py#L190)
- `quant_spec_dummy(t, min, max)` — [`L36`](../../../../../raw/code/learning-machine/export_test/export_test.py#L36)
- `quantization_module_to_quantspec_pass(gm: torch.fx.GraphModule)` — [`L138`](../../../../../raw/code/learning-machine/export_test/export_test.py#L138)
- `quantization_saturation_pass(gm: torch.fx.GraphModule)` — [`L99`](../../../../../raw/code/learning-machine/export_test/export_test.py#L99)
- `quantstub(x, tmin, tmax)` — [`L148`](../../../../../raw/code/learning-machine/export_test/export_test.py#L148)
- `timeit(func)` — [`L17`](../../../../../raw/code/learning-machine/export_test/export_test.py#L17)
- `trip_softmax(x, dim)` — [`L55`](../../../../../raw/code/learning-machine/export_test/export_test.py#L55)
- `wrapper(*args, **kwargs)` — [`L19`](../../../../../raw/code/learning-machine/export_test/export_test.py#L19)

## Module values
- `NNMLIR_LIB` — [`L30`](../../../../../raw/code/learning-machine/export_test/export_test.py#L30)
- `a` — [`L118`](../../../../../raw/code/learning-machine/export_test/export_test.py#L118)
- `b` — [`L118`](../../../../../raw/code/learning-machine/export_test/export_test.py#L118)
- `end` — [`L203`](../../../../../raw/code/learning-machine/export_test/export_test.py#L203)
- `env` — [`L195`](../../../../../raw/code/learning-machine/export_test/export_test.py#L195)
- `example_inputs_jax` — [`L201`](../../../../../raw/code/learning-machine/export_test/export_test.py#L201)
- `exported` — [`L94`](../../../../../raw/code/learning-machine/export_test/export_test.py#L94)
- `forward` — [`L114`](../../../../../raw/code/learning-machine/export_test/export_test.py#L114)
- `inputs` — [`L88`](../../../../../raw/code/learning-machine/export_test/export_test.py#L88)
- `m` — [`L84`](../../../../../raw/code/learning-machine/export_test/export_test.py#L84)
- `model` — [`L197`](../../../../../raw/code/learning-machine/export_test/export_test.py#L197)
- `old_params` — [`L85`](../../../../../raw/code/learning-machine/export_test/export_test.py#L85)
- `optim` — [`L121`](../../../../../raw/code/learning-machine/export_test/export_test.py#L121)
- `out` — [`L124`](../../../../../raw/code/learning-machine/export_test/export_test.py#L124)
- `output` — [`L91`](../../../../../raw/code/learning-machine/export_test/export_test.py#L91)
- `res` — [`L202`](../../../../../raw/code/learning-machine/export_test/export_test.py#L202)
- `sample_input` — [`L90`](../../../../../raw/code/learning-machine/export_test/export_test.py#L90)
- `start` — [`L188`](../../../../../raw/code/learning-machine/export_test/export_test.py#L188)
- `t` — [`L123`](../../../../../raw/code/learning-machine/export_test/export_test.py#L123)
- `use_jax_for_the_last_step` — [`L174`](../../../../../raw/code/learning-machine/export_test/export_test.py#L174)

