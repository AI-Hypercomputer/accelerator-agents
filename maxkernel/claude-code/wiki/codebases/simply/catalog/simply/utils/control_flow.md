---
title: 'Module: simply/utils/control_flow.py'
type: catalog
provenance: extracted
module: simply/utils/control_flow.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.control_flow`/
symbols:
  ControlFlow.apply: ControlFlow#apply().
  ScanModule._process_module: ScanModule#_process_module().
  ScanModule.apply: ScanModule#apply().
  ScanModule.init: ScanModule#init().
  ControlFlow.modules: ControlFlow#modules.
  Constant: Constant#
  ControlFlow: ControlFlow#
  ScanModule: ScanModule#
  ScanModule.module: ScanModule#module.
  OutputSpec.OutputSpec: OutputSpec.OutputSpec.
  ControlFlow._set_state_value: ControlFlow#_set_state_value().
  ControlFlow.steps: ControlFlow#steps.
  ControlFlow.init: ControlFlow#init().
  ScanModule.setup: ScanModule#setup().
  ControlFlow.output_spec: ControlFlow#output_spec.
  ScanModule.overwrite_input_spec: ScanModule#overwrite_input_spec.
  Constant.value: Constant#value.
  ControlStep.module: ControlStep#module.
  ControlStep.overwrite_input_spec: ControlStep#overwrite_input_spec.
  ControlStep.output_spec: ControlStep#output_spec.
  ControlStep: ControlStep#
  default_input_spec: default_input_spec().
  ScanModule.length: ScanModule#length.
  ScanModule.unroll: ScanModule#unroll.
  ScanModule.per_step_args: ScanModule#per_step_args.
  ControlStep.module_name: ControlStep#module_name.
  ControlStep.fn: ControlStep#fn.
  ControlStep.module_fn: ControlStep#module_fn.
  ControlFlow.setup: ControlFlow#setup().
---
# Module: [`simply/utils/control_flow.py`](../../../../../../raw/code/simply/simply/utils/control_flow.py)

## Classes
### `Constant`
- def: [`simply/utils/control_flow.py:35`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L35)
- signature: `class Constant:`
- members:
  - `value` — [`L36`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L36)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry), [`PyTree`](common.md#PyTree.PyTree)
- used by: [`apply`](control_flow.md#ControlFlow.apply), [`_process_module`](control_flow.md#ScanModule._process_module), [`overwrite_input_spec`](control_flow.md#ScanModule.overwrite_input_spec), [`overwrite_input_spec`](control_flow.md#ControlStep.overwrite_input_spec)

### `ControlFlow`
- def: [`simply/utils/control_flow.py:93`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L93)
- doc: A sequence of ControlSteps that can be executed in order.
- signature: `class ControlFlow(module_lib.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: PyTree, **kwargs)` — [`L121`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L121)
  - `init(self, prng_key: jax.Array)` — [`L113`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L113)
  - `setup(self)` — [`L99`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L99)
  - `modules` — [`L101`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L101)
  - `output_spec` — [`L97`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L97)
  - `steps` — [`L96`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L96)
- protocol/private: `_set_state_value`[`L163`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L163)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`PyTree`](common.md#PyTree.PyTree), [`get`](registry.md#RootRegistry.get), [`traverse_tree_with_path`](pytree.md#traverse_tree_with_path), [`set_tree_value`](pytree.md#set_tree_value), [`SimplyModule`](module.md#SimplyModule), [`ModuleRegistry`](module.md#ModuleRegistry), [`tree_value`](pytree.md#tree_value), [`FunctionRegistry`](registry.md#FunctionRegistry), [`Constant`](control_flow.md#Constant), [`OutputSpec`](control_flow.md#OutputSpec.OutputSpec), [`ControlStep`](control_flow.md#ControlStep), [`default_input_spec`](control_flow.md#default_input_spec)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#SimplyModule.setup), [`ControlFlow`](control_flow_test.md#ControlFlow)

### `ControlStep`
- def: [`simply/utils/control_flow.py:46`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L46)
- doc: Control step.
- signature: `class ControlStep:`
- members:
  - `fn` — [`L72`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L72)
  - `module` — [`L71`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L71)
  - `module_fn` — [`L73`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L73)
  - `module_name` — [`L70`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L70)
  - `output_spec` — [`L78`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L78)
  - `overwrite_input_spec` — [`L75`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L75)
- uses (calls/refs, reference-scoped): [`SimplyModule`](module.md#SimplyModule), [`Constant`](control_flow.md#Constant), [`OutputSpec`](control_flow.md#OutputSpec.OutputSpec)
- used by: [`ControlStep`](control_flow_test.md#ControlStep), [`steps`](control_flow.md#ControlFlow.steps)

### `ScanModule`
- def: [`simply/utils/control_flow.py:176`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L176)
- doc: Scan module.
- signature: `class ScanModule(module_lib.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: PyTree, **kwargs: Mapping[str, Any])` — [`L197`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L197)
  - `init(self, prng_key: jax.Array)` — [`L190`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L190)
  - `setup(self)` — [`L187`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L187)
  - `length` — [`L180`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L180)
  - `module` — [`L179`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L179)
  - `overwrite_input_spec` — [`L183`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L183)
  - `per_step_args` — [`L182`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L182)
  - `unroll` — [`L181`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L181)
- protocol/private: `_process_module`[`L208`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L208)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`PyTree`](common.md#PyTree.PyTree), [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#SimplyModule.setup), [`SimplyModule`](module.md#SimplyModule), [`ModuleRegistry`](module.md#ModuleRegistry), [`tree_value`](pytree.md#tree_value), [`Constant`](control_flow.md#Constant), [`default_input_spec`](control_flow.md#default_input_spec)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#SimplyModule.setup), [`test_scan`](control_flow_test.md#ControlFlowTest.test_scan)

## Functions
- `default_input_spec(fn: Callable[..., Any])` — [`L81`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L81)

## Module values
- `OutputSpec` — [`L39`](../../../../../../raw/code/simply/simply/utils/control_flow.py#L39)

