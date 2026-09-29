---
title: 'Module: examples/eager_mode.py'
type: catalog
provenance: extracted
module: examples/eager_mode.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.eager_mode`/
symbols:
  m_compiled: m_compiled.
  m: m.
  MyModel.forward: MyModel#forward().
  inputs: inputs.
  xla_env: xla_env.
  MyModel: MyModel#
  MyModel.fc1: MyModel#fc1.
  MyModel.fc2: MyModel#fc2.
  MyModel.fc3: MyModel#fc3.
  MyModel.__init__: MyModel#__init__().
---
# Module: [`examples/eager_mode.py`](../../../../../raw/code/torchax/examples/eager_mode.py)

## Classes
### `MyModel`  ·  implements/extends Module
- def: [`examples/eager_mode.py:23`](../../../../../raw/code/torchax/examples/eager_mode.py#L23)
- signature: `class MyModel(nn.Module):`
- members:
  - `forward(self, x)` — [`L30`](../../../../../raw/code/torchax/examples/eager_mode.py#L30)
  - `fc1` — [`L26`](../../../../../raw/code/torchax/examples/eager_mode.py#L26)
  - `fc2` — [`L27`](../../../../../raw/code/torchax/examples/eager_mode.py#L27)
  - `fc3` — [`L28`](../../../../../raw/code/torchax/examples/eager_mode.py#L28)
- protocol/private: `__init__`[`L24`](../../../../../raw/code/torchax/examples/eager_mode.py#L24)
- used by: (1 test-only callers)

## Module values
- `inputs` — [`L42`](../../../../../raw/code/torchax/examples/eager_mode.py#L42)
- `m` — [`L38`](../../../../../raw/code/torchax/examples/eager_mode.py#L38)
- `m_compiled` — [`L47`](../../../../../raw/code/torchax/examples/eager_mode.py#L47)
- `xla_env` — [`L20`](../../../../../raw/code/torchax/examples/eager_mode.py#L20)

