---
title: 'Module: test/test_functions.py'
type: catalog
provenance: extracted
module: test/test_functions.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_functions`/
symbols:
  TestTorchFunctions.setUp: TestTorchFunctions#setUp().
  TestTorchFunctions.env: TestTorchFunctions#env.
  SeqModel.forward: SeqModel#forward().
  TestTorchFunctions.test_dont_capture_conversion: TestTorchFunctions#test_dont_capture_conversion().
  TestTorchFunctions.test_rnn: TestTorchFunctions#test_rnn().
  TestTorchFunctions.test_tensor_constructor: TestTorchFunctions#test_tensor_constructor().
  TestTorchFunctions.test_brackets: TestTorchFunctions#test_brackets().
  TestTorchFunctions.test_bernoulli_inplace: TestTorchFunctions#test_bernoulli_inplace().
  TestTorchFunctions.test_flatten: TestTorchFunctions#test_flatten().
  TestTorchFunctions.test_rms_norm: TestTorchFunctions#test_rms_norm().
  SeqModel: SeqModel#
  SeqModel.gru: SeqModel#gru.
  SeqModel.linear: SeqModel#linear.
  SeqModel.__init__: SeqModel#__init__().
  TestTorchFunctions: TestTorchFunctions#
  TestTorchFunctions.test_full_int: TestTorchFunctions#test_full_int().
  TestTorchFunctions.test_requires_grad: TestTorchFunctions#test_requires_grad().
---
# Module: [`test/test_functions.py`](../../../../../raw/code/torchax/test/test_functions.py)

## Classes
### `SeqModel`  ·  implements/extends Module
- def: [`test/test_functions.py:24`](../../../../../raw/code/torchax/test/test_functions.py#L24)
- doc: Architecture is LLM generated
- signature: `class SeqModel(torch.nn.Module):`
- members:
  - `forward(self, x: torch.Tensor)` — [`L32`](../../../../../raw/code/torchax/test/test_functions.py#L32)
  - `gru` — [`L29`](../../../../../raw/code/torchax/test/test_functions.py#L29)
  - `linear` — [`L30`](../../../../../raw/code/torchax/test/test_functions.py#L30)
- protocol/private: `__init__`[`L27`](../../../../../raw/code/torchax/test/test_functions.py#L27)
- used by: (1 test-only callers)

### `TestTorchFunctions`  ·  implements/extends TestCase
- def: [`test/test_functions.py:38`](../../../../../raw/code/torchax/test/test_functions.py#L38)
- signature: `class TestTorchFunctions(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L39`](../../../../../raw/code/torchax/test/test_functions.py#L39)
  - `test_bernoulli_inplace(self)` — [`L76`](../../../../../raw/code/torchax/test/test_functions.py#L76)
  - `test_brackets(self)` — [`L70`](../../../../../raw/code/torchax/test/test_functions.py#L70)
  - `test_dont_capture_conversion(self)` — [`L64`](../../../../../raw/code/torchax/test/test_functions.py#L64)
  - `test_flatten(self)` — [`L81`](../../../../../raw/code/torchax/test/test_functions.py#L81)
  - `test_full_int(self)` — [`L60`](../../../../../raw/code/torchax/test/test_functions.py#L60)
  - `test_requires_grad(self, func, args)` — [`L141`](../../../../../raw/code/torchax/test/test_functions.py#L141)
  - `test_rms_norm(self)` — [`L119`](../../../../../raw/code/torchax/test/test_functions.py#L119)
  - `test_rnn(self)` — [`L107`](../../../../../raw/code/torchax/test/test_functions.py#L107)
  - `test_tensor_constructor(self, arg, kwargs=None)` — [`L51`](../../../../../raw/code/torchax/test/test_functions.py#L51)
  - `env` — [`L42`](../../../../../raw/code/torchax/test/test_functions.py#L42)
- uses (calls/refs, reference-scoped): [`Tensor`](../torchax/tensor.md#Tensor), [`default_env`](../torchax/__init__.md#default_env), [`enable_globally`](../torchax/__init__.md#enable_globally), [`enable_accuracy_mode`](../torchax/__init__.md#enable_accuracy_mode), [`to_xla`](../torchax/tensor.md#Environment.to_xla)  (1 test-only)

