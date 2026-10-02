---
title: 'Module: test/test_conv.py'
type: catalog
provenance: extracted
module: test/test_conv.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_conv`/C
symbols:
  ConvTest.test_conv1: onvTest#test_conv1().
  ConvTest.test_conv2: onvTest#test_conv2().
  CustomConv1.forward: ustomConv1#forward().
  CustomConv2.forward: ustomConv2#forward().
  ConvTest: onvTest#
  CustomConv1: ustomConv1#
  CustomConv1.conv1: ustomConv1#conv1.
  CustomConv1.conv2: ustomConv1#conv2.
  CustomConv1.fc1: ustomConv1#fc1.
  CustomConv2: ustomConv2#
  CustomConv2.conv: ustomConv2#conv.
  CustomConv2.avg_pool: ustomConv2#avg_pool.
  CustomConv2.scale: ustomConv2#scale.
  CustomConv1.__init__: ustomConv1#__init__().
  CustomConv2.__init__: ustomConv2#__init__().
---
# Module: [`test/test_conv.py`](../../../../../raw/code/torchax/test/test_conv.py)

## Classes
### `ConvTest`  ·  implements/extends TestCase
- def: [`test/test_conv.py:69`](../../../../../raw/code/torchax/test/test_conv.py#L69)
- signature: `class ConvTest(base_test_util.TestCase):`
- members:
  - `test_conv1(self)` — [`L70`](../../../../../raw/code/torchax/test/test_conv.py#L70)
  - `test_conv2(self)` — [`L82`](../../../../../raw/code/torchax/test/test_conv.py#L82)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`extract_jax`](../torchax/__init__.md#extract_jax), [`t2j_copy`](../torchax/tensor.md#Environment.t2j_copy), [`j2t_copy`](../torchax/tensor.md#Environment.j2t_copy)  (3 test-only)

### `CustomConv1`  ·  implements/extends Module
- def: [`test/test_conv.py:23`](../../../../../raw/code/torchax/test/test_conv.py#L23)
- signature: `class CustomConv1(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L37`](../../../../../raw/code/torchax/test/test_conv.py#L37)
  - `conv1` — [`L33`](../../../../../raw/code/torchax/test/test_conv.py#L33)
  - `conv2` — [`L34`](../../../../../raw/code/torchax/test/test_conv.py#L34)
  - `fc1` — [`L35`](../../../../../raw/code/torchax/test/test_conv.py#L35)
- protocol/private: `__init__`[`L24`](../../../../../raw/code/torchax/test/test_conv.py#L24)
- used by: (1 test-only callers)

### `CustomConv2`  ·  implements/extends Module
- def: [`test/test_conv.py:45`](../../../../../raw/code/torchax/test/test_conv.py#L45)
- signature: `class CustomConv2(nn.Module):`
- members:
  - `forward(self, x)` — [`L58`](../../../../../raw/code/torchax/test/test_conv.py#L58)
  - `avg_pool` — [`L54`](../../../../../raw/code/torchax/test/test_conv.py#L54)
  - `conv` — [`L51`](../../../../../raw/code/torchax/test/test_conv.py#L51)
  - `scale` — [`L56`](../../../../../raw/code/torchax/test/test_conv.py#L56)
- protocol/private: `__init__`[`L46`](../../../../../raw/code/torchax/test/test_conv.py#L46)
- used by: (1 test-only callers)

