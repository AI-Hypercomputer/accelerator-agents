---
title: 'Module: test/test_view.py'
type: catalog
provenance: extracted
module: test/test_view.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_view`/TrainTest#
symbols:
  TrainTest.test_transivity: test_transivity().
  TrainTest.test_outofplace_add: test_outofplace_add().
  TrainTest.test_multiply_tensor_and_view: test_multiply_tensor_and_view().
  TrainTest.test_multiply_views: test_multiply_views().
  TrainTest.setUp: setUp().
  TrainTest.test_copy_: test_copy_().
  TrainTest.test_setitem: test_setitem().
  TrainTest.test_add_: test_add_().
  TrainTest.test_sub_: test_sub_().
  TrainTest.test_mul_: test_mul_().
  TrainTest.test_div_: test_div_().
  TrainTest.test_pow_: test_pow_().
  TrainTest.test_clamp_: test_clamp_().
  TrainTest.test_lt_: test_lt_().
  TrainTest.test_le_: test_le_().
  TrainTest.test_gt_: test_gt_().
  TrainTest.test_ge_: test_ge_().
  TrainTest.test_eq_: test_eq_().
  TrainTest.test_ne_: test_ne_().
  TrainTest.test_bernoulli_: test_bernoulli_().
  TrainTest.test_geometric_: test_geometric_().
  TrainTest.test_normal_: test_normal_().
  TrainTest.test_uniform_: test_uniform_().
  TrainTest.test_relu_: test_relu_().
  TrainTest.test_squeeze_: test_squeeze_().
  TrainTest.test_sqrt_: test_sqrt_().
  TrainTest.test_clamp_min_: test_clamp_min_().
  TrainTest.test_sigmoid_: test_sigmoid_().
  TrainTest.test_tanh_: test_tanh_().
  TrainTest.test_ceil_: test_ceil_().
  TrainTest.test_logical_not_: test_logical_not_().
  TrainTest.test_unsqueeze_: test_unsqueeze_().
  TrainTest.test_transpose_: test_transpose_().
  TrainTest.test_log_normal_: test_log_normal_().
  TrainTest.test_scatter_add_: test_scatter_add_().
  TrainTest.test_scatter_: test_scatter_().
  TrainTest.test_scatter_reduce_: test_scatter_reduce_().
  TrainTest: ''
  TrainTest.test_setitem2: test_setitem2().
---
# Module: [`test/test_view.py`](../../../../../raw/code/torchax/test/test_view.py)

## Classes
### `TrainTest`  ·  implements/extends TestCase
- def: [`test/test_view.py:26`](../../../../../raw/code/torchax/test/test_view.py#L26)
- signature: `class TrainTest(unittest.TestCase):`
- members:
  - `setUp(self)` — [`L27`](../../../../../raw/code/torchax/test/test_view.py#L27)
  - `test_add_(self)` — [`L101`](../../../../../raw/code/torchax/test/test_view.py#L101)
  - `test_bernoulli_(self)` — [`L190`](../../../../../raw/code/torchax/test/test_view.py#L190)
  - `test_ceil_(self)` — [`L303`](../../../../../raw/code/torchax/test/test_view.py#L303)
  - `test_clamp_(self)` — [`L136`](../../../../../raw/code/torchax/test/test_view.py#L136)
  - `test_clamp_min_(self)` — [`L268`](../../../../../raw/code/torchax/test/test_view.py#L268)
  - `test_copy_(self)` — [`L31`](../../../../../raw/code/torchax/test/test_view.py#L31)
  - `test_div_(self)` — [`L122`](../../../../../raw/code/torchax/test/test_view.py#L122)
  - `test_eq_(self)` — [`L175`](../../../../../raw/code/torchax/test/test_view.py#L175)
  - `test_ge_(self)` — [`L168`](../../../../../raw/code/torchax/test/test_view.py#L168)
  - `test_geometric_(self)` — [`L203`](../../../../../raw/code/torchax/test/test_view.py#L203)
  - `test_gt_(self)` — [`L160`](../../../../../raw/code/torchax/test/test_view.py#L160)
  - `test_le_(self)` — [`L153`](../../../../../raw/code/torchax/test/test_view.py#L153)
  - `test_log_normal_(self)` — [`L345`](../../../../../raw/code/torchax/test/test_view.py#L345)
  - `test_logical_not_(self)` — [`L314`](../../../../../raw/code/torchax/test/test_view.py#L314)
  - `test_lt_(self)` — [`L144`](../../../../../raw/code/torchax/test/test_view.py#L144)
  - `test_mul_(self)` — [`L115`](../../../../../raw/code/torchax/test/test_view.py#L115)
  - `test_multiply_tensor_and_view(self)` — [`L62`](../../../../../raw/code/torchax/test/test_view.py#L62)
  - `test_multiply_views(self)` — [`L73`](../../../../../raw/code/torchax/test/test_view.py#L73)
  - `test_ne_(self)` — [`L182`](../../../../../raw/code/torchax/test/test_view.py#L182)
  - `test_normal_(self)` — [`L215`](../../../../../raw/code/torchax/test/test_view.py#L215)
  - `test_outofplace_add(self)` — [`L51`](../../../../../raw/code/torchax/test/test_view.py#L51)
  - `test_pow_(self)` — [`L129`](../../../../../raw/code/torchax/test/test_view.py#L129)
  - `test_relu_(self)` — [`L236`](../../../../../raw/code/torchax/test/test_view.py#L236)
  - `test_scatter_(self)` — [`L371`](../../../../../raw/code/torchax/test/test_view.py#L371)
  - `test_scatter_add_(self)` — [`L356`](../../../../../raw/code/torchax/test/test_view.py#L356)
  - `test_scatter_reduce_(self)` — [`L388`](../../../../../raw/code/torchax/test/test_view.py#L388)
  - `test_setitem(self)` — [`L86`](../../../../../raw/code/torchax/test/test_view.py#L86)
  - `test_setitem2(self)` — [`L93`](../../../../../raw/code/torchax/test/test_view.py#L93)
  - `test_sigmoid_(self)` — [`L279`](../../../../../raw/code/torchax/test/test_view.py#L279)
  - `test_sqrt_(self)` — [`L257`](../../../../../raw/code/torchax/test/test_view.py#L257)
  - `test_squeeze_(self)` — [`L247`](../../../../../raw/code/torchax/test/test_view.py#L247)
  - `test_sub_(self)` — [`L108`](../../../../../raw/code/torchax/test/test_view.py#L108)
  - `test_tanh_(self)` — [`L291`](../../../../../raw/code/torchax/test/test_view.py#L291)
  - `test_transivity(self)` — [`L40`](../../../../../raw/code/torchax/test/test_view.py#L40)
  - `test_transpose_(self)` — [`L335`](../../../../../raw/code/torchax/test/test_view.py#L335)
  - `test_uniform_(self)` — [`L224`](../../../../../raw/code/torchax/test/test_view.py#L224)
  - `test_unsqueeze_(self)` — [`L325`](../../../../../raw/code/torchax/test/test_view.py#L325)
- uses (calls/refs, reference-scoped): [`Tensor`](../torchax/tensor.md#Tensor), [`View`](../torchax/view.md#View), [`enable_globally`](../torchax/__init__.md#enable_globally)

