---
title: 'Module: axlearn/common/flash_attention/neuron_attention_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/neuron_attention_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.neuron_attention_test`/NeuronAttentionTest#
symbols:
  NeuronAttentionTest: ''
  NeuronAttentionTest.setUp: setUp().
  NeuronAttentionTest.test_fwd_against_ref: test_fwd_against_ref().
  NeuronAttentionTest.test_bwd_against_ref: test_bwd_against_ref().
---
# Module: [`axlearn/common/flash_attention/neuron_attention_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention_test.py)

## Classes
### `NeuronAttentionTest`
- def: [`axlearn/common/flash_attention/neuron_attention_test.py:16`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention_test.py#L16)
- doc: Tests Neuron FlashAttention kernels.
- signature: `class NeuronAttentionTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L19`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention_test.py#L19)
  - `test_bwd_against_ref(self, batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, causal: bool, input_dtype: jnp.dtype, attention_bias_type: Literal[None, '2d'])` — [`L89`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention_test.py#L89)
  - `test_fwd_against_ref(self, batch_size: int, seq_len: int, num_heads: int, per_head_dim: int, causal: bool, input_dtype: jnp.dtype, attention_bias_type: Literal[None, '2d'])` — [`L37`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention_test.py#L37)

