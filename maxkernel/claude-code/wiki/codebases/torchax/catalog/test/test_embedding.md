---
title: 'Module: test/test_embedding.py'
type: catalog
provenance: extracted
module: test/test_embedding.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_embedding`/
symbols:
  TestEmbeddingPaddingIdx.run_train: TestEmbeddingPaddingIdx#run_train().
  make_train_step: make_train_step().
  TestEmbeddingPaddingIdx.test_embedding_grad_with_padding_idx_jitted: TestEmbeddingPaddingIdx#test_embedding_grad_with_padding_idx_jitted().
  TestEmbeddingPaddingIdx.env: TestEmbeddingPaddingIdx#env.
  make_train_step.step: make_train_step().step().
  EmbeddingTestModule.forward: EmbeddingTestModule#forward().
  TestEmbeddingPaddingIdx.setUp: TestEmbeddingPaddingIdx#setUp().
  EmbeddingTestModule: EmbeddingTestModule#
  EmbeddingTestModule.embedding: EmbeddingTestModule#embedding.
  make_train_step.loss_and_aux: make_train_step().loss_and_aux().
  EmbeddingTestModule.__init__: EmbeddingTestModule#__init__().
  TestEmbeddingPaddingIdx: TestEmbeddingPaddingIdx#
---
# Module: [`test/test_embedding.py`](../../../../../raw/code/torchax/test/test_embedding.py)

## Classes
### `EmbeddingTestModule`  ·  implements/extends Module
- def: [`test/test_embedding.py:13`](../../../../../raw/code/torchax/test/test_embedding.py#L13)
- signature: `class EmbeddingTestModule(nn.Module):`
- members:
  - `forward(self, x)` — [`L22`](../../../../../raw/code/torchax/test/test_embedding.py#L22)
  - `embedding` — [`L16`](../../../../../raw/code/torchax/test/test_embedding.py#L16)
- protocol/private: `__init__`[`L14`](../../../../../raw/code/torchax/test/test_embedding.py#L14)
- used by: (1 test-only callers)

### `TestEmbeddingPaddingIdx`  ·  implements/extends TestCase
- def: [`test/test_embedding.py:28`](../../../../../raw/code/torchax/test/test_embedding.py#L28)
- signature: `class TestEmbeddingPaddingIdx(unittest.TestCase):`
- members:
  - `run_train(model)` — [`L51`](../../../../../raw/code/torchax/test/test_embedding.py#L51)
  - `setUp(self)` — [`L29`](../../../../../raw/code/torchax/test/test_embedding.py#L29)
  - `test_embedding_grad_with_padding_idx_jitted(self)` — [`L33`](../../../../../raw/code/torchax/test/test_embedding.py#L33) — Tests that the gradient for the padding_idx in an embedding layer is zero
  - `env` — [`L30`](../../../../../raw/code/torchax/test/test_embedding.py#L30)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`call_jax`](../torchax/interop.md#call_jax), [`jax_jit`](../torchax/interop.md#jax_jit), [`to_xla`](../torchax/tensor.md#Environment.to_xla)  (2 test-only)

## Functions
- `loss_and_aux(weights, buffers, *args)` — [`L139`](../../../../../raw/code/torchax/test/test_embedding.py#L139)
- `make_train_step(model_fn, optax_optimizer)` — [`L130`](../../../../../raw/code/torchax/test/test_embedding.py#L130) — A slightly modified version of torchax.train.make_train_step()
- `step(weights, buffers, opt_state, *args)` — [`L146`](../../../../../raw/code/torchax/test/test_embedding.py#L146)

