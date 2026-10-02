---
title: 'Module: tpu_inference/layers/vllm/custom_ops/embedding.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/embedding.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.embedding`/Vllm
symbols:
  VllmVocabParallelEmbedding: VocabParallelEmbedding#
  VllmVocabParallelEmbedding.__init__: VocabParallelEmbedding#__init__().
  VllmVocabParallelEmbedding.forward: VocabParallelEmbedding#forward().
  VllmParallelLMHead: ParallelLMHead#
  VllmParallelLMHead.__init__: ParallelLMHead#__init__().
  VllmParallelLMHead.forward: ParallelLMHead#forward().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/embedding.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py)

## Classes
### `VllmParallelLMHead`
- def: [`tpu_inference/layers/vllm/custom_ops/embedding.py:42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L42)
- signature: `class VllmParallelLMHead(ParallelLMHead):`
- members:
  - `forward(self, input_)` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L55)
- protocol/private: `__init__`[`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L44)

### `VllmVocabParallelEmbedding`
- def: [`tpu_inference/layers/vllm/custom_ops/embedding.py:24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L24)
- signature: `class VllmVocabParallelEmbedding(VocabParallelEmbedding):`
- members:
  - `forward(self, input_)` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L37)
- protocol/private: `__init__`[`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/embedding.py#L26)

