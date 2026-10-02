---
title: 'Module: tpu_inference/models/vllm/dflash.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/dflash.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.dflash`/
symbols:
  logger: logger.
  _DFlashRunner: _DFlashRunner#
  _DFlashRunner.__init__: _DFlashRunner#__init__().
  _DFlashRunner.forward: _DFlashRunner#forward().
  _DFlashRunner._draft_forward: _DFlashRunner#_draft_forward().
  _DFlashRunner._combine_hidden: _DFlashRunner#_combine_hidden().
  _DFlashRunner._compute_logits: _DFlashRunner#_compute_logits().
  DFlashTorchaxWrapper: DFlashTorchaxWrapper#
  DFlashTorchaxWrapper.__init__: DFlashTorchaxWrapper#__init__().
  DFlashTorchaxWrapper.load: DFlashTorchaxWrapper#load().
  DFlashTorchaxWrapper.get_draft_forward_fn: DFlashTorchaxWrapper#get_draft_forward_fn().
  DFlashTorchaxWrapper.get_combine_hidden_fn: DFlashTorchaxWrapper#get_combine_hidden_fn().
  DFlashTorchaxWrapper.get_compute_logits_fn: DFlashTorchaxWrapper#get_compute_logits_fn().
---
# Module: [`tpu_inference/models/vllm/dflash.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py)

## Classes
### `DFlashTorchaxWrapper`
- def: [`tpu_inference/models/vllm/dflash.py:105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L105)
- doc: Load the HF DFlash model on CPU, shard to TPU, expose JIT-compiled
- signature: `class DFlashTorchaxWrapper:`
- members:
  - `get_combine_hidden_fn(self)` — [`L220`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L220) — Return a JIT-compiled combine_hidden_states function.
  - `get_compute_logits_fn(self)` — [`L250`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L250) — Return a JIT-compiled compute_logits function.
  - `get_draft_forward_fn(self)` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L169) — Return a JIT-compiled draft forward function.
  - `load(self, draft_model_path: str, target_model_state: Any)` — [`L115`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L115) — Load HF DFlash model, shard weights to TPU, share embeddings.
- protocol/private: `__init__`[`L109`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L109)

### `_DFlashRunner`
- def: [`tpu_inference/models/vllm/dflash.py:40`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L40)
- doc: Wrapper that adapts the HF DFlash model for ``functional_call``.
- signature: `class _DFlashRunner(torch.nn.Module):`
- members:
  - `_combine_hidden(self, raw_hidden: torch.Tensor)` — [`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L92) — Project concatenated target hidden states through fc + norm.
  - `_compute_logits(hidden_state: torch.Tensor, embed_weight: torch.Tensor)` — [`L97`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L97) — Logits via tied embeddings: hidden @ embed^T.
  - `_draft_forward(self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None=None)` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L61) — Run the DFlash model (no KV cache, no causal mask).
  - `forward(self, **kwargs)` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L47)
- protocol/private: `__init__`[`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L43)

## Module values
- `logger` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/dflash.py#L37)

