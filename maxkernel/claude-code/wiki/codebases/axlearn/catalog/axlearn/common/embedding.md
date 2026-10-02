---
title: 'Module: axlearn/common/embedding.py'
type: catalog
provenance: extracted
module: axlearn/common/embedding.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.embedding`/
symbols:
  TransformerTextEmbeddings.__init__: TransformerTextEmbeddings#__init__().
  TransformerTextEmbeddings.forward: TransformerTextEmbeddings#forward().
  BaseEmbedding: BaseEmbedding#
  BaseEmbedding.Config: BaseEmbedding#Config#
  TransformerTextEmbeddings.Config: TransformerTextEmbeddings#Config#
  BaseEmbedding.Config.dim: BaseEmbedding#Config#dim.
  BaseEmbedding.forward: BaseEmbedding#forward().
  TransformerTextEmbeddings.Config.token_emb: TransformerTextEmbeddings#Config#token_emb.
  TransformerTextEmbeddings.Config.dropout: TransformerTextEmbeddings#Config#dropout.
  BaseEmbedding.init_states: BaseEmbedding#init_states().
  BaseEmbedding.extend_step: BaseEmbedding#extend_step().
  TransformerTextEmbeddings: TransformerTextEmbeddings#
  TransformerTextEmbeddings.attend: TransformerTextEmbeddings#attend().
  ModalityEmbedding.Config: ModalityEmbedding#Config#
  ModalityEmbedding.Config.modality_vocab_info: ModalityEmbedding#Config#modality_vocab_info.
  TransformerTextEmbeddings.Config.vocab_size: TransformerTextEmbeddings#Config#vocab_size.
  TransformerTextEmbeddings.Config.type_emb: TransformerTextEmbeddings#Config#type_emb.
  TransformerTextEmbeddings.Config.pos_emb: TransformerTextEmbeddings#Config#pos_emb.
  TransformerTextEmbeddings.Config.norm: TransformerTextEmbeddings#Config#norm.
  BaseEmbedding.attend: BaseEmbedding#attend().
  ModalityVocabInfo.vocab_size: ModalityVocabInfo#vocab_size().
  ModalityVocabInfo: ModalityVocabInfo#
  ModalityEmbedding: ModalityEmbedding#
  ModalityEmbedding.Output.ids: ModalityEmbedding#Output#ids.
  ModalityEmbedding.Output.embeddings: ModalityEmbedding#Output#embeddings.
  ModalityEmbedding.Output.paddings: ModalityEmbedding#Output#paddings.
  ModalityEmbedding.Output.batch_idx: ModalityEmbedding#Output#batch_idx.
  ModalityEmbedding.lookup_modality_embeddings: ModalityEmbedding#lookup_modality_embeddings().
  ModalityVocabInfo.placeholder_start: ModalityVocabInfo#placeholder_start.
  ModalityVocabInfo.placeholder_end: ModalityVocabInfo#placeholder_end.
  ModalityVocabInfo.vocab_start: ModalityVocabInfo#vocab_start.
  ModalityVocabInfo.vocab_end: ModalityVocabInfo#vocab_end.
  TransformerTextEmbeddings.Config.soft_cap_logits: TransformerTextEmbeddings#Config#soft_cap_logits.
  ModalityVocabInfo.modality_name: ModalityVocabInfo#modality_name.
  ModalityVocabInfo.generate_logits: ModalityVocabInfo#generate_logits.
  ModalityEmbedding.Config.in_partition_spec: ModalityEmbedding#Config#in_partition_spec.
  ModalityEmbedding.Config.out_partition_spec: ModalityEmbedding#Config#out_partition_spec.
  ModalityEmbedding.Output: ModalityEmbedding#Output#
---
# Module: [`axlearn/common/embedding.py`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py)

## Classes
### `BaseEmbedding`  ·  implements/extends BaseLayer
- def: [`axlearn/common/embedding.py:18`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L18)
- doc: The base class of a embedding layer.
- signature: `class BaseEmbedding(BaseLayer):`
- members:
  - `attend(self, x: Tensor)` — [`L42`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L42) — Computes logits with token embedding.
  - `extend_step(self, *, cached_states: Nested[Tensor], input_batch: Nested[Tensor], is_prefill: bool = False)` — [`L67`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L67) — Extends one step for streaming decode.
  - `forward(self, input_batch: Nested[Tensor])` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L31) — Computes embeddings.
  - `init_states(self, *, batch_size: int, dtype: jnp.dtype)` — [`L54`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L54) — Initializes state for streaming decode.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`BaseLayer`](base_layer.md#BaseLayer), [`nowrap`](module.md#nowrap), [`forward`](embedding.md#TransformerTextEmbeddings.forward), [`TransformerTextEmbeddings`](embedding.md#TransformerTextEmbeddings), [`attend`](embedding.md#TransformerTextEmbeddings.attend), [`ModalityEmbedding`](embedding.md#ModalityEmbedding)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`emb`](decoder.md#Decoder.Config.emb), [`Config`](embedding.md#TransformerTextEmbeddings.Config), [`Config`](embedding.md#ModalityEmbedding.Config), [`TransformerTextEmbeddings`](embedding.md#TransformerTextEmbeddings), [`ModalityEmbedding`](embedding.md#ModalityEmbedding)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/embedding.py:218`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L218)
- doc: Configures ModalityEmbedding.
- signature: `class Config(BaseEmbedding.Config):`
- members:
  - `dim` — [`L29`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L29)
  - `dropout` — [`L100`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L100)
  - `in_partition_spec` — [`L233`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L233)
  - `modality_vocab_info` — [`L232`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L232)
  - `norm` — [`L99`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L99)
  - `out_partition_spec` — [`L234`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L234)
  - `pos_emb` — [`L98`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L98)
  - `soft_cap_logits` — [`L103`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L103)
  - `token_emb` — [`L96`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L96)
  - `type_emb` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L97)
  - `vocab_size` — [`L95`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L95)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`Dropout`](layers.md#Dropout), [`BaseEmbedding`](embedding.md#BaseEmbedding), [`default_config`](layers.md#Embedding.default_config), [`Config`](embedding.md#BaseEmbedding.Config), [`Embedding`](layers.md#Embedding), [`ModalityVocabInfo`](embedding.md#ModalityVocabInfo)
- used by: [`__init__`](embedding.md#TransformerTextEmbeddings.__init__), [`forward`](embedding.md#TransformerTextEmbeddings.forward)

### `ModalityEmbedding`  ·  implements/extends BaseEmbedding
- def: [`axlearn/common/embedding.py:214`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L214)
- doc: The base class of a modality-specific embedding layer.
- signature: `class ModalityEmbedding(BaseEmbedding):`
- members:
  - `lookup_modality_embeddings(self, input_ids: Tensor, accum: Tensor)` — [`L262`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L262) — Looks up modality embeddings.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`BaseEmbedding`](embedding.md#BaseEmbedding)
- used by: [`BaseEmbedding`](embedding.md#BaseEmbedding)

### `ModalityVocabInfo`
- def: [`axlearn/common/embedding.py:174`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L174)
- doc: A modality-specific vocab info.
- signature: `class ModalityVocabInfo:`
- members:
  - `vocab_size(self)` — [`L210`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L210)
  - `generate_logits` — [`L207`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L207)
  - `modality_name` — [`L202`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L202)
  - `placeholder_end` — [`L204`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L204)
  - `placeholder_start` — [`L203`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L203)
  - `vocab_end` — [`L206`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L206)
  - `vocab_start` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L205)
- used by: [`forward`](loss_metrics.md#ModalityLossMetrics.forward), [`modality_vocab_info`](embedding.md#ModalityEmbedding.Config.modality_vocab_info), [`modality_vocab_info`](loss_metrics.md#ModalityLossMetrics.Config.modality_vocab_info)

### `Output`
- def: [`axlearn/common/embedding.py:237`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L237)
- doc: Modality-specific outputs.
- signature: `class Output:`
- members:
  - `batch_idx` — [`L260`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L260)
  - `embeddings` — [`L258`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L258)
  - `ids` — [`L257`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L257)
  - `paddings` — [`L259`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L259)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)

### `TransformerTextEmbeddings`  ·  implements/extends BaseEmbedding
- def: [`axlearn/common/embedding.py:88`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L88)
- doc: Textual embeddings from token id, position and token type embeddings.
- signature: `class TransformerTextEmbeddings(BaseEmbedding):`
- members:
  - `attend(self, x: Tensor)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L154) — Computes logits with token embedding.
  - `forward(self, input_batch: Nested[Tensor])` — [`L117`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L117) — Computes input embeddings with positional embeddings.
- protocol/private: `__init__`[`L105`](../../../../../../raw/code/axlearn/axlearn/common/embedding.py#L105)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`child_context`](module.md#child_context), [`validate_contains_paths`](utils.md#validate_contains_paths), [`BaseEmbedding`](embedding.md#BaseEmbedding), [`Config`](embedding.md#TransformerTextEmbeddings.Config), [`dim`](embedding.md#BaseEmbedding.Config.dim), [`dropout`](embedding.md#TransformerTextEmbeddings.Config.dropout), [`token_emb`](embedding.md#TransformerTextEmbeddings.Config.token_emb), [`vocab_size`](embedding.md#TransformerTextEmbeddings.Config.vocab_size), [`norm`](embedding.md#TransformerTextEmbeddings.Config.norm), [`pos_emb`](embedding.md#TransformerTextEmbeddings.Config.pos_emb), [`type_emb`](embedding.md#TransformerTextEmbeddings.Config.type_emb)
- used by: `_test_decoder_with_transformer`, [`BaseEmbedding`](embedding.md#BaseEmbedding), [`emb`](decoder.md#Decoder.Config.emb), [`forward`](embedding.md#BaseEmbedding.forward), [`attend`](embedding.md#BaseEmbedding.attend)

