---
title: 'Module: llama_ref/model_with_scan.py'
type: catalog
provenance: extracted
module: llama_ref/model_with_scan.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.model_with_scan`/
symbols:
  Attention.forward: Attention#forward().
  ModelArgs.dim: ModelArgs#dim.
  Attention.head_dim: Attention#head_dim.
  ModelArgs.n_heads: ModelArgs#n_heads.
  TransformerBlock.feed_forward: TransformerBlock#feed_forward.
  TransformerBlock.forward: TransformerBlock#forward().
  ScanLayer.forward: ScanLayer#forward().
  Transformer.forward: Transformer#forward().
  Attention.wq: Attention#wq.
  Attention.wk: Attention#wk.
  Attention.wv: Attention#wv.
  Attention.wo: Attention#wo.
  TransformerBlock.attention_norm: TransformerBlock#attention_norm.
  TransformerBlock.ffn_norm: TransformerBlock#ffn_norm.
  Transformer.layers: Transformer#layers.
  Transformer.norm: Transformer#norm.
  Attention.n_kv_heads: Attention#n_kv_heads.
  ModelArgs.__init__: ModelArgs#__init__().
  FeedForward.forward: FeedForward#forward().
  Attention.n_rep: Attention#n_rep.
  ModelArgs.n_kv_heads: ModelArgs#n_kv_heads.
  Transformer.tok_embeddings: Transformer#tok_embeddings.
  Transformer.output: Transformer#output.
  Attention.n_local_kv_heads: Attention#n_local_kv_heads.
  RMSNorm.forward: RMSNorm#forward().
  TransformerBlock.head_dim: TransformerBlock#head_dim.
  Attention.n_local_heads: Attention#n_local_heads.
  with_sharding_constraint: with_sharding_constraint.
  RMSNorm._norm: RMSNorm#_norm().
  TransformerBlock.attention: TransformerBlock#attention.
  ScanLayer.layer_weights_keys: ScanLayer#layer_weights_keys.
  ScanLayer.params: ScanLayer#params.
  ScanLayer.eval_one_layer: ScanLayer#eval_one_layer().
  ModelArgs: ModelArgs#
  ModelArgs.vocab_size: ModelArgs#vocab_size.
  ModelArgs.norm_eps: ModelArgs#norm_eps.
  RMSNorm: RMSNorm#
  precompute_freqs_cis: precompute_freqs_cis().
  apply_rotary_emb: apply_rotary_emb().
  Attention.__init__: Attention#__init__().
  TransformerBlock.__init__: TransformerBlock#__init__().
  TransformerBlock.n_heads: TransformerBlock#n_heads.
  TransformerBlock.dim: TransformerBlock#dim.
  ScanLayer.num_layers: ScanLayer#num_layers.
  Transformer.__init__: Transformer#__init__().
  Transformer.vocab_size: Transformer#vocab_size.
  Transformer.n_layers: Transformer#n_layers.
  ModelArgs.n_layers: ModelArgs#n_layers.
  repeat_kv: repeat_kv().
  ScanLayer.m: ScanLayer#m.
  ScanLayer._param_name_new: ScanLayer#_param_name_new().
  ModelArgs.multiple_of: ModelArgs#multiple_of.
  ModelArgs.ffn_dim_multiplier: ModelArgs#ffn_dim_multiplier.
  RMSNorm.eps: RMSNorm#eps.
  RMSNorm.weight: RMSNorm#weight.
  apply_scaling: apply_scaling().
  reshape_for_broadcast: reshape_for_broadcast().
  Attention: Attention#
  FeedForward: FeedForward#
  FeedForward.w1: FeedForward#w1.
  FeedForward.w2: FeedForward#w2.
  FeedForward.w3: FeedForward#w3.
  TransformerBlock: TransformerBlock#
  ScanLayer: ScanLayer#
  ScanLayer._stack_layer_weights: ScanLayer#_stack_layer_weights().
  Transformer: Transformer#
  ModelArgs.rope_theta: ModelArgs#rope_theta.
  ModelArgs.use_scaled_rope: ModelArgs#use_scaled_rope.
  ModelArgs.max_batch_size: ModelArgs#max_batch_size.
  ModelArgs.max_seq_len: ModelArgs#max_seq_len.
  ModelArgs.vision_chunk_size: ModelArgs#vision_chunk_size.
  ModelArgs.vision_max_num_chunks: ModelArgs#vision_max_num_chunks.
  ModelArgs.vision_num_cross_attention_layers: ModelArgs#vision_num_cross_attention_layers.
  transformer_configs: transformer_configs.
  RMSNorm.__init__: RMSNorm#__init__().
  FeedForward.__init__: FeedForward#__init__().
  TransformerBlock.layer_id: TransformerBlock#layer_id.
  ScanLayer.__init__: ScanLayer#__init__().
  ScanLayer._param_name_old: ScanLayer#_param_name_old().
  Transformer.params: Transformer#params.
---
# Module: [`llama_ref/model_with_scan.py`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py)

## Classes
### `Attention`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:189`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L189)
- signature: `class Attention(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L219`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L219) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `head_dim` — [`L196`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L196) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `n_kv_heads` — [`L192`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L192)
  - `n_local_heads` — [`L193`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L193) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `n_local_kv_heads` — [`L194`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L194) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `n_rep` — [`L195`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L195) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `wk` — [`L203`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L203) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `wo` — [`L213`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L213) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `wq` — [`L198`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L198) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `wv` — [`L208`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L208) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
- protocol/private: `__init__`[`L190`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L190)
- uses (calls/refs, reference-scoped): [`dim`](model_with_scan.md#ModelArgs.dim), [`n_heads`](model_with_scan.md#ModelArgs.n_heads), [`n_kv_heads`](model_with_scan.md#ModelArgs.n_kv_heads), [`with_sharding_constraint`](model_with_scan.md#with_sharding_constraint), [`ModelArgs`](model_with_scan.md#ModelArgs), [`repeat_kv`](model_with_scan.md#repeat_kv)
- used by: [`attention`](model_with_scan.md#TransformerBlock.attention)

### `FeedForward`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:271`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L271)
- signature: `class FeedForward(nn.Module):`
- members:
  - `forward(self, x)` — [`L296`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L296)
  - `w1` — [`L286`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L286)
  - `w2` — [`L289`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L289)
  - `w3` — [`L292`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L292)
- protocol/private: `__init__`[`L272`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L272)
- used by: [`feed_forward`](model_with_scan.md#TransformerBlock.feed_forward)

### `ModelArgs`
- def: [`llama_ref/model_with_scan.py:30`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L30)
- signature: `class ModelArgs:`
- members:
  - `dim` — [`L31`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L31)
  - `ffn_dim_multiplier` — [`L37`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L37)
  - `max_batch_size` — [`L42`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L42)
  - `max_seq_len` — [`L43`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L43)
  - `multiple_of` — [`L36`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L36)
  - `n_heads` — [`L33`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L33)
  - `n_kv_heads` — [`L34`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L34)
  - `n_layers` — [`L32`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L32)
  - `norm_eps` — [`L38`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L38)
  - `rope_theta` — [`L39`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L39)
  - `use_scaled_rope` — [`L40`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L40)
  - `vision_chunk_size` — [`L46`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L46)
  - `vision_max_num_chunks` — [`L47`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L47)
  - `vision_num_cross_attention_layers` — [`L48`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L48)
  - `vocab_size` — [`L35`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L35)
- protocol/private: `__init__`[`L50`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L50)
- used by: [`head_dim`](model_with_scan.md#Attention.head_dim), [`feed_forward`](model_with_scan.md#TransformerBlock.feed_forward), [`attention_norm`](model_with_scan.md#TransformerBlock.attention_norm), [`ffn_norm`](model_with_scan.md#TransformerBlock.ffn_norm), [`layers`](model_with_scan.md#Transformer.layers), [`norm`](model_with_scan.md#Transformer.norm), [`wk`](model_with_scan.md#Attention.wk), [`wo`](model_with_scan.md#Attention.wo), [`wq`](model_with_scan.md#Attention.wq), [`wv`](model_with_scan.md#Attention.wv), [`n_kv_heads`](model_with_scan.md#Attention.n_kv_heads), [`output`](model_with_scan.md#Transformer.output), [`tok_embeddings`](model_with_scan.md#Transformer.tok_embeddings), [`head_dim`](model_with_scan.md#TransformerBlock.head_dim), [`n_local_heads`](model_with_scan.md#Attention.n_local_heads), [`__init__`](model_with_scan.md#Attention.__init__), [`__init__`](model_with_scan.md#Transformer.__init__), [`__init__`](model_with_scan.md#TransformerBlock.__init__), [`dim`](model_with_scan.md#TransformerBlock.dim), [`n_heads`](model_with_scan.md#TransformerBlock.n_heads), [`n_layers`](model_with_scan.md#Transformer.n_layers), [`vocab_size`](model_with_scan.md#Transformer.vocab_size)

### `RMSNorm`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:105`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L105)
- signature: `class RMSNorm(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L114`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L114)
  - `eps` — [`L108`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L108)
  - `weight` — [`L109`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L109)
- protocol/private: `__init__`[`L106`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L106), `_norm`[`L111`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L111)
- used by: [`attention_norm`](model_with_scan.md#TransformerBlock.attention_norm), [`ffn_norm`](model_with_scan.md#TransformerBlock.ffn_norm), [`norm`](model_with_scan.md#Transformer.norm)

### `ScanLayer`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:336`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L336)
- signature: `class ScanLayer(nn.Module):`
- members:
  - `eval_one_layer(args, weight)` — [`L387`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L387)
  - `forward(self, *args, **kwargs)` — [`L370`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L370)
  - `layer_weights_keys` — [`L344`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L344)
  - `m` — [`L341`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L341)
  - `num_layers` — [`L342`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L342)
  - `params` — [`L348`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L348)
- protocol/private: `__init__`[`L339`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L339), `_param_name_new`[`L364`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L364), `_param_name_old`[`L367`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L367), `_stack_layer_weights`[`L353`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L353)
- used by: [`layers`](model_with_scan.md#Transformer.layers)

### `Transformer`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:410`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L410)
- signature: `class Transformer(nn.Module):`
- members:
  - `forward(self, tokens: torch.Tensor, start_pos: int, freqs_cis, mask)` — [`L431`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L431)
  - `layers` — [`L424`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L424)
  - `n_layers` — [`L415`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L415)
  - `norm` — [`L426`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L426) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `output` — [`L427`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L427) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `params` — [`L413`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L413)
  - `tok_embeddings` — [`L417`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L417) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
  - `vocab_size` — [`L414`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L414)
- protocol/private: `__init__`[`L411`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L411)
- uses (calls/refs, reference-scoped): [`dim`](model_with_scan.md#ModelArgs.dim), [`ModelArgs`](model_with_scan.md#ModelArgs), [`RMSNorm`](model_with_scan.md#RMSNorm), [`norm_eps`](model_with_scan.md#ModelArgs.norm_eps), [`vocab_size`](model_with_scan.md#ModelArgs.vocab_size), [`n_layers`](model_with_scan.md#ModelArgs.n_layers), [`ScanLayer`](model_with_scan.md#ScanLayer), [`TransformerBlock`](model_with_scan.md#TransformerBlock)
- used by: [`main`](run.md#main)

### `TransformerBlock`  ·  implements/extends Module
- def: [`llama_ref/model_with_scan.py:306`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L306)
- signature: `class TransformerBlock(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L323`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L323)
  - `attention` — [`L312`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L312)
  - `attention_norm` — [`L320`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L320)
  - `dim` — [`L310`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L310)
  - `feed_forward` — [`L313`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L313)
  - `ffn_norm` — [`L321`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L321)
  - `head_dim` — [`L311`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L311)
  - `layer_id` — [`L319`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L319)
  - `n_heads` — [`L309`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L309)
- protocol/private: `__init__`[`L307`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L307)
- uses (calls/refs, reference-scoped): [`dim`](model_with_scan.md#ModelArgs.dim), [`n_heads`](model_with_scan.md#ModelArgs.n_heads), [`ModelArgs`](model_with_scan.md#ModelArgs), [`RMSNorm`](model_with_scan.md#RMSNorm), [`norm_eps`](model_with_scan.md#ModelArgs.norm_eps), [`Attention`](model_with_scan.md#Attention), [`FeedForward`](model_with_scan.md#FeedForward), [`ffn_dim_multiplier`](model_with_scan.md#ModelArgs.ffn_dim_multiplier), [`multiple_of`](model_with_scan.md#ModelArgs.multiple_of)
- used by: [`layers`](model_with_scan.md#Transformer.layers)

## Functions
- `apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)` — [`L164`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L164)
- `apply_scaling(freqs: torch.Tensor)` — [`L119`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L119)
- `precompute_freqs_cis(dim: int, end: int, theta: float = 10000, use_scaled: bool = False)` — [`L144`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L144)
- `repeat_kv(x: torch.Tensor, n_rep: int)` — [`L177`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L177) — torch.repeat_interleave(x, dim=2, repeats=n_rep) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)
- `reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor)` — [`L156`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L156)

## Module values
- `transformer_configs` — [`L65`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L65)
- `with_sharding_constraint` — [`L27`](../../../../../raw/code/learning-machine/llama_ref/model_with_scan.py#L27) — documented in [llama_ref-model_with_scan](../../concepts/llama_ref-model_with_scan.md)

