---
title: 'Module: llama_ref/model_with_collectives.py'
type: catalog
provenance: extracted
module: llama_ref/model_with_collectives.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.model_with_collectives`/
symbols:
  Attention.forward: Attention#forward().
  ModelArgs.dim: ModelArgs#dim.
  ScanLayer.forward: ScanLayer#forward().
  Transformer.forward: Transformer#forward().
  Attention.head_dim: Attention#head_dim.
  ModelArgs.n_heads: ModelArgs#n_heads.
  TransformerBlock.feed_forward: TransformerBlock#feed_forward.
  FeedForward.forward: FeedForward#forward().
  TransformerBlock.forward: TransformerBlock#forward().
  Attention.wq: Attention#wq.
  Attention.wk: Attention#wk.
  Attention.wv: Attention#wv.
  Attention.wo: Attention#wo.
  TransformerBlock.attention_norm: TransformerBlock#attention_norm.
  TransformerBlock.ffn_norm: TransformerBlock#ffn_norm.
  ScanLayer.gather_and_call: ScanLayer#gather_and_call().
  Transformer.layers: Transformer#layers.
  Transformer.norm: Transformer#norm.
  Attention.n_kv_heads: Attention#n_kv_heads.
  ModelArgs.__init__: ModelArgs#__init__().
  Attention.n_rep: Attention#n_rep.
  ModelArgs.n_kv_heads: ModelArgs#n_kv_heads.
  ScanLayer.eval_one_layer: ScanLayer#eval_one_layer().
  Transformer.tok_embeddings: Transformer#tok_embeddings.
  Transformer.output: Transformer#output.
  Attention.n_local_kv_heads: Attention#n_local_kv_heads.
  ScanLayer.unroll_layers: ScanLayer#unroll_layers.
  RMSNorm.forward: RMSNorm#forward().
  TransformerBlock.head_dim: TransformerBlock#head_dim.
  Attention.n_local_heads: Attention#n_local_heads.
  RMSNorm._norm: RMSNorm#_norm().
  apply_rotary_emb: apply_rotary_emb().
  TransformerBlock.attention: TransformerBlock#attention.
  ScanLayer.layer_weights_keys: ScanLayer#layer_weights_keys.
  ScanLayer.params: ScanLayer#params.
  all_gather: all_gather.
  ModelArgs: ModelArgs#
  ModelArgs.vocab_size: ModelArgs#vocab_size.
  ModelArgs.norm_eps: ModelArgs#norm_eps.
  RMSNorm: RMSNorm#
  precompute_freqs_cis: precompute_freqs_cis().
  Attention.__init__: Attention#__init__().
  TransformerBlock.__init__: TransformerBlock#__init__().
  TransformerBlock.n_heads: TransformerBlock#n_heads.
  TransformerBlock.dim: TransformerBlock#dim.
  Transformer.__init__: Transformer#__init__().
  Transformer.vocab_size: Transformer#vocab_size.
  Transformer.n_layers: Transformer#n_layers.
  all_reduce_sum: all_reduce_sum.
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
  _fsdp_axis: _fsdp_axis().
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
  ScanLayer.num_layers: ScanLayer#num_layers.
  ScanLayer._param_name_old: ScanLayer#_param_name_old().
  Transformer.params: Transformer#params.
---
# Module: [`llama_ref/model_with_collectives.py`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py)

## Classes
### `Attention`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:191`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L191)
- signature: `class Attention(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L221`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L221) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `head_dim` — [`L198`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L198)
  - `n_kv_heads` — [`L194`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L194)
  - `n_local_heads` — [`L195`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L195)
  - `n_local_kv_heads` — [`L196`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L196)
  - `n_rep` — [`L197`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L197)
  - `wk` — [`L205`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L205)
  - `wo` — [`L215`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L215)
  - `wq` — [`L200`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L200)
  - `wv` — [`L210`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L210)
- protocol/private: `__init__`[`L192`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L192)
- uses (calls/refs, reference-scoped): [`dim`](model_with_collectives.md#ModelArgs.dim), [`n_heads`](model_with_collectives.md#ModelArgs.n_heads), [`n_kv_heads`](model_with_collectives.md#ModelArgs.n_kv_heads), [`apply_rotary_emb`](model_with_collectives.md#apply_rotary_emb), [`ModelArgs`](model_with_collectives.md#ModelArgs), [`all_reduce_sum`](model_with_collectives.md#all_reduce_sum), [`repeat_kv`](model_with_collectives.md#repeat_kv)
- used by: [`attention`](model_with_collectives.md#TransformerBlock.attention)

### `FeedForward`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:269`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L269)
- signature: `class FeedForward(nn.Module):`
- members:
  - `forward(self, x)` — [`L294`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L294)
  - `w1` — [`L284`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L284)
  - `w2` — [`L287`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L287)
  - `w3` — [`L290`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L290)
- protocol/private: `__init__`[`L270`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L270)
- uses (calls/refs, reference-scoped): [`all_reduce_sum`](model_with_collectives.md#all_reduce_sum)
- used by: [`feed_forward`](model_with_collectives.md#TransformerBlock.feed_forward)

### `ModelArgs`
- def: [`llama_ref/model_with_collectives.py:32`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L32)
- signature: `class ModelArgs:`
- members:
  - `dim` — [`L33`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L33)
  - `ffn_dim_multiplier` — [`L39`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L39)
  - `max_batch_size` — [`L44`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L44)
  - `max_seq_len` — [`L45`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L45)
  - `multiple_of` — [`L38`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L38)
  - `n_heads` — [`L35`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L35)
  - `n_kv_heads` — [`L36`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L36)
  - `n_layers` — [`L34`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L34)
  - `norm_eps` — [`L40`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L40)
  - `rope_theta` — [`L41`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L41)
  - `use_scaled_rope` — [`L42`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L42)
  - `vision_chunk_size` — [`L48`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L48)
  - `vision_max_num_chunks` — [`L49`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L49)
  - `vision_num_cross_attention_layers` — [`L50`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L50)
  - `vocab_size` — [`L37`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L37)
- protocol/private: `__init__`[`L52`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L52)
- used by: [`head_dim`](model_with_collectives.md#Attention.head_dim), [`feed_forward`](model_with_collectives.md#TransformerBlock.feed_forward), [`attention_norm`](model_with_collectives.md#TransformerBlock.attention_norm), [`ffn_norm`](model_with_collectives.md#TransformerBlock.ffn_norm), [`layers`](model_with_collectives.md#Transformer.layers), [`norm`](model_with_collectives.md#Transformer.norm), [`wk`](model_with_collectives.md#Attention.wk), [`wo`](model_with_collectives.md#Attention.wo), [`wq`](model_with_collectives.md#Attention.wq), [`wv`](model_with_collectives.md#Attention.wv), [`n_kv_heads`](model_with_collectives.md#Attention.n_kv_heads), [`output`](model_with_collectives.md#Transformer.output), [`tok_embeddings`](model_with_collectives.md#Transformer.tok_embeddings), [`head_dim`](model_with_collectives.md#TransformerBlock.head_dim), [`n_local_heads`](model_with_collectives.md#Attention.n_local_heads), [`__init__`](model_with_collectives.md#Attention.__init__), [`__init__`](model_with_collectives.md#Transformer.__init__), [`__init__`](model_with_collectives.md#TransformerBlock.__init__), [`dim`](model_with_collectives.md#TransformerBlock.dim), [`n_heads`](model_with_collectives.md#TransformerBlock.n_heads), [`n_layers`](model_with_collectives.md#Transformer.n_layers), [`vocab_size`](model_with_collectives.md#Transformer.vocab_size)

### `RMSNorm`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:107`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L107)
- signature: `class RMSNorm(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L116`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L116)
  - `eps` — [`L110`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L110)
  - `weight` — [`L111`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L111)
- protocol/private: `__init__`[`L108`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L108), `_norm`[`L113`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L113)
- used by: [`attention_norm`](model_with_collectives.md#TransformerBlock.attention_norm), [`ffn_norm`](model_with_collectives.md#TransformerBlock.ffn_norm), [`norm`](model_with_collectives.md#Transformer.norm)

### `ScanLayer`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:349`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L349)
- signature: `class ScanLayer(nn.Module):`
- members:
  - `eval_one_layer(h, weight)` — [`L393`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L393) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `forward(self, *args, **kwargs)` — [`L387`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L387) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `gather_and_call(h, new_weights)` — [`L395`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L395) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `layer_weights_keys` — [`L360`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L360) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `m` — [`L356`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L356)
  - `num_layers` — [`L357`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L357)
  - `params` — [`L364`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L364) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `unroll_layers` — [`L358`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L358) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
- protocol/private: `__init__`[`L354`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L354), `_param_name_new`[`L381`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L381), `_param_name_old`[`L384`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L384), `_stack_layer_weights`[`L370`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L370)
- uses (calls/refs, reference-scoped): [`all_gather`](model_with_collectives.md#all_gather), [`_fsdp_axis`](model_with_collectives.md#_fsdp_axis)
- used by: [`layers`](model_with_collectives.md#Transformer.layers)

### `Transformer`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:443`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L443)
- signature: `class Transformer(nn.Module):`
- members:
  - `forward(self, tokens: torch.Tensor, start_pos: int, freqs_cis, mask)` — [`L464`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L464) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `layers` — [`L457`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L457) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `n_layers` — [`L448`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L448)
  - `norm` — [`L459`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L459)
  - `output` — [`L460`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L460) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `params` — [`L446`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L446)
  - `tok_embeddings` — [`L450`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L450) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
  - `vocab_size` — [`L447`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L447)
- protocol/private: `__init__`[`L444`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L444)
- uses (calls/refs, reference-scoped): [`dim`](model_with_collectives.md#ModelArgs.dim), [`ModelArgs`](model_with_collectives.md#ModelArgs), [`RMSNorm`](model_with_collectives.md#RMSNorm), [`all_gather`](model_with_collectives.md#all_gather), [`norm_eps`](model_with_collectives.md#ModelArgs.norm_eps), [`vocab_size`](model_with_collectives.md#ModelArgs.vocab_size), [`n_layers`](model_with_collectives.md#ModelArgs.n_layers), [`ScanLayer`](model_with_collectives.md#ScanLayer), [`TransformerBlock`](model_with_collectives.md#TransformerBlock)
- used by: [`main`](run.md#main)

### `TransformerBlock`  ·  implements/extends Module
- def: [`llama_ref/model_with_collectives.py:305`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L305)
- signature: `class TransformerBlock(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L322`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L322)
  - `attention` — [`L311`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L311)
  - `attention_norm` — [`L319`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L319)
  - `dim` — [`L309`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L309)
  - `feed_forward` — [`L312`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L312)
  - `ffn_norm` — [`L320`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L320)
  - `head_dim` — [`L310`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L310)
  - `layer_id` — [`L318`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L318)
  - `n_heads` — [`L308`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L308)
- protocol/private: `__init__`[`L306`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L306)
- uses (calls/refs, reference-scoped): [`dim`](model_with_collectives.md#ModelArgs.dim), [`n_heads`](model_with_collectives.md#ModelArgs.n_heads), [`ModelArgs`](model_with_collectives.md#ModelArgs), [`RMSNorm`](model_with_collectives.md#RMSNorm), [`norm_eps`](model_with_collectives.md#ModelArgs.norm_eps), [`Attention`](model_with_collectives.md#Attention), [`FeedForward`](model_with_collectives.md#FeedForward), [`ffn_dim_multiplier`](model_with_collectives.md#ModelArgs.ffn_dim_multiplier), [`multiple_of`](model_with_collectives.md#ModelArgs.multiple_of)
- used by: [`layers`](model_with_collectives.md#Transformer.layers)

## Functions
- `_fsdp_axis(name)` — [`L342`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L342)
- `apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)` — [`L166`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L166) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
- `apply_scaling(freqs: torch.Tensor)` — [`L121`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L121)
- `precompute_freqs_cis(dim: int, end: int, theta: float = 10000, use_scaled: bool = False)` — [`L146`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L146)
- `repeat_kv(x: torch.Tensor, n_rep: int)` — [`L179`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L179) — torch.repeat_interleave(x, dim=2, repeats=n_rep)
- `reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor)` — [`L158`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L158) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)

## Module values
- `all_gather` — [`L27`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L27) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
- `all_reduce_sum` — [`L28`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L28) — documented in [llama_ref-model_with_collectives](../../concepts/llama_ref-model_with_collectives.md)
- `transformer_configs` — [`L67`](../../../../../raw/code/learning-machine/llama_ref/model_with_collectives.py#L67)

