---
title: 'Module: llama_ref/model.py'
type: catalog
provenance: extracted
module: llama_ref/model.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.model`/
symbols:
  Attention.forward: Attention#forward().
  ModelArgs.dim: ModelArgs#dim.
  ModelArgs.n_heads: ModelArgs#n_heads.
  Attention.head_dim: Attention#head_dim.
  TransformerBlock.feed_forward: TransformerBlock#feed_forward.
  TransformerBlock.forward: TransformerBlock#forward().
  Transformer.forward: Transformer#forward().
  Transformer.layers: Transformer#layers.
  Attention.wq: Attention#wq.
  Attention.wk: Attention#wk.
  Attention.wv: Attention#wv.
  Attention.wo: Attention#wo.
  TransformerBlock.attention_norm: TransformerBlock#attention_norm.
  TransformerBlock.ffn_norm: TransformerBlock#ffn_norm.
  Transformer.norm: Transformer#norm.
  Attention.n_kv_heads: Attention#n_kv_heads.
  ModelArgs.__init__: ModelArgs#__init__().
  FeedForward.forward: FeedForward#forward().
  Attention.n_rep: Attention#n_rep.
  ModelArgs.n_kv_heads: ModelArgs#n_kv_heads.
  Transformer.tok_embeddings: Transformer#tok_embeddings.
  Transformer.output: Transformer#output.
  Attention.n_local_kv_heads: Attention#n_local_kv_heads.
  ModelArgs: ModelArgs#
  RMSNorm.forward: RMSNorm#forward().
  TransformerBlock.head_dim: TransformerBlock#head_dim.
  Attention.n_local_heads: Attention#n_local_heads.
  RMSNorm._norm: RMSNorm#_norm().
  precompute_freqs_cis: precompute_freqs_cis().
  apply_rotary_emb: apply_rotary_emb().
  TransformerBlock.attention: TransformerBlock#attention.
  ModelArgs.n_layers: ModelArgs#n_layers.
  ModelArgs.vocab_size: ModelArgs#vocab_size.
  ModelArgs.norm_eps: ModelArgs#norm_eps.
  RMSNorm: RMSNorm#
  Attention.__init__: Attention#__init__().
  TransformerBlock.__init__: TransformerBlock#__init__().
  TransformerBlock.n_heads: TransformerBlock#n_heads.
  TransformerBlock.dim: TransformerBlock#dim.
  Transformer.__init__: Transformer#__init__().
  Transformer.vocab_size: Transformer#vocab_size.
  Transformer.n_layers: Transformer#n_layers.
  transformer_configs: transformer_configs.
  repeat_kv: repeat_kv().
  ModelArgs.multiple_of: ModelArgs#multiple_of.
  ModelArgs.ffn_dim_multiplier: ModelArgs#ffn_dim_multiplier.
  ModelArgs.rope_theta: ModelArgs#rope_theta.
  ModelArgs.use_scaled_rope: ModelArgs#use_scaled_rope.
  ModelArgs.max_seq_len: ModelArgs#max_seq_len.
  ModelArgs.tp_size: ModelArgs#tp_size.
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
  Transformer: Transformer#
  ModelArgs.max_batch_size: ModelArgs#max_batch_size.
  ModelArgs.vision_chunk_size: ModelArgs#vision_chunk_size.
  ModelArgs.vision_max_num_chunks: ModelArgs#vision_max_num_chunks.
  ModelArgs.vision_num_cross_attention_layers: ModelArgs#vision_num_cross_attention_layers.
  RMSNorm.__init__: RMSNorm#__init__().
  FeedForward.__init__: FeedForward#__init__().
  TransformerBlock.layer_id: TransformerBlock#layer_id.
  Transformer.params: Transformer#params.
---
# Module: [`llama_ref/model.py`](../../../../../raw/code/learning-machine/llama_ref/model.py)

## Classes
### `Attention`  ·  implements/extends Module
- def: [`llama_ref/model.py:183`](../../../../../raw/code/learning-machine/llama_ref/model.py#L183)
- signature: `class Attention(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L213`](../../../../../raw/code/learning-machine/llama_ref/model.py#L213) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `head_dim` — [`L190`](../../../../../raw/code/learning-machine/llama_ref/model.py#L190) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `n_kv_heads` — [`L186`](../../../../../raw/code/learning-machine/llama_ref/model.py#L186)
  - `n_local_heads` — [`L187`](../../../../../raw/code/learning-machine/llama_ref/model.py#L187)
  - `n_local_kv_heads` — [`L188`](../../../../../raw/code/learning-machine/llama_ref/model.py#L188)
  - `n_rep` — [`L189`](../../../../../raw/code/learning-machine/llama_ref/model.py#L189) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `wk` — [`L197`](../../../../../raw/code/learning-machine/llama_ref/model.py#L197) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `wo` — [`L207`](../../../../../raw/code/learning-machine/llama_ref/model.py#L207) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `wq` — [`L192`](../../../../../raw/code/learning-machine/llama_ref/model.py#L192) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `wv` — [`L202`](../../../../../raw/code/learning-machine/llama_ref/model.py#L202) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
- protocol/private: `__init__`[`L184`](../../../../../raw/code/learning-machine/llama_ref/model.py#L184)
- uses (calls/refs, reference-scoped): [`dim`](model.md#ModelArgs.dim), [`n_heads`](model.md#ModelArgs.n_heads), [`n_kv_heads`](model.md#ModelArgs.n_kv_heads), [`ModelArgs`](model.md#ModelArgs), [`apply_rotary_emb`](model.md#apply_rotary_emb), [`repeat_kv`](model.md#repeat_kv)
- used by: [`attention`](model.md#TransformerBlock.attention)

### `FeedForward`  ·  implements/extends Module
- def: [`llama_ref/model.py:254`](../../../../../raw/code/learning-machine/llama_ref/model.py#L254)
- signature: `class FeedForward(nn.Module):`
- members:
  - `forward(self, x)` — [`L279`](../../../../../raw/code/learning-machine/llama_ref/model.py#L279)
  - `w1` — [`L269`](../../../../../raw/code/learning-machine/llama_ref/model.py#L269)
  - `w2` — [`L272`](../../../../../raw/code/learning-machine/llama_ref/model.py#L272)
  - `w3` — [`L275`](../../../../../raw/code/learning-machine/llama_ref/model.py#L275)
- protocol/private: `__init__`[`L255`](../../../../../raw/code/learning-machine/llama_ref/model.py#L255)
- used by: [`feed_forward`](model.md#TransformerBlock.feed_forward)

### `ModelArgs`
- def: [`llama_ref/model.py:22`](../../../../../raw/code/learning-machine/llama_ref/model.py#L22)
- signature: `class ModelArgs:`
- members:
  - `dim` — [`L23`](../../../../../raw/code/learning-machine/llama_ref/model.py#L23) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `ffn_dim_multiplier` — [`L29`](../../../../../raw/code/learning-machine/llama_ref/model.py#L29)
  - `max_batch_size` — [`L34`](../../../../../raw/code/learning-machine/llama_ref/model.py#L34)
  - `max_seq_len` — [`L35`](../../../../../raw/code/learning-machine/llama_ref/model.py#L35)
  - `multiple_of` — [`L28`](../../../../../raw/code/learning-machine/llama_ref/model.py#L28)
  - `n_heads` — [`L25`](../../../../../raw/code/learning-machine/llama_ref/model.py#L25) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `n_kv_heads` — [`L26`](../../../../../raw/code/learning-machine/llama_ref/model.py#L26) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `n_layers` — [`L24`](../../../../../raw/code/learning-machine/llama_ref/model.py#L24)
  - `norm_eps` — [`L30`](../../../../../raw/code/learning-machine/llama_ref/model.py#L30)
  - `rope_theta` — [`L31`](../../../../../raw/code/learning-machine/llama_ref/model.py#L31)
  - `tp_size` — [`L36`](../../../../../raw/code/learning-machine/llama_ref/model.py#L36)
  - `use_scaled_rope` — [`L32`](../../../../../raw/code/learning-machine/llama_ref/model.py#L32)
  - `vision_chunk_size` — [`L39`](../../../../../raw/code/learning-machine/llama_ref/model.py#L39)
  - `vision_max_num_chunks` — [`L40`](../../../../../raw/code/learning-machine/llama_ref/model.py#L40)
  - `vision_num_cross_attention_layers` — [`L41`](../../../../../raw/code/learning-machine/llama_ref/model.py#L41)
  - `vocab_size` — [`L27`](../../../../../raw/code/learning-machine/llama_ref/model.py#L27)
- protocol/private: `__init__`[`L43`](../../../../../raw/code/learning-machine/llama_ref/model.py#L43)
- used by: [`main`](run.md#main), [`head_dim`](model.md#Attention.head_dim), [`feed_forward`](model.md#TransformerBlock.feed_forward), [`layers`](model.md#Transformer.layers), [`attention_norm`](model.md#TransformerBlock.attention_norm), [`ffn_norm`](model.md#TransformerBlock.ffn_norm), [`main2`](run.md#main2), [`norm`](model.md#Transformer.norm), [`wk`](model.md#Attention.wk), [`wo`](model.md#Attention.wo), [`wq`](model.md#Attention.wq), [`wv`](model.md#Attention.wv), [`n_kv_heads`](model.md#Attention.n_kv_heads), [`output`](model.md#Transformer.output), [`tok_embeddings`](model.md#Transformer.tok_embeddings), [`head_dim`](model.md#TransformerBlock.head_dim), [`n_local_heads`](model.md#Attention.n_local_heads), [`__init__`](model.md#Attention.__init__), [`__init__`](model.md#Transformer.__init__), [`__init__`](model.md#TransformerBlock.__init__), [`dim`](model.md#TransformerBlock.dim), [`n_heads`](model.md#TransformerBlock.n_heads), [`n_layers`](model.md#Transformer.n_layers), [`vocab_size`](model.md#Transformer.vocab_size)

### `RMSNorm`  ·  implements/extends Module
- def: [`llama_ref/model.py:98`](../../../../../raw/code/learning-machine/llama_ref/model.py#L98)
- signature: `class RMSNorm(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L107`](../../../../../raw/code/learning-machine/llama_ref/model.py#L107)
  - `eps` — [`L101`](../../../../../raw/code/learning-machine/llama_ref/model.py#L101)
  - `weight` — [`L102`](../../../../../raw/code/learning-machine/llama_ref/model.py#L102)
- protocol/private: `__init__`[`L99`](../../../../../raw/code/learning-machine/llama_ref/model.py#L99), `_norm`[`L104`](../../../../../raw/code/learning-machine/llama_ref/model.py#L104)
- used by: [`attention_norm`](model.md#TransformerBlock.attention_norm), [`ffn_norm`](model.md#TransformerBlock.ffn_norm), [`norm`](model.md#Transformer.norm)

### `Transformer`  ·  implements/extends Module
- def: [`llama_ref/model.py:312`](../../../../../raw/code/learning-machine/llama_ref/model.py#L312)
- signature: `class Transformer(nn.Module):`
- members:
  - `forward(self, tokens: torch.Tensor, start_pos: int, freqs_cis, mask)` — [`L333`](../../../../../raw/code/learning-machine/llama_ref/model.py#L333)
  - `layers` — [`L323`](../../../../../raw/code/learning-machine/llama_ref/model.py#L323)
  - `n_layers` — [`L317`](../../../../../raw/code/learning-machine/llama_ref/model.py#L317)
  - `norm` — [`L327`](../../../../../raw/code/learning-machine/llama_ref/model.py#L327) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `output` — [`L328`](../../../../../raw/code/learning-machine/llama_ref/model.py#L328) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `params` — [`L315`](../../../../../raw/code/learning-machine/llama_ref/model.py#L315)
  - `tok_embeddings` — [`L319`](../../../../../raw/code/learning-machine/llama_ref/model.py#L319) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `vocab_size` — [`L316`](../../../../../raw/code/learning-machine/llama_ref/model.py#L316)
- protocol/private: `__init__`[`L313`](../../../../../raw/code/learning-machine/llama_ref/model.py#L313)
- uses (calls/refs, reference-scoped): [`dim`](model.md#ModelArgs.dim), [`ModelArgs`](model.md#ModelArgs), [`RMSNorm`](model.md#RMSNorm), [`n_layers`](model.md#ModelArgs.n_layers), [`norm_eps`](model.md#ModelArgs.norm_eps), [`vocab_size`](model.md#ModelArgs.vocab_size), [`TransformerBlock`](model.md#TransformerBlock)
- used by: [`main`](run.md#main)

### `TransformerBlock`  ·  implements/extends Module
- def: [`llama_ref/model.py:283`](../../../../../raw/code/learning-machine/llama_ref/model.py#L283)
- signature: `class TransformerBlock(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L300`](../../../../../raw/code/learning-machine/llama_ref/model.py#L300)
  - `attention` — [`L289`](../../../../../raw/code/learning-machine/llama_ref/model.py#L289)
  - `attention_norm` — [`L297`](../../../../../raw/code/learning-machine/llama_ref/model.py#L297)
  - `dim` — [`L287`](../../../../../raw/code/learning-machine/llama_ref/model.py#L287) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `feed_forward` — [`L290`](../../../../../raw/code/learning-machine/llama_ref/model.py#L290) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
  - `ffn_norm` — [`L298`](../../../../../raw/code/learning-machine/llama_ref/model.py#L298)
  - `head_dim` — [`L288`](../../../../../raw/code/learning-machine/llama_ref/model.py#L288)
  - `layer_id` — [`L296`](../../../../../raw/code/learning-machine/llama_ref/model.py#L296)
  - `n_heads` — [`L286`](../../../../../raw/code/learning-machine/llama_ref/model.py#L286)
- protocol/private: `__init__`[`L284`](../../../../../raw/code/learning-machine/llama_ref/model.py#L284)
- uses (calls/refs, reference-scoped): [`dim`](model.md#ModelArgs.dim), [`n_heads`](model.md#ModelArgs.n_heads), [`ModelArgs`](model.md#ModelArgs), [`RMSNorm`](model.md#RMSNorm), [`norm_eps`](model.md#ModelArgs.norm_eps), [`Attention`](model.md#Attention), [`FeedForward`](model.md#FeedForward), [`ffn_dim_multiplier`](model.md#ModelArgs.ffn_dim_multiplier), [`multiple_of`](model.md#ModelArgs.multiple_of)
- used by: [`layers`](model.md#Transformer.layers)

## Functions
- `apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)` — [`L157`](../../../../../raw/code/learning-machine/llama_ref/model.py#L157)
- `apply_scaling(freqs: torch.Tensor)` — [`L112`](../../../../../raw/code/learning-machine/llama_ref/model.py#L112)
- `precompute_freqs_cis(dim: int, end: int, theta: float = 10000, use_scaled: bool = False)` — [`L137`](../../../../../raw/code/learning-machine/llama_ref/model.py#L137)
- `repeat_kv(x: torch.Tensor, n_rep: int)` — [`L171`](../../../../../raw/code/learning-machine/llama_ref/model.py#L171) — torch.repeat_interleave(x, dim=2, repeats=n_rep) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
- `reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor)` — [`L149`](../../../../../raw/code/learning-machine/llama_ref/model.py#L149)

## Module values
- `transformer_configs` — [`L58`](../../../../../raw/code/learning-machine/llama_ref/model.py#L58)

