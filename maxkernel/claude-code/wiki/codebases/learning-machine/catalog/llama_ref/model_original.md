---
title: 'Module: llama_ref/model_original.py'
type: catalog
provenance: extracted
module: llama_ref/model_original.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.model_original`/
symbols:
  Attention.forward: Attention#forward().
  Transformer.forward: Transformer#forward().
  Attention.head_dim: Attention#head_dim.
  TransformerBlock.forward: TransformerBlock#forward().
  Attention.cache_k: Attention#cache_k.
  Attention.cache_v: Attention#cache_v.
  Attention.n_local_kv_heads: Attention#n_local_kv_heads.
  FeedForward.forward: FeedForward#forward().
  Attention.n_rep: Attention#n_rep.
  Transformer.layers: Transformer#layers.
  Attention.wk: Attention#wk.
  Attention.wv: Attention#wv.
  Transformer.freqs_cis: Transformer#freqs_cis.
  RMSNorm.forward: RMSNorm#forward().
  RMSNorm._norm: RMSNorm#_norm().
  precompute_freqs_cis: precompute_freqs_cis().
  apply_rotary_emb: apply_rotary_emb().
  Attention.wq: Attention#wq.
  Attention.wo: Attention#wo.
  TransformerBlock.attention: TransformerBlock#attention.
  TransformerBlock.feed_forward: TransformerBlock#feed_forward.
  TransformerBlock.attention_norm: TransformerBlock#attention_norm.
  TransformerBlock.ffn_norm: TransformerBlock#ffn_norm.
  Transformer.norm: Transformer#norm.
  RMSNorm: RMSNorm#
  Attention.n_kv_heads: Attention#n_kv_heads.
  repeat_kv: repeat_kv().
  Attention.n_local_heads: Attention#n_local_heads.
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
  Transformer.tok_embeddings: Transformer#tok_embeddings.
  Transformer.output: Transformer#output.
  RMSNorm.__init__: RMSNorm#__init__().
  Attention.__init__: Attention#__init__().
  FeedForward.__init__: FeedForward#__init__().
  TransformerBlock.__init__: TransformerBlock#__init__().
  TransformerBlock.n_heads: TransformerBlock#n_heads.
  TransformerBlock.dim: TransformerBlock#dim.
  TransformerBlock.head_dim: TransformerBlock#head_dim.
  TransformerBlock.layer_id: TransformerBlock#layer_id.
  Transformer: Transformer#
  Transformer.__init__: Transformer#__init__().
  Transformer.params: Transformer#params.
  Transformer.vocab_size: Transformer#vocab_size.
  Transformer.n_layers: Transformer#n_layers.
---
# Module: [`llama_ref/model_original.py`](../../../../../raw/code/learning-machine/llama_ref/model_original.py)

## Classes
### `Attention`  ·  implements/extends Module
- def: [`llama_ref/model_original.py:117`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L117)
- signature: `class Attention(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L173`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L173) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `cache_k` — [`L156`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L156) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `cache_v` — [`L164`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L164) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `head_dim` — [`L125`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L125) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `n_kv_heads` — [`L120`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L120)
  - `n_local_heads` — [`L122`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L122)
  - `n_local_kv_heads` — [`L123`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L123) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `n_rep` — [`L124`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L124) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `wk` — [`L134`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L134) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `wo` — [`L148`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L148) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `wq` — [`L127`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L127) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `wv` — [`L141`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L141) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
- protocol/private: `__init__`[`L118`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L118)
- uses (calls/refs, reference-scoped): [`apply_rotary_emb`](model_original.md#apply_rotary_emb), [`repeat_kv`](model_original.md#repeat_kv)
- used by: [`attention`](model_original.md#TransformerBlock.attention)

### `FeedForward`  ·  implements/extends Module
- def: [`llama_ref/model_original.py:220`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L220)
- signature: `class FeedForward(nn.Module):`
- members:
  - `forward(self, x)` — [`L245`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L245)
  - `w1` — [`L235`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L235)
  - `w2` — [`L238`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L238)
  - `w3` — [`L241`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L241)
- protocol/private: `__init__`[`L221`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L221)
- used by: [`feed_forward`](model_original.md#TransformerBlock.feed_forward)

### `RMSNorm`  ·  implements/extends Module
- def: [`llama_ref/model_original.py:33`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L33) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
- signature: `class RMSNorm(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L42`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L42)
  - `eps` — [`L36`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L36)
  - `weight` — [`L37`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L37)
- protocol/private: `__init__`[`L34`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L34), `_norm`[`L39`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L39)
- used by: [`attention_norm`](model_original.md#TransformerBlock.attention_norm), [`ffn_norm`](model_original.md#TransformerBlock.ffn_norm), [`norm`](model_original.md#Transformer.norm)

### `Transformer`  ·  implements/extends Module
- def: [`llama_ref/model_original.py:278`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L278)
- signature: `class Transformer(nn.Module):`
- members:
  - `forward(self, tokens: torch.Tensor, start_pos: int)` — [`L306`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L306) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `freqs_cis` — [`L298`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L298) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `layers` — [`L289`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L289) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `n_layers` — [`L283`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L283)
  - `norm` — [`L293`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L293) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `output` — [`L294`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L294) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `params` — [`L281`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L281)
  - `tok_embeddings` — [`L285`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L285) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `vocab_size` — [`L282`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L282)
- protocol/private: `__init__`[`L279`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L279)
- uses (calls/refs, reference-scoped): [`precompute_freqs_cis`](model_original.md#precompute_freqs_cis), [`RMSNorm`](model_original.md#RMSNorm), [`TransformerBlock`](model_original.md#TransformerBlock)

### `TransformerBlock`  ·  implements/extends Module
- def: [`llama_ref/model_original.py:249`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L249)
- signature: `class TransformerBlock(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor])` — [`L266`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L266) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `attention` — [`L255`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L255) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `attention_norm` — [`L263`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L263) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `dim` — [`L253`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L253)
  - `feed_forward` — [`L256`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L256) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `ffn_norm` — [`L264`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L264) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
  - `head_dim` — [`L254`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L254)
  - `layer_id` — [`L262`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L262)
  - `n_heads` — [`L252`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L252)
- protocol/private: `__init__`[`L250`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L250)
- uses (calls/refs, reference-scoped): [`RMSNorm`](model_original.md#RMSNorm), [`Attention`](model_original.md#Attention), [`FeedForward`](model_original.md#FeedForward)
- used by: [`layers`](model_original.md#Transformer.layers)

## Functions
- `apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)` — [`L92`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L92) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
- `apply_scaling(freqs: torch.Tensor)` — [`L47`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L47)
- `precompute_freqs_cis(dim: int, end: int, theta: float = 10000, use_scaled: bool = False)` — [`L72`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L72)
- `repeat_kv(x: torch.Tensor, n_rep: int)` — [`L105`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L105) — torch.repeat_interleave(x, dim=2, repeats=n_rep) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)
- `reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor)` — [`L84`](../../../../../raw/code/learning-machine/llama_ref/model_original.py#L84) — documented in [llama_ref-model_original](../../concepts/llama_ref-model_original.md)

