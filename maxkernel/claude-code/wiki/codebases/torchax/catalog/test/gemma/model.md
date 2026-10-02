---
title: 'Module: test/gemma/model.py'
type: catalog
provenance: extracted
module: test/gemma/model.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.gemma.model`/
symbols:
  GemmaAttention.forward: GemmaAttention#forward().
  GemmaForCausalLM.config: GemmaForCausalLM#config.
  GemmaForCausalLM.forward: GemmaForCausalLM#forward().
  GemmaForCausalLM.generate: GemmaForCausalLM#generate().
  GemmaDecoderLayer.self_attn: GemmaDecoderLayer#self_attn.
  GemmaAttention.qkv_proj: GemmaAttention#qkv_proj.
  GemmaAttention.num_kv_heads: GemmaAttention#num_kv_heads.
  GemmaAttention.o_proj: GemmaAttention#o_proj.
  GemmaDecoderLayer.mlp: GemmaDecoderLayer#mlp.
  GemmaModel.layers: GemmaModel#layers.
  GemmaForCausalLM.embedder: GemmaForCausalLM#embedder.
  GemmaDecoderLayer.forward: GemmaDecoderLayer#forward().
  GemmaDecoderLayer.input_layernorm: GemmaDecoderLayer#input_layernorm.
  GemmaDecoderLayer.post_attention_layernorm: GemmaDecoderLayer#post_attention_layernorm.
  GemmaModel.norm: GemmaModel#norm.
  GemmaAttention.head_dim: GemmaAttention#head_dim.
  Linear.forward: Linear#forward().
  Embedding.forward: Embedding#forward().
  RMSNorm.forward: RMSNorm#forward().
  GemmaMLP.forward: GemmaMLP#forward().
  GemmaAttention.num_heads: GemmaAttention#num_heads.
  GemmaAttention.num_queries_per_kv: GemmaAttention#num_queries_per_kv.
  GemmaAttention.kv_size: GemmaAttention#kv_size.
  GemmaForCausalLM.tokenizer: GemmaForCausalLM#tokenizer.
  GemmaAttention.q_size: GemmaAttention#q_size.
  Linear: Linear#
  GemmaModel.forward: GemmaModel#forward().
  GemmaForCausalLM.sampler: GemmaForCausalLM#sampler.
  Embedding.weight_scaler: Embedding#weight_scaler.
  Linear.weight_scaler: Linear#weight_scaler.
  RMSNorm._norm: RMSNorm#_norm().
  GemmaMLP.gate_proj: GemmaMLP#gate_proj.
  GemmaMLP.up_proj: GemmaMLP#up_proj.
  GemmaMLP.down_proj: GemmaMLP#down_proj.
  GemmaAttention.scaling: GemmaAttention#scaling.
  GemmaForCausalLM.model: GemmaForCausalLM#model.
  Embedding.weight: Embedding#weight.
  RMSNorm: RMSNorm#
  GemmaDecoderLayer.__init__: GemmaDecoderLayer#__init__().
  GemmaModel.__init__: GemmaModel#__init__().
  GemmaModel.vocab_size: GemmaModel#vocab_size.
  GemmaForCausalLM.__init__: GemmaForCausalLM#__init__().
  apply_rotary_emb: apply_rotary_emb().
  Linear.weight: Linear#weight.
  RMSNorm.weight: RMSNorm#weight.
  GemmaAttention.hidden_size: GemmaAttention#hidden_size.
  Sampler: Sampler#
  precompute_freqs_cis: precompute_freqs_cis().
  Linear.quant: Linear#quant.
  Embedding: Embedding#
  Embedding.quant: Embedding#quant.
  RMSNorm.eps: RMSNorm#eps.
  RMSNorm.add_unit_offset: RMSNorm#add_unit_offset.
  GemmaMLP: GemmaMLP#
  GemmaAttention: GemmaAttention#
  GemmaDecoderLayer: GemmaDecoderLayer#
  GemmaModel: GemmaModel#
  GemmaForCausalLM: GemmaForCausalLM#
  Sampler.__init__: Sampler#__init__().
  Sampler.vocab_size: Sampler#vocab_size.
  Sampler.forward: Sampler#forward().
  Linear.__init__: Linear#__init__().
  Embedding.__init__: Embedding#__init__().
  RMSNorm.__init__: RMSNorm#__init__().
  GemmaMLP.__init__: GemmaMLP#__init__().
  GemmaAttention.__init__: GemmaAttention#__init__().
  GemmaModel.config: GemmaModel#config.
  GemmaForCausalLM.load_weights: GemmaForCausalLM#load_weights().
---
# Module: [`test/gemma/model.py`](../../../../../../raw/code/torchax/test/gemma/model.py)

## Classes
### `Embedding`  ·  implements/extends Module
- def: [`test/gemma/model.py:126`](../../../../../../raw/code/torchax/test/gemma/model.py#L126)
- signature: `class Embedding(nn.Module):`
- members:
  - `forward(self, x)` — [`L142`](../../../../../../raw/code/torchax/test/gemma/model.py#L142)
  - `quant` — [`L140`](../../../../../../raw/code/torchax/test/gemma/model.py#L140)
  - `weight` — [`L130`](../../../../../../raw/code/torchax/test/gemma/model.py#L130)
  - `weight_scaler` — [`L134`](../../../../../../raw/code/torchax/test/gemma/model.py#L134)
- protocol/private: `__init__`[`L127`](../../../../../../raw/code/torchax/test/gemma/model.py#L127)
- used by: (2 test-only callers)

### `GemmaAttention`  ·  implements/extends Module
- def: [`test/gemma/model.py:195`](../../../../../../raw/code/torchax/test/gemma/model.py#L195)
- signature: `class GemmaAttention(nn.Module):`
- members:
  - `forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor)` — [`L227`](../../../../../../raw/code/torchax/test/gemma/model.py#L227)
  - `head_dim` — [`L213`](../../../../../../raw/code/torchax/test/gemma/model.py#L213)
  - `hidden_size` — [`L212`](../../../../../../raw/code/torchax/test/gemma/model.py#L212)
  - `kv_size` — [`L216`](../../../../../../raw/code/torchax/test/gemma/model.py#L216)
  - `num_heads` — [`L206`](../../../../../../raw/code/torchax/test/gemma/model.py#L206)
  - `num_kv_heads` — [`L207`](../../../../../../raw/code/torchax/test/gemma/model.py#L207)
  - `num_queries_per_kv` — [`L210`](../../../../../../raw/code/torchax/test/gemma/model.py#L210)
  - `o_proj` — [`L225`](../../../../../../raw/code/torchax/test/gemma/model.py#L225)
  - `q_size` — [`L215`](../../../../../../raw/code/torchax/test/gemma/model.py#L215)
  - `qkv_proj` — [`L220`](../../../../../../raw/code/torchax/test/gemma/model.py#L220)
  - `scaling` — [`L218`](../../../../../../raw/code/torchax/test/gemma/model.py#L218)
- protocol/private: `__init__`[`L196`](../../../../../../raw/code/torchax/test/gemma/model.py#L196)
- uses (calls/refs, reference-scoped): (2 test-only callers)
- used by: (1 test-only callers)

### `GemmaDecoderLayer`  ·  implements/extends Module
- def: [`test/gemma/model.py:284`](../../../../../../raw/code/torchax/test/gemma/model.py#L284)
- signature: `class GemmaDecoderLayer(nn.Module):`
- members:
  - `forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor)` — [`L305`](../../../../../../raw/code/torchax/test/gemma/model.py#L305)
  - `input_layernorm` — [`L302`](../../../../../../raw/code/torchax/test/gemma/model.py#L302)
  - `mlp` — [`L297`](../../../../../../raw/code/torchax/test/gemma/model.py#L297)
  - `post_attention_layernorm` — [`L303`](../../../../../../raw/code/torchax/test/gemma/model.py#L303)
  - `self_attn` — [`L290`](../../../../../../raw/code/torchax/test/gemma/model.py#L290)
- protocol/private: `__init__`[`L285`](../../../../../../raw/code/torchax/test/gemma/model.py#L285)
- uses (calls/refs, reference-scoped): (11 test-only callers)
- used by: (1 test-only callers)

### `GemmaForCausalLM`  ·  implements/extends Module
- def: [`test/gemma/model.py:366`](../../../../../../raw/code/torchax/test/gemma/model.py#L366)
- signature: `class GemmaForCausalLM(nn.Module):`
- members:
  - `forward(self, input_token_ids: torch.Tensor, input_positions: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor, output_positions: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor, **kwargs)` — [`L390`](../../../../../../raw/code/torchax/test/gemma/model.py#L390)
  - `generate(self, prompts: str | Sequence[str], device: Any, output_len: int = 100, temperature: float = 0.95, top_p: float = 1, top_k: int = 100)` — [`L425`](../../../../../../raw/code/torchax/test/gemma/model.py#L425) — Generates responses for given prompts using Gemma model.
  - `load_weights(self, model_path: str)` — [`L531`](../../../../../../raw/code/torchax/test/gemma/model.py#L531)
  - `config` — [`L372`](../../../../../../raw/code/torchax/test/gemma/model.py#L372)
  - `embedder` — [`L380`](../../../../../../raw/code/torchax/test/gemma/model.py#L380)
  - `model` — [`L381`](../../../../../../raw/code/torchax/test/gemma/model.py#L381)
  - `sampler` — [`L382`](../../../../../../raw/code/torchax/test/gemma/model.py#L382)
  - `tokenizer` — [`L379`](../../../../../../raw/code/torchax/test/gemma/model.py#L379)
- protocol/private: `__init__`[`L367`](../../../../../../raw/code/torchax/test/gemma/model.py#L367)
- uses (calls/refs, reference-scoped): (16 test-only callers)
- used by: (1 test-only callers)

### `GemmaMLP`  ·  implements/extends Module
- def: [`test/gemma/model.py:174`](../../../../../../raw/code/torchax/test/gemma/model.py#L174)
- signature: `class GemmaMLP(nn.Module):`
- members:
  - `forward(self, x)` — [`L186`](../../../../../../raw/code/torchax/test/gemma/model.py#L186)
  - `down_proj` — [`L184`](../../../../../../raw/code/torchax/test/gemma/model.py#L184)
  - `gate_proj` — [`L182`](../../../../../../raw/code/torchax/test/gemma/model.py#L182)
  - `up_proj` — [`L183`](../../../../../../raw/code/torchax/test/gemma/model.py#L183)
- protocol/private: `__init__`[`L175`](../../../../../../raw/code/torchax/test/gemma/model.py#L175)
- uses (calls/refs, reference-scoped): (1 test-only callers)
- used by: (1 test-only callers)

### `GemmaModel`  ·  implements/extends Module
- def: [`test/gemma/model.py:334`](../../../../../../raw/code/torchax/test/gemma/model.py#L334)
- signature: `class GemmaModel(nn.Module):`
- members:
  - `forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor)` — [`L345`](../../../../../../raw/code/torchax/test/gemma/model.py#L345)
  - `config` — [`L337`](../../../../../../raw/code/torchax/test/gemma/model.py#L337)
  - `layers` — [`L340`](../../../../../../raw/code/torchax/test/gemma/model.py#L340)
  - `norm` — [`L343`](../../../../../../raw/code/torchax/test/gemma/model.py#L343)
  - `vocab_size` — [`L338`](../../../../../../raw/code/torchax/test/gemma/model.py#L338)
- protocol/private: `__init__`[`L335`](../../../../../../raw/code/torchax/test/gemma/model.py#L335)
- uses (calls/refs, reference-scoped): (7 test-only callers)
- used by: (1 test-only callers)

### `Linear`  ·  implements/extends Module
- def: [`test/gemma/model.py:102`](../../../../../../raw/code/torchax/test/gemma/model.py#L102)
- signature: `class Linear(nn.Module):`
- members:
  - `forward(self, x)` — [`L118`](../../../../../../raw/code/torchax/test/gemma/model.py#L118)
  - `quant` — [`L116`](../../../../../../raw/code/torchax/test/gemma/model.py#L116)
  - `weight` — [`L106`](../../../../../../raw/code/torchax/test/gemma/model.py#L106)
  - `weight_scaler` — [`L110`](../../../../../../raw/code/torchax/test/gemma/model.py#L110)
- protocol/private: `__init__`[`L103`](../../../../../../raw/code/torchax/test/gemma/model.py#L103)
- used by: (5 test-only callers)

### `RMSNorm`  ·  implements/extends Module
- def: [`test/gemma/model.py:150`](../../../../../../raw/code/torchax/test/gemma/model.py#L150)
- signature: `class RMSNorm(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L165`](../../../../../../raw/code/torchax/test/gemma/model.py#L165)
  - `add_unit_offset` — [`L159`](../../../../../../raw/code/torchax/test/gemma/model.py#L159)
  - `eps` — [`L158`](../../../../../../raw/code/torchax/test/gemma/model.py#L158)
  - `weight` — [`L160`](../../../../../../raw/code/torchax/test/gemma/model.py#L160)
- protocol/private: `__init__`[`L151`](../../../../../../raw/code/torchax/test/gemma/model.py#L151), `_norm`[`L162`](../../../../../../raw/code/torchax/test/gemma/model.py#L162)
- used by: (3 test-only callers)

### `Sampler`  ·  implements/extends Module
- def: [`test/gemma/model.py:28`](../../../../../../raw/code/torchax/test/gemma/model.py#L28)
- signature: `class Sampler(nn.Module):`
- members:
  - `forward(self, embedding: torch.Tensor, hidden_states: torch.Tensor, output_positions: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor, embedding_bias: torch.Tensor | None = None)` — [`L34`](../../../../../../raw/code/torchax/test/gemma/model.py#L34)
  - `vocab_size` — [`L31`](../../../../../../raw/code/torchax/test/gemma/model.py#L31)
- protocol/private: `__init__`[`L29`](../../../../../../raw/code/torchax/test/gemma/model.py#L29)
- used by: (1 test-only callers)

## Functions
- `apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor)` — [`L89`](../../../../../../raw/code/torchax/test/gemma/model.py#L89) — Applies the rotary embedding to the query and key tensors.
- `precompute_freqs_cis(dim: int, end: int, theta: float = 10000)` — [`L80`](../../../../../../raw/code/torchax/test/gemma/model.py#L80) — Precomputes the frequency cis.

