# `model/` — Llama 3 8B model source

The trainer uses HuggingFace `transformers` `LlamaForCausalLM` directly. We
don't fork the model; we re-export the upstream classes from this package and
add a sharding plan keyed on HF parameter names.

## Files

- [`__init__.py`](__init__.py) — re-exports `LlamaConfig`, `LlamaForCausalLM`,
  `AutoTokenizer`, and the `sharding` submodule.
- [`sharding.py`](sharding.py) — `SHARDING_MAP` keyed on HF parameter names
  (with wildcard `*` for layer indices) and `get_sharding(mesh, name)` helper.

## HF Llama-3-8B config (text-only, from `meta-llama/Meta-Llama-3-8B`)

| Field | Value |
|---|---|
| `num_hidden_layers` | 32 |
| `hidden_size` | 4096 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA, 4× repeat) |
| `head_dim` | 128 |
| `intermediate_size` | 14336 |
| `vocab_size` | 128256 |
| `rope_theta` | 500000 |
| `max_position_embeddings` | 8192 |

Total params ≈ **8.03 B** at bf16 (~16 GB on disk).

## Sharding scheme

Single mesh with axes `(fsdp, tp)`. For TP=1 (default on v6e-4 / v6e-8) only
FSDP is active.

- **Embedding** (`embed_tokens.weight`, `(vocab, hidden)`): sharded
  `('fsdp', 'tp')` — vocab axis FSDP, hidden axis TP. With TP=1 vocab gets
  4-way (or 8-way) split.
- **QKV projections** (`q/k/v_proj`): `('tp', 'fsdp')` — output (head) axis TP,
  input (hidden) axis FSDP. K/V are smaller under GQA but use the same scheme.
- **O projection** (`o_proj`): `('fsdp', 'tp')` — opposite of QKV; output is
  hidden, input is head dim.
- **FFN gate/up** (`gate_proj`, `up_proj`): `('tp', 'fsdp')`.
- **FFN down** (`down_proj`): `('fsdp', 'tp')`.
- **RMSNorm 1-D** (`*_layernorm.weight`, `model.norm.weight`): `('fsdp',)`.
- **LM head** (`lm_head.weight`): `('tp', 'fsdp')`.

## Caveats

- `num_key_value_heads = 8` means a TP mesh > 8 over-shards K/V (under-utilized
  if TP=16 etc.). Stay at TP ≤ 8.
- HF computes RoPE caches and causal masks lazily inside the layer's forward.
  Under torchax + JAX `jit`, these become traced ops — fine.
- HF returns `CausalLMOutputWithPast` (a tuple-ish wrapper). The trainer extracts
  `.logits` and computes the loss outside the model.
