# `model/` — Gemma 4 model source

## Where the model code lives

The authoritative PyTorch implementation of Gemma 4 is in HuggingFace
`transformers` at `src/transformers/models/gemma4/` — confirmed present on
`main` as of **2026-04-22**:

```
transformers/src/transformers/models/gemma4/
  configuration_gemma4.py     # Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig, ...
  modeling_gemma4.py          # Gemma4Model, Gemma4ForCausalLM, Gemma4ForConditionalGeneration
  modular_gemma4.py           # shared submodules (attention, MLP, decoder layer)
  processing_gemma4.py        # multimodal Gemma4Processor
  feature_extraction_gemma4.py
  image_processing_gemma4.py  # + image_processing_pil_gemma4.py
  video_processing_gemma4.py
  convert_gemma4_weights.py
  __init__.py
```

`__init__.py` in this folder **re-exports** the HF classes. It does not copy
their implementation. This keeps us in sync with upstream bugfixes and avoids
a 2000-line modeling file drifting in-tree. When a hypothesis needs to modify
the forward (e.g., swap in Splash attention, replace RMSNorm, stack scan
over layers), the modification should go in a separate module in this
folder and **subclass** / monkey-patch the upstream class — experiment pages
must document the patch explicitly.

The **DeepMind JAX reference** (`google-deepmind/gemma` GitHub) also ships a
Gemma 4 implementation, including an explicit `gm.nn.Gemma4_E4B()` class and
`gm.ckpts.CheckpointPath.GEMMA4_E4B_IT` checkpoint loader. We keep this as
the comparison baseline for the native-JAX port under `../../jax/` — not used
here.

## Gemma 4 E4B config (from `config.json`, 2026-04-22)

Text-only path (the path this program optimizes):

| Field | Value |
|---|---|
| `model_type` | `gemma4` |
| `architectures` | `["Gemma4ForConditionalGeneration"]` |
| `num_hidden_layers` | **42** |
| `hidden_size` | **2560** |
| `num_attention_heads` | **8** |
| `num_key_value_heads` | **2** |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `intermediate_size` | **10240** |
| `vocab_size` | **262144** |
| `max_position_embeddings` | 131072 |
| `sliding_window` | 512 |
| `num_kv_shared_layers` | 18 |
| `rms_norm_eps` | 1e-6 |
| `final_logit_softcapping` | 30.0 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `tie_word_embeddings` | true |
| `dtype` | `bfloat16` |

RoPE:
- Full-attention layers: `rope_theta=1e6`, `rope_type=proportional`,
  `partial_rotary_factor=0.25`.
- Sliding-window layers: `rope_theta=10000`, `rope_type=default`.

"E" in E4B stands for **effective** parameters (Per-Layer Embeddings,
"PLE"). Headline: **4.5B effective / 8B including embeddings**. The
training FLOPs math uses effective params.

Parameters that **matter for sharding**:
- `num_attention_heads = 8` → with TP=8 on v6e-8, exactly 1 Q head per chip.
  Divides cleanly.
- `num_key_value_heads = 2` → with TP=8, **K/V cannot be head-sharded**
  (would leave 6 chips with zero KV heads). `sharding.py` replicates K/V by
  default. Future hypothesis: GQA-aware sharding that broadcasts KV groups.
- `num_kv_shared_layers = 18` → 18 layers share KV with some other layer.
  This cross-layer KV sharing is a Gemma 4 novelty and is **not reflected**
  in the initial torchax sharding rules. If HF lays the shared-KV weights
  out as a special structure, the regex matcher may miss them. **Verify
  after first run.**
- Hybrid attention (SW 512 + global) is **per-layer** in the config. XLA
  will generate different attention HLO per layer type; scan-over-layers
  (hypothesis 7 in the program) will need to handle this (two scans, one
  per layer type, or a single scan with a conditional).

## ASSUMED — verify against real config

- **`Gemma4ForCausalLM` is the right text-only head.** The config lists
  `Gemma4ForConditionalGeneration` as the canonical architecture
  (multimodal). We load the causal-LM head directly. If that class does not
  exist in the installed transformers version, fall back to
  `Gemma4ForConditionalGeneration` and pass text-only inputs; HF will skip
  the vision/audio towers. `train.py` documents this branch.
- **GQA grouping for TP sharding.** We assume Q projections are laid out as
  a single `(hidden, num_heads * head_dim) = (2560, 2048)` linear. Gemma 3
  did this; Gemma 4 config is consistent. **Verify by inspecting a real
  state_dict key.** The regex in `sharding.py` matches substrings
  `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj` —
  HF Gemma-family convention. If Gemma 4's modular file fused any of these
  (e.g., a single `qkv_proj`), the regex must be updated.
- **LM head weight tying.** `tie_word_embeddings=true` means `lm_head.weight`
  and `embed_tokens.weight` alias the same tensor. JittableModule's
  parameter deduplication handles this (see torchax `interop.py:88-98`) —
  but a sharding rule placed on only one of the two names can be silently
  dropped. We assign the row-sharded spec `P(None, 'tp')` to **both**
  substrings and rely on the dedup to pick the survivor.
- **`final_logit_softcapping=30.0` is implemented as a `tanh` + scale** in
  HF (per Gemma 2/3 precedent). This is a pointwise op and does not affect
  sharding, but it **does** show up as a non-trivial HLO node near the end
  of the forward — note for baseline profile analysis.

## See also

- Program page: [`../../../README.md`](../../README.md)
- torchax codebase page: [`../../../../codebases/torchax.md`](../../../../codebases/torchax.md)
- jax-huggingface part 2 (TP recipe):
  [`../../../../sources/2026-jax-huggingface-part-2.md`](../../../../sources/2026-jax-huggingface-part-2.md)
- DeepMind Gemma JAX reference: <https://github.com/google-deepmind/gemma>
