---
slug: maxdiffusion
commit: 09844570abd80e0c88da1677ec047a8a094d26d2
scip_tool: scip-python
updated: 2026-07-03
---

# maxdiffusion internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — the whole system in one page (main concepts + core diagrams + a map of the wiki).

## Concepts (deep)
| Concept | Page | Status |
|---|---|---|
| maxdiffusion-common_types | [maxdiffusion-common_types](concepts/maxdiffusion-common_types.md) | fresh |
| maxdiffusion-configuration_utils | [maxdiffusion-configuration_utils](concepts/maxdiffusion-configuration_utils.md) | fresh |
| maxdiffusion-kernels-splash_attention-splash_attention_kernel | [maxdiffusion-kernels-splash_attention-splash_attention_kernel](concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel.md) | fresh |
| maxdiffusion-kernels-splash_attention-splash_attention_kernel_test | [maxdiffusion-kernels-splash_attention-splash_attention_kernel_test](concepts/maxdiffusion-kernels-splash_attention-splash_attention_kernel_test.md) | fresh |
| maxdiffusion-kernels-splash_attention-splash_attention_mask | [maxdiffusion-kernels-splash_attention-splash_attention_mask](concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask.md) | fresh |
| maxdiffusion-kernels-splash_attention-splash_attention_mask_info | [maxdiffusion-kernels-splash_attention-splash_attention_mask_info](concepts/maxdiffusion-kernels-splash_attention-splash_attention_mask_info.md) | fresh |
| maxdiffusion-models-attention_flax | [maxdiffusion-models-attention_flax](concepts/maxdiffusion-models-attention_flax.md) | fresh |
| maxdiffusion-models-embeddings_flax | [maxdiffusion-models-embeddings_flax](concepts/maxdiffusion-models-embeddings_flax.md) | fresh |
| maxdiffusion-models-flux-transformers-transformer_flux_flax | [maxdiffusion-models-flux-transformers-transformer_flux_flax](concepts/maxdiffusion-models-flux-transformers-transformer_flux_flax.md) | fresh |
| maxdiffusion-models-ltx2-autoencoder_kl_ltx2 | [maxdiffusion-models-ltx2-autoencoder_kl_ltx2](concepts/maxdiffusion-models-ltx2-autoencoder_kl_ltx2.md) | fresh |
| maxdiffusion-models-ltx2-autoencoder_kl_ltx2_audio | [maxdiffusion-models-ltx2-autoencoder_kl_ltx2_audio](concepts/maxdiffusion-models-ltx2-autoencoder_kl_ltx2_audio.md) | fresh |
| maxdiffusion-models-ltx2-transformer_ltx2 | [maxdiffusion-models-ltx2-transformer_ltx2](concepts/maxdiffusion-models-ltx2-transformer_ltx2.md) | fresh |
| maxdiffusion-models-ltx2-vocoder_ltx2 | [maxdiffusion-models-ltx2-vocoder_ltx2](concepts/maxdiffusion-models-ltx2-vocoder_ltx2.md) | fresh |
| maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder | [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md) | fresh |
| maxdiffusion-models-ltx_video-transformers-attention | [maxdiffusion-models-ltx_video-transformers-attention](concepts/maxdiffusion-models-ltx_video-transformers-attention.md) | fresh |
| maxdiffusion-models-ltx_video-transformers_pytorch-attention | [maxdiffusion-models-ltx_video-transformers_pytorch-attention](concepts/maxdiffusion-models-ltx_video-transformers_pytorch-attention.md) | fresh |
| maxdiffusion-models-resnet_flax | [maxdiffusion-models-resnet_flax](concepts/maxdiffusion-models-resnet_flax.md) | fresh |
| maxdiffusion-models-unet_2d_blocks_flax | [maxdiffusion-models-unet_2d_blocks_flax](concepts/maxdiffusion-models-unet_2d_blocks_flax.md) | fresh |
| maxdiffusion-models-vae_flax | [maxdiffusion-models-vae_flax](concepts/maxdiffusion-models-vae_flax.md) | fresh |
| maxdiffusion-models-wan-autoencoder_kl_wan | [maxdiffusion-models-wan-autoencoder_kl_wan](concepts/maxdiffusion-models-wan-autoencoder_kl_wan.md) | fresh |
| maxdiffusion-models-wan-autoencoder_kl_wan_2p2 | [maxdiffusion-models-wan-autoencoder_kl_wan_2p2](concepts/maxdiffusion-models-wan-autoencoder_kl_wan_2p2.md) | fresh |
| maxdiffusion-models-wan-transformers-transformer_wan_animate | [maxdiffusion-models-wan-transformers-transformer_wan_animate](concepts/maxdiffusion-models-wan-transformers-transformer_wan_animate.md) | fresh |
| maxdiffusion-schedulers-scheduling_utils_flax | [maxdiffusion-schedulers-scheduling_utils_flax](concepts/maxdiffusion-schedulers-scheduling_utils_flax.md) | fresh |
| maxdiffusion-utils-import_utils | [maxdiffusion-utils-import_utils](concepts/maxdiffusion-utils-import_utils.md) | fresh |

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); **module
catalogs** represent the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **4614** across 135 modules
- deep (concept pages): **262** (5.7%)
- catalog-only: **4352**
- represented total: **4614** (100.0%)
- classes represented: **354/354**

See [`catalog/`](catalog/) for the generated per-module structural index.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
