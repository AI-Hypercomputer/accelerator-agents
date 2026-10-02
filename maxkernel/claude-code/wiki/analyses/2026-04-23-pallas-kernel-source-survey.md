---
title: "Survey: Public sources of JAX Pallas kernel code"
type: analysis
tags: [survey, pallas, kernels, tpu, gpu, ingest-candidates]
created: 2026-04-23
updated: 2026-04-23
---

Web-research survey of every public source of JAX Pallas kernel code I could find as of 2026-04-23. The goal is to identify **which repos are worth ingesting as wiki codebases** to widen the optimization search space beyond what the currently-ingested [tokamax](../codebases/tokamax.md) + upstream JAX (`jax.experimental.pallas.ops.tpu`) provide. Findings are organized by tier (production-grade libraries → embedded kernels in monorepos → research companion code → pedagogy → meta/agents → marginal). Each entry includes what kernels it carries that are **not** already in tokamax or upstream JAX, which is the criterion for ingestion priority. Several entries directly affect previously filed hypothesis candidates on [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) — see § *Direct impact on open wiki state*.

## Methodology

The search was delegated to a general-purpose agent on 2026-04-23 with instructions to WebSearch broadly, then WebFetch / GitHub-API verify top candidates. Queries included variants of `site:github.com pallas tpu kernel`, `"from jax.experimental import pallas"` code search, repo-name probes for known orgs (`google-deepmind/*`, `google/*`, `AI-Hypercomputer/*`, `openxla/*`, vLLM/SGLang projects, Apple axlearn, etc.). Results below are categorized by whether (a) the agent verified the repo tree / commit metadata, or (b) it saw the URL but could not verify depth — the latter are marked *unverified*. The raw agent report is preserved for traceability.

Omitted from the tables below: **[jax-ml/jax](https://github.com/jax-ml/jax)** (upstream Pallas), **[openxla/tokamax](../codebases/tokamax.md)** (already ingested), and adjacent non-kernel repos ([openxla/xprof](../codebases/xprof.md), [openxla/stablehlo](../codebases/stablehlo.md), [google-deepmind/scaling-book](../codebases/scaling-book.md)). These are the anchors the survey runs *against*.

## Tier 1 — Production-grade kernel libraries (top ingest candidates)

These are actively maintained, are used to train or serve production models, and carry kernels not present in tokamax or upstream JAX. Verified.

| Repo | What it ships that isn't in tokamax/upstream | TPU | GPU | Last push | License |
|---|---|---|---|---|---|
| [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) | `ragged_paged_attention v2/v3`, `mla v1/v2`, `gdn` (gated delta net), `sparse_core` gather/scatter, `structured_sparse_matmul`, `all_gather_matmul`, experimental `batched_rpa`, `fused_moe`, `quantized_matmul (blockwise)`. **Tuned block-size tables shipped.** | Yes | JAX→XLA | 2026-04-23 | Apache-2.0 |
| [sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax) | `simple_gla` (gated linear attention), speculative-decoding tree kernels, multimodal `flash_attention`, `fused_moe v1`, `update_kv_cache`, `ragged_paged_attention v3`. | Yes | No | 2026-04-23 | Apache-2.0 |
| [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) | `gather_reduce_sc`, `sort_activations`, `paged_attention_v2`; vendored splash + `ragged_attention`; megablox {gmm, backend, ops}. **Gemma/Llama/DeepSeek/Qwen/Mistral production training.** | Yes | — | 2026-04-23 | Apache-2.0 |
| [apple/axlearn](https://github.com/apple/axlearn) | **Mamba/SSD Pallas** (`ssm_kernels/mamba_kernels.py`, `ssd_kernels.py`), **rattention linear-attention** kernels; also parallel TPU+GPU Pallas coverage for attention/paged/decoding/kv-cache/megablock. | Yes | Yes | 2026-04-16 | Apache-2.0 |
| [AI-Hypercomputer/maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) | **Pallas `ring_attention_kernel` integrated with splash** at `src/maxdiffusion/kernels/splash_attention/`. Reference implementation for wiring ring attention into a splash-style public API. | Yes | — | 2026-04-23 | Apache-2.0 |
| [AI-Hypercomputer/JetStream](https://github.com/AI-Hypercomputer/JetStream) | `chunked_prefill_attention` (TPU), `paged_attention` (TPU), `collective_matmul` (TPU) under `experimental/jax/inference/kernel/`. Older than tpu-inference but still maintained. | Yes | — | 2026-01-05 | Apache-2.0 |

**Ingest priority ranking** (my read of the survey, considering novelty-vs-tokamax and direct relevance to the Gemma 4 optimization loop):

1. **AI-Hypercomputer/maxtext** — direct analogue to your current trainer; every kernel choice it makes is a prior on what works on TPU for Gemma-class models.
2. **vllm-project/tpu-inference** — broadest kernel surface not in tokamax; tuned block-size tables are a cheap source of hypothesis priors.
3. **AI-Hypercomputer/maxdiffusion** — direct reference impl for the open "wire `ring_attention_kernel` through public API" hypothesis.
4. **apple/axlearn** — unique coverage of SSM/Mamba/rattention; matters once a non-transformer model enters the loop.
5. **sgl-project/sglang-jax** — `simple_gla` and speculative-decoding kernels are closest to inference-time optimization work.

## Tier 2 — Monorepos with embedded Pallas kernels

Not dedicated kernel libraries but ship useful kernels as part of a larger system. Ingest the kernel subdirectory rather than the whole repo.

| Repo | Kernel path | What's there | Active |
|---|---|---|---|
| [google/tunix](https://github.com/google/tunix) | `tunix/models/qwen{2,3}/model.py`, `gemma4/model.py` | Inline Pallas call-site examples — not a kernel library, but real-world consumption patterns. | Yes |
| [google/qwix](https://github.com/google/qwix) | `qwix/_src/core/pallas.py`, `qwix/contrib/kernels/quantized_matmul.py` | JAX-native quantization (replacing AQT); Pallas hooks + quantized matmul kernel. | Yes |
| [google/aqt](https://github.com/google/aqt) | `aqt/jax/v2/pallas/` | `pallas_call.py`, `dot_general.py`, `pallas_tensor.py`, `quantizer.py`. Predecessor to qwix; EOL-ish. | Yes |
| [google/torchax](../codebases/torchax.md) | `examples/train_llama_torchtitan/splash_attn.py` | Splash wrapper for torchax — small. Codebase already ingested. | Yes |
| [google/jaxite](https://github.com/google/jaxite) | `jaxite/jaxite_lib/polymul_kernel.py` | **FHE polynomial multiplication** Pallas kernel (TFHE). Non-ML. Niche. | Yes |
| [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma) | `recurrentgemma/jax/pallas.py` | Griffin linear-recurrence scan Pallas kernel (TPU). Single kernel, well-documented. | Yes |
| [google-deepmind/simply](https://github.com/google-deepmind/simply) | `simply/utils/ragged_paged_attention.py` | Ragged paged attention (vendored/reference). | Yes |
| [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) | `graphcast/sparse_transformer.py`, `sparse_transformer_utils.py` | Block-sparse attention Pallas for weather models. Non-LLM. | Yes |
| [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) **@ tag v3.0.1** | `src/alphafold3/jax/gated_linear_unit/*`, `src/alphafold3/jax/attention/flash_attention.py` | **Production fused GLU Pallas kernel** (fused matmul+GLU with tiling) and triangle-attn flash. **Removed from main — pin tag v3.0.1.** | Yes |
| [google/paxml](https://github.com/google/paxml) + [google/praxis](https://github.com/google/praxis) | `praxis/layers/gpu_fast_attention.py` | Mostly consumes upstream Pallas; one GPU fast-attention wrapper. Legacy but maintained. | Yes |
| [marin-community/marin](https://github.com/marin-community/marin) | `lib/levanter/src/levanter/kernels/pallas/` | `fused_cross_entropy_loss/pallas_tpu.py`, `pallas_gpu.py`, `ssd/`, `template_kernel.py`, **`autotune_utils.py`, `cost_estimate_utils.py`** (autotune harness worth studying). | Yes |
| [marin-community/levanter](https://github.com/marin-community/levanter) | same kernel path | Upstream of above; same kernels. | Yes |
| [pytorch/xla](https://github.com/pytorch/xla) | `torch_xla/experimental/pallas_kernels/` | `multi_queries_paged_attention`, `quantized_matmul`, `ragged_paged_attention`, `ragged_paged_attention_v2`. **Relevant to torchax-based trainer.** | Yes |
| [google-pytorch/torchtitan](https://github.com/google-pytorch/torchtitan) | `torchtitan/experiments/tpu/kernels/linear_softmax_cross_entropy_loss.py` | Tokamax-derived fused CE-loss kernel integrated into torchtitan on TPU. | Yes |
| [aphrodite-engine/aphrodite-engine](https://github.com/aphrodite-engine/aphrodite-engine) | `aphrodite/attention/ops/pallas_kv_cache_update.py` | KV-cache-update Pallas kernel. vLLM fork. **AGPL-3.0 license** — incompatible with some downstream use; note before vendoring. | Yes |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | `torch/_inductor/codegen/pallas.py` | **Pallas as a TorchInductor codegen target** — generates Pallas from Inductor IR. Not a kernel library but relevant to PyTorch-on-TPU futures. | Yes |

## Tier 3 — Community kernel libraries

| Repo | What's there | Notes |
|---|---|---|
| [erfanzar/ejkernel](https://github.com/erfanzar/ejkernel) | TPU Pallas: flash_attention, **flash_mla, blocksparse_attention, deepseek_attn**, page_attention, prefill_page_attention, ragged_page_attention v2/v3, ragged_decode_attention, **multi_latent_ragged_page_attention v1/v2**, grouped_matmul v1/v2/v3, quantized_matmul, **gated_delta_rule, ragged_gated_delta_rule**, all_gather_matmul, **ring_attention**. GPU Triton/CuTE counterparts. | 22⭐ Apache-2.0, active. Community coverage of kernel variants not elsewhere. |
| [erfanzar/EasyDeL](https://github.com/erfanzar/EasyDeL) | `easydel/kernels/tpu_ops/`: ragged_attention_pallas, ragged_paged_attention_pallas, inference GDN ops. | 355⭐. Framework wrapping ejkernel. |
| [erfanzar/jax-flash-attn2](https://github.com/erfanzar/jax-flash-attn2) | Multi-backend FA2 in JAX (Triton/Pallas/JAX). | 34⭐. |
| [Essential-AI/maxtext-external](https://github.com/Essential-AI/maxtext-external) | Essential AI's MaxText fork with additional kernels. | Unverified depth. |

## Tier 4 — Research / paper companion code

| Repo | Paper | Kernels | Notes |
|---|---|---|---|
| [haoliuhl/ringattention](https://github.com/haoliuhl/ringattention) | Liu, Zaharia, Abbeel 2023, *Ring Attention with Blockwise Transformers for Near-Infinite Context* (arXiv:2310.01889) | `ringattention/ringattention_pallas_tpu.py` | **Canonical Pallas Ring Attention.** 770⭐ Apache-2.0. Updates the previously-noted gap ("no public zig-zag or ring-attention impl on TPU") — ring exists, zig-zag balancing still unverified. |
| [jondeaton/ring-attention-jax-pallas](https://github.com/jondeaton/ring-attention-jax-pallas) | Derivative of above | Ring adapted to FA2 + flex-style biases | **Survey URL returned 404 during verification** — confirm it still exists before relying on it. |
| [lengstrom/flashback](https://github.com/lengstrom/flashback) | FlashAttention backward-over-backward | Higher-order-grad FA Pallas | 11⭐. Relevant to 2nd-order training. |
| [zhixuan-lin/gla-jax](https://github.com/zhixuan-lin/gla-jax) | Gated Linear Attention (Yang et al.) | GLA Pallas | 8⭐. |
| [AlexG1105/mamba2-jax-pallas](https://github.com/AlexG1105/mamba2-jax-pallas) | Mamba2 | SSD Pallas | 0⭐, verify. |
| [rdyro/gpu_ragged_dot](https://github.com/rdyro/gpu_ragged_dot) | Blog (Robert Dyro) | Ragged-dot with autodiff-through-Pallas example. | GPU-focused; good for autodiff demo. |
| [rdyro/moe_in_jax](https://github.com/rdyro/moe_in_jax) | Blog companion | MoE sparse-core kernels. | Small. |
| [sqtian/PALLAS_TPU_KERNEL_MATMUL](https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL) | sqtian.com blog | Step-by-step matmul Pallas optimizations | **87⭐ MIT. Excellent teaching material** for the MXU tiling story. |
| [ashioyajotham/recompute_dont_restore](https://github.com/ashioyajotham/recompute_dont_restore) | "Flash Attention from first principles on TPU" | FA fwd/bwd Pallas | 1⭐ pedagogical. |
| [labyrinth-ssr/tpu-research](https://github.com/labyrinth-ssr/tpu-research) | Research | `tpu_inference_kernel/fused_moe/v1/kernel.py`, megablox gmm | Fork/research of tpu-inference kernels. |

## Tier 5 — Kernel-optimization agents (meta-tooling)

Novel category — these *use* Pallas as an optimization target rather than providing hand-written kernels. Directly relevant to this wiki's autoresearch mission.

- [ucb-bar/autocomp](https://github.com/ucb-bar/autocomp) — "Optimize any AI kernel, anywhere"; ships TPU v5e/v6e agent knowledge plus `examples/jaxbench-pallas/flash_attention*.py`. 123⭐ BSD-3.
- [primatrix/Glaucis](https://github.com/primatrix/Glaucis) (mirror: [sii-xinglong/Glaucis](https://github.com/sii-xinglong/Glaucis)) — **evolutionary search over Pallas kernel variants** (`chunk_gla`, `gmm_fp8_blockwise`). Apache-2.0.
- [aryatschand/JAXBench](https://github.com/aryatschand/JAXBench) — LLM-authored Pallas kernel benchmark — interesting for *evaluating* a kernel-authoring agent, not a source library.
- [Cusp-AI/tonno](https://github.com/Cusp-AI/tonno) — `@autotune` decorator for Pallas, Triton-style.

## Tier 6 — Tutorials & pedagogical repos

- [vorushin/pallas_puzzles](https://github.com/vorushin/pallas_puzzles) — exercises. 20⭐.
- [p-nordmann/pallas-puzzles-2025](https://github.com/p-nordmann/pallas-puzzles-2025) — 2025 fork.
- [nthnluu/gpu-kernel-codelab](https://github.com/nthnluu/gpu-kernel-codelab) — fused softmax, tiled matmul, flash attention.
- [linhkid/pallas-forge](https://github.com/linhkid/pallas-forge) — Neuropurrfect "Why Pallas" Part 3 companion (autotuning). 3⭐ MIT; possibly not the canonical pallas-forge.

External docs (not repos but with substantive inline code):

- [JAX Pallas docs](https://docs.jax.dev/en/latest/pallas/) — quickstart, TPU details, TPU sparse, Mosaic-GPU refs. Source under `jax-ml/jax/docs/pallas/`.
- [JAX Pallas design doc](https://docs.jax.dev/en/latest/pallas/design/design.html) — authoritative reference; no dedicated Pallas paper exists.
- [PyTorch/XLA Pallas feature docs](https://docs.pytorch.org/xla/master/features/pallas.html).
- [MaxText Pallas kernels guide](https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html).
- [Keras custom TPU/GPU kernel guide](https://keras.io/guides/define_custom_kernel/).
- [Henry Ko — Optimizing NSA for TPUs](https://henryhmko.github.io/posts/nsa_tpu/nsa_tpu.html) — DeepSeek Native Sparse Attention ported to TPU v5e; Colab with code.
- [Shanquan Tian — Unlocking TPU Performance with Custom MatMul Kernels](https://sqtian.com/post/pallas_tpu_matmul_kernel/) — companion to sqtian/PALLAS_TPU_KERNEL_MATMUL.
- [Chaim Rand — The Rise of Pallas](https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a/).
- [Robert Dyro — Pallas-Triton kernels and auto-tuning](https://robertdyro.com/articles/pallas-triton_kernels/).
- [gdymind.com — Pallas 101](https://gdymind.com/2026/02/19/Pallas-101/).

## Tier 7 — Marginal / unverified

Skip unless you have a specific need for the named kernel.

- [carlesoctav/llm](https://github.com/carlesoctav/llm) — `src/jaxformers/ops/cross_entropy/pallas_tpu.py`, possibly Levanter-derived.
- [azzeddineCH/flash-nanoGPT](https://github.com/azzeddineCH/flash-nanoGPT) — JAX nanoGPT with Pallas + shmap + jmp. 3⭐.
- [Beomi/dllm-jax](https://github.com/Beomi/dllm-jax) — diffusion-LM on TPU.
- [saran-gangster/nanovllm_jax](https://github.com/saran-gangster/nanovllm_jax) — minimal vLLM in JAX + Mosaic-GPU.
- [stilxam/pallas-jax-flash-attention](https://github.com/stilxam/pallas-jax-flash-attention), [RorroArt/pallas_attn](https://github.com/RorroArt/pallas_attn), [ArjunSom9/Pallas-Unbound](https://github.com/ArjunSom9/Pallas-Unbound), [okiyama/jax-pallas-odes](https://github.com/okiyama/jax-pallas-odes), [AnilKy001/Spiking-Pallas](https://github.com/AnilKy001/Spiking-Pallas), [panalexeu/jax_and_pallas](https://github.com/panalexeu/jax_and_pallas), [2986002971/jax-pallas-learning](https://github.com/2986002971/jax-pallas-learning) — personal / small; skip unless the specific kernel matters.
- [tengyifei/learning_machine](https://github.com/tengyifei/learning_machine) — small personal; Pallas llama_ref.
- `eggduzao/Olympus` — unverified, may contain `pallas/mosaic/core.py`.

## Key papers

Annotated references. Verified links unless noted.

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Foundational; every Pallas flash-attn is a port.
- Dao, *FlashAttention-2* — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context* — [arXiv:2310.01889](https://arxiv.org/abs/2310.01889). Pallas impls in `haoliuhl/ringattention` and maxdiffusion.
- *Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU* — arXiv ID in search result scraping appeared as 2604.15464 (likely typo / mis-scrape); actual ID probably 2504.15464 or similar. Describes the TPU RPA kernel shipped in vllm-project/tpu-inference and jax-ml/jax. Confirm ID before citing.
- DeepSeek, *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089). Henry Ko's TPU-Pallas worklog ports this.
- Lin et al., *AdaSplash: Adaptive Sparse Flash Attention* — [arXiv:2502.12082](https://arxiv.org/abs/2502.12082). Naming-lineage relevance.
- *Block Sparse Flash Attention* — [arXiv:2512.07011](https://arxiv.org/abs/2512.07011). Date in search result is future-dated — verify.
- Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* — [arXiv:2211.15841](https://arxiv.org/abs/2211.15841). Basis for the megablox Pallas kernel.
- Apple, *AxLearn: Modular, Hardware-Agnostic Large Model Training* — [arXiv:2507.05411](https://arxiv.org/pdf/2507.05411). Documents Pallas/Mosaic splash-attention dispatch.
- **No dedicated Pallas-introducing paper exists.** Authoritative references remain the JAX Pallas [docs](https://docs.jax.dev/en/latest/pallas/) and [design doc](https://docs.jax.dev/en/latest/pallas/design/design.html).

## Direct impact on open wiki state

Mapping survey findings back to items already tracked in the wiki:

1. **[sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) § Gaps & caveats, candidate #2** — *"Wire tokamax `ring_attention_kernel` through `dot_product_attention` dispatch — kernel exists, API gap only."* The survey confirms a public reference implementation at [AI-Hypercomputer/maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) `src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py` (integrated with splash-style dispatch). This reduces the hypothesis from "open research" to "port + adapt."

2. **[sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) § Gaps & caveats, candidate #3** — *"Zig-Zag Ring Attention on TPU — no implementation found in any ingested codebase; open algorithmic port from Brandon et al. 2023."* **Partial retraction warranted.** Ring Attention Pallas exists publicly in [haoliuhl/ringattention](https://github.com/haoliuhl/ringattention). Whether the causal-mask load balancing (Zig-Zag specifically) is implemented in either `haoliuhl/ringattention` or `maxdiffusion`'s copy needs to be read; the survey did not verify that level of detail. The broader statement "no ring-attention Pallas on TPU" was wrong.

3. **[sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) § Gaps & caveats, candidate #4** — *"TPU-native Pallas kernels for `gated_linear_unit` and `layer_norm` in tokamax (currently XLA fallback)."* [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) @ tag `v3.0.1` ships a production Pallas **fused GLU kernel** (matmul+GLU with tiling). [marin-community/marin](https://github.com/marin-community/marin) / [levanter](https://github.com/marin-community/levanter) ships a fused CE-loss kernel with an autotune harness that would transfer. Reduces "build from scratch" to "port + autotune". **However**, the log.md `[2026-04-23] analyze | Gemma 4 E4B on v6e-4` entry documents that Pallas RMSNorm lost −8.1% at exp 33 *because XLA already fuses it with neighbor matmuls*. The same caveat applies to GLU — before porting AlphaFold3's GLU, read the HLO to confirm XLA is not already doing the same fusion on your workload.

4. **Gemma 4 program, exp 33 lesson** — *"Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion."* The survey gives external evidence-base support for this lesson: maxtext and tpu-inference *do* hand-write megablox, ragged_paged_attention, and gather_reduce_sc, but *don't* hand-write RMSNorm or SwiGLU. That absence is data — those patterns are left to XLA for the same reason your exp 33 surfaced.

5. **Kernel-optimization agents (Tier 5)** are a direct analogue of this wiki's autoresearch loop. [ucb-bar/autocomp](https://github.com/ucb-bar/autocomp) and [primatrix/Glaucis](https://github.com/primatrix/Glaucis) are worth a shallow read before investing further in the loop's search procedure — may carry useful priors on kernel-space search.

## Gaps the survey could not close

- **NVIDIA Mosaic-GPU work** lives inside `jax-ml/jax` directly (`jax/_src/pallas/mosaic_gpu/`, `jax/experimental/pallas/ops/gpu/*_mgpu.py`). GPU-side ops catalog (`collective_matmul_mgpu`, `transposed_ragged_dot_mgpu`, `decode_attention`, `rms_norm`, etc.) was not enumerated.
- **Closed-source training stacks** — Gemini/Gemma training stacks are likely in private repos; only partial mirrors on GitHub. The `google-deepmind/gemma` repo was not enumerated and may ship Pallas.
- **GoogleCloudPlatform/\*** sample / cookbook repos — likely contain Pallas demos; not enumerated.
- **Anthropic / xAI / Cohere / Character** — confirmed Pallas users per job postings but no public repos found. Follow conference talks and job postings for signals.
- **Ragged Paged Attention paper arXiv ID** — the scrape returned `2604.15464` (likely typo / future-dated). Confirm real ID before citing.
- **`jondeaton/ring-attention-jax-pallas`** — URL returned 404 during verification; confirm.
- **`eggduzao/Olympus`** — flagged as potentially containing `pallas/mosaic/core.py`; unverified.

## Recommended next ingestion wave

If pursuing a "Wave 4 — external Pallas libraries" ingest, my proposed order:

1. **[AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext)** — closest analogue to the current trainer; ingest as a codebase page + subpages per kernel family (`kernels/attention/`, `kernels/megablox/`, `kernels/inference/`).
2. **[vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)** — broadest novel kernel surface; ingest parent codebase page + subpages for paged attention, MLA, sparse-core, gdn, megablox, all-gather-matmul.
3. **[AI-Hypercomputer/maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)** — scope narrowly to `src/maxdiffusion/kernels/splash_attention/` (splash + ring); this is the ring-attention reference impl.
4. **[apple/axlearn](https://github.com/apple/axlearn)** — scope to the kernel subdirectory (`axlearn/common/flash_attention/`, `ssm_kernels/`, `megablock/`, `kv_cache/`, `rattention/kernels/`); parent repo is far too large to ingest whole.
5. **[haoliuhl/ringattention](https://github.com/haoliuhl/ringattention)** — small; ingest as source (paper companion) plus the single-file kernel.
6. **[sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax)** — after #1-3 if `simple_gla` / spec-decoding kernels become relevant.

Deferrable: axlearn SSM/Mamba kernels (only when a non-transformer model enters the loop); jaxite / recurrentgemma / graphcast (niche applications); AlphaFold3 GLU (valuable but needs tag pinning and is narrower in scope than #1-3).

## Sources

- Raw agent research report, 2026-04-23 — conducted via general-purpose subagent using WebSearch + WebFetch + GitHub API.
- [wiki/sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) — hypothesis candidates this survey updates.
- [wiki/log.md](../log.md) — `[2026-04-23] analyze | Gemma 4 E4B on v6e-4 — optimization ceiling reached at exp 25` entry documenting the Pallas-vs-XLA-fusion lesson referenced above.
- [wiki/codebases/tokamax.md](../codebases/tokamax.md) — the anchor library this survey runs against.
- Each repo and paper URL is inlined at its first mention.

## See also

- [codebases/tokamax.md](../codebases/tokamax.md)
- [codebases/torchax.md](../codebases/torchax.md)
- [codebases/scaling-book.md](../codebases/scaling-book.md)
- [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md)
- [concepts/splash-attention.md](../concepts/splash-attention.md)
- [concepts/ring-attention.md](../concepts/ring-attention.md)
- [concepts/flash-attention.md](../concepts/flash-attention.md)
- [concepts/pallas-kernel.md](../concepts/pallas-kernel.md)
- [concepts/mosaic-kernel.md](../concepts/mosaic-kernel.md)
- [concepts/ragged-dot.md](../concepts/ragged-dot.md)
