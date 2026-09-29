---
title: Qwen3-8B maxtext Retrospective
date: 2026-06-09
model: Qwen3-8B
lane: maxtext
author: Antigravity
---

# Qwen3-8B maxtext Retrospective

## Context
The goal of this exploration was to port the Qwen3-8B model to the `maxtext` lane and achieve >45% Model Flops Utilization (MFU) on a `v6e-8` slice. While a `llama3-8b` recipe was provided as a baseline, the Qwen3-8B architecture deviates significantly, most notably in its immense vocabulary size (151,936 tokens compared to Llama3's ~32,000).

## Explored Search Space
The tuning process required exploring the intersection of distributed parallelism strategies, memory offloading, and batch sizing due to the `31.25 GB` HBM limit per chip on v6e.

1. **Pure FSDP Baseline:** We verified that `ici_fsdp_parallelism=-1` completely shards the model weights and fits comfortably when the batch size is very small (`bs=2`). However, this yields a low arithmetic intensity and an MFU of `~29.7%`.
2. **Tensor Parallelism:** Introducing 2D parallelism via Tensor Parallelism (`ici_tensor_parallelism=2` / `ici_tensor_parallelism=4`) was catastrophic. The high interconnect communication overhead needed for the massive vocabulary dimension decimated MFU to `~13%`.
3. **Sequence Parallelism:** Attempting to process larger batches (`bs=8`) via sequence parallelism introduced a large temporary buffer footprint during Ring Attention communication. This immediately blew up the XLA compiler memory budget (`CompileTimeHbmOom`), demanding over `58.7 GB` of HBM space.
4. **Vocabulary Tiling:** By splitting the `152k` vocabulary across `num_vocab_tiling=16` slices, the intermediate logit buffers generated during the loss computation were reduced safely.
5. **Batch Size limits under FSDP:** Pure FSDP *does not shard* the activation memory (e.g. `decoder_layer_input`). Thus, attempting `bs=8` demanded `18 GB` for activations. Added to the `10.6 GB` needed for vocabulary parameter/optimizer states, the compiler OOMed at `35.60 GB`.

## The Breakthrough
The pivotal mechanism was `optimizer_memory_host_offload=True`.

Because the Qwen3 vocabulary is so large, the Adam optimizer momentum states consume a massive percentage of the HBM per chip. By transferring these states to the host CPU, the persistent memory footprint on the TPU dropped from `10.67 GB` to `3.61 GB`.

This freed up `~7 GB` of HBM!

Furthermore, because the optimizer step occurs asynchronously after the backward pass, offloading it over PCIe does not stall the dense matrix computations in the MXUs.

Combining this with `bs=4`, we achieved an optimal density that perfectly filled the HBM budget and achieved **51.65% MFU (237 TFLOP/s/device)**.

## Conclusion
The `maxtext` lane for Qwen3-8B is successfully stabilized at >45% MFU. The tuning has verified that for models with extraordinarily large vocabularies, optimizer offloading is an invaluable zero-overhead mechanism for freeing up HBM capacity on v6e to increase batch size arithmetic intensity.
