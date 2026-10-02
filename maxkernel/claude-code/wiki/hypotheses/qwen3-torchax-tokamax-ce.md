---
title: "Qwen3 torchax — tokamax streamed cross-entropy"
type: hypothesis
model: qwen3-cc-torchax
variants: ["8B/v6e-8"]
status: open
expected_gain: "HBM unlock at seq 8192 (~2.5 GiB/chip logits saved)"
confidence: medium
effort: M
origin: 2026-06-02-qwen3-torchax-v6e8-baseline
hlo_prefilter: pending
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 torchax — tokamax streamed cross-entropy

*Hypothesis*: Replacing the plain `torch.nn.functional.cross_entropy` (which
materializes `[B, L, V=151936]` logits at the lm_head) with
`tokamax.linear_softmax_cross_entropy_loss` (mosaic_tpu / chunked_xla — streams
logsumexp over V) frees ~2.5 GiB/chip at seq 8192, removing the lm_head memory
wall that blocks longer seq / bigger batch.

*Mechanism*: Skip the lm_head projection in `model_fn`; compute the loss from
`hidden` + `lm_head.weight` inside the streamed CE kernel (shard_map over fsdp).
Qwen3-8B has `tie_word_embeddings=False`, so `lm_head.weight` is a distinct
sharded param available in the weights dict.

*Predicted signal*: peak HBM drops at fixed shape; seq 8192 (or larger batch at
seq 2048-4096) becomes runnable.

*Falsification criterion*: no HBM reduction vs the plain CE (XLA already fuses the
logit+softmax favorably — check HLO pre-filter first) → refuted with
`xla-already-fuses`.
