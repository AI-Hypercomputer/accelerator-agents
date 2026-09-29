---
title: "Source: training/ironwood/README.md"
type: source-page
doc: training/ironwood/README.md
updated: 2026-07-03
---
# training/ironwood/README.md

The hub page for every Ironwood (TPU7x) training recipe, and the single place the repo publishes
measured performance numbers for that hardware: a table of model ID × chip count × global batch
size × sequence length × precision → step time, TFLOPs/sec/chip, and tokens/sec/chip
([training-recipes-for-ironwood-tpu](src:training/ironwood/README.md#training-recipes-for-ironwood-tpu)).
It covers DeepSeek-V3-671B, Gemma4 (2B/4B/26B/31B), GPT-OSS-120B, Llama3.1 (70B/405B), Qwen3-235B-A22B,
and Wan2.1-14B, each measured in at least one precision (bf16, and fp8_full for several) and at least
one chip count (64/128/256 chips), with several models measured at two chip counts to show
weak-scaling behavior.

## Key data points

| Model | Chips | GBS | Seq len | Precision | Step time (s) | TFLOPs/s/chip | Tokens/s/chip |
|---|---|---|---|---|---|---|---|
| deepseek-v3 | 128 | 2048 | 4096 | bf16 | 27.02 | 607.53 | 2,425.75 |
| deepseek-v3 | 128 | 2048 | 4096 | fp8_full | 22.47 | 730.60 | 2,917.15 |
| deepseek-v3 | 256 | 4096 | 4096 | bf16 | 26.79 | 612.66 | 2,446.25 |
| deepseek-v3 | 256 | 4096 | 4096 | fp8_full | 22.08 | 743.46 | 2,968.49 |
| llama3.1-405b | 256 | 1536 | 8192 | bf16 | 98.34 | 1,261.40 | 499.77 |
| llama3.1-405b | 256 | 1536 | 8192 | fp8_full | 64.37 | 1,927.15 | 763.54 |
| llama3.1-70b | 64 | 256 | 8192 | bf16 | 12.20 | 1,207.05 | 2,686.98 |
| llama3.1-70b | 64 | 256 | 8192 | fp8_full | 7.93 | 1,854.19 | 4,127.56 |
| qwen3-235b-a22b | 256 | 8192 | 4096 | bf16 | 30.87 | 629.79 | 4,245.89 |
| qwen3-235b-a22b | 256 | 8192 | 4096 | fp8_full | 27.67 | 702.60 | 4,736.72 |

fp8_full consistently cuts step time by roughly 20-35% relative to bf16 at matched chip
count/batch/sequence length across every model pair measured (deepseek-v3, llama3.1-405b,
llama3.1-70b, qwen3-235b-a22b) — see the full table in the source for the remaining rows (gemma4,
gpt-oss-120b, wan2.1-14b, and the 131072-sequence-length llama3.1-70b variants).

## Feeds
- [Ironwood TPU7x training recipe matrix](../topics/ironwood-training-recipes.md) — this table *is*
  the performance ground truth that topic's recipe-matrix rows point back to.
