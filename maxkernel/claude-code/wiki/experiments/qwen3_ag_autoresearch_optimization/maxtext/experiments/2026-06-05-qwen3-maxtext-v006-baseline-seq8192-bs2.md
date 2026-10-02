---
model: qwen3
lane: maxtext
date: 2026-06-05
version: v006
hypothesis: seq=8192 bs=2 baseline
status: complete
---

# v006

Baseline achieved with `per_device_batch_size=2` and `sa_block_*=1024`. Tokens/s/device: 5355. Step Time: 3.06s. MFU: ~34.9%.
