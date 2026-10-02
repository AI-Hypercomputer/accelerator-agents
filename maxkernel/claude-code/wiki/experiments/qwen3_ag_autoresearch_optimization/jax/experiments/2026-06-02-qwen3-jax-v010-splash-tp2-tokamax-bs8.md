---
status: "pending"
---

# 2026-06-02-qwen3-jax-v010-splash-tp2-tokamax-bs8

## Hypothesis
Using Tokamax CE with TP=2 and Splash attention will enable scaling the batch size to 8 (from 4). Tokamax's Pallas kernel natively supports tensor parallelism (`tp`). Since the base image uses TP=2, `tokamax.linear_softmax_cross_entropy_loss` will work out-of-the-box and provide the necessary memory relief to fit `bs=8`.

## Mechanism
- Removed the standard `optax.softmax_cross_entropy_with_integer_labels` and explicit `lm_head` dot product.
- Replaced with `tokamax.linear_softmax_cross_entropy_loss`.
- Kept `absl.flags.FLAGS(sys.argv[:1])` eager initialization for `tokamax`.

## Configuration
- Model: Qwen3-8B
- Parallelism: FSDP with TP=2
- Attention: Splash
- Loss: Tokamax Cross Entropy
- Batch Size: 8

## Setup Steps
1. Based on image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v008-splash-tp2-bs4`
2. Edited `train.py` inside `.repo/2026-06-02-v010-splash-tp2-tokamax-bs8` to switch to Tokamax CE.
3. Built and pushed Docker image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v010-splash-tp2-tokamax-bs8`

## Training Run
(To be executed and validated)
