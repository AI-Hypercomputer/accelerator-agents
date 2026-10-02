# maxtext experiments — Llama 3 8B autoresearch

Dated experiment pages for the MaxText reference stack. Each page follows the SCHEMA `experiment` template (see [SCHEMA.md](../../../../../SCHEMA.md)).

The MaxText baseline serves as the reference ceiling that the [torchax sibling](../../torchax/experiments/README.md) and [native-JAX sibling](../../jax/experiments/README.md) target. **As of 2026-04-27 the native-JAX stack has exceeded this baseline by +8.9 % per-chip** at bs=4 seq=8192 (~7,700/chip 43.3 % MFU mean, peak 7,768/43.6 %); the 1.0 pp reported MFU gap is FLOP-counter normalization (under MaxText's accounting we measure 49.0 % MFU, +4.4 pp above their 44.6 %). See the [native-JAX frontier writeup](../../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md).

## Index

- [2026-04-25 — MaxText Llama3.1-8B v6e-8 reference baseline](2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) — **supported** — 409.4 TFLOP/s/device, 7,069.7 Tokens/s/device, 44.6 % MFU at fsdp=8, seq=8192, bs=3 (`tpu-recipes-v0.1.4` + `jax0.6.1-rev1`). Reproduces recipe README's published 413.4 / 7,138.9 within −1.0 %.
