# maxtext experiments — Gemma 4 E4B autoresearch

Dated experiment pages for the MaxText reference stack. Each page follows the SCHEMA `experiment` template (see [SCHEMA.md](../../../../../SCHEMA.md)).

## Index

- [2026-04-25 — MaxText Gemma 4 E4B v6e-8 baseline](2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md) — **supported** (with approximation caveat) — 282.9 TFLOP/s/device, 10,003 Tokens/s/device, **30.8 % MFU** at `bs=2 seq=8192 fsdp=8 remat=full`. Authored wiki-local `gemma4-e4b.yml` config (no upstream MaxText E4B exists) + Pydantic literal patch. **Approximation**: `num_kv_shared_layers=18` not implemented in MaxText (+47M extra params, ~0.6 % over true E4B).
