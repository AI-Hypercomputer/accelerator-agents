# qwen3_cc5 / maxtext — lane log

## [2026-06-12] start | maxtext lane opened (user-directed reference benchmark)

**Op**: start
**Notes**: user asked to run the closest MaxText recipe for a similar-size Qwen3, analyze it, and close the gap on the jax lane. No tpu-recipes training recipe exists for Qwen3 dense on Trillium; using MaxText's own `qwen3-8b.yml` (architecture identical to our jax lane) with the llama3-8B v6e-8 recipe tuning (validated at 44.6% MFU on this cluster, 2026-04-25). mt-v001-base dispatched on the existing tsij-2026-06-02 runner image.

## [2026-06-12] loop-iteration | mt-v001 baseline SUPPORTED (attempt-b): 3.535 s / 416.5 TFLOP/s/dev / 6,953 tok/s/chip @ bs3/seq8192 — llama3-recipe parity on qwen3-8b. Attempt-1: SC-offload flags compile-fatal on v6e-8 (HasSparseCoreLayout RET_CHECK). Gap analysis filed; port = jax-lane v024
