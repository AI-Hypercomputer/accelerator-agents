---
variant: 8B/v6e-8
hypothesis: qwen3-jax-8k-baseline
commit: HEAD
status: running
verdict: pending
hlo_prefilter:
  status: passed
  notes: Baseline measurement, no custom kernel.
---

# v047: 8k sequence length baseline on v6e-8

## Hypothesis under test
- **Mechanism**: Use the best valid stack (Splash, Tokamax CE, gate-up-fusion) at `bs=1` and `seq=8192` to capture a profiling baseline for 8k context length optimization.
- **Predicted signal**: MFU and an xprof trace.

## Results
(pending)

## Analysis
(pending)
