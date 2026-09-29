---
title: "Profiling JAX on GPUs"
type: source
provenance: docs
source_doc: docs/jax_gpu_profiling.md
updated: 2026-07-03
status: fresh
---
# Profiling JAX on GPUs

End-to-end guide for profiling JAX GPU workloads with XProf and ML diagnostics: instrument the workload, containerize, deploy on Kubernetes, and capture/visualize.

## Key points
- Prerequisites and on-demand vs. programmatic capture for JAX-on-GPU ([prerequisites](src:docs/jax_gpu_profiling.md#prerequisites))
- Step 1 instrument the JAX workload; Step 2 containerize (Dockerfile); Step 3 deploy (Kubernetes manifest); Step 4 capture + visualize ([step-1-instrumenting-the-jax-workload](src:docs/jax_gpu_profiling.md#step-1-instrumenting-the-jax-workload))

## Feeds topics
- [Framework-specific profiling](../topics/framework-profiling.md)
- [Deploying XProf](../topics/deployment.md)

## Sources
- raw/code/xprof/docs/jax_gpu_profiling.md
