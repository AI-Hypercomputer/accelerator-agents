---
title: "Capturing profiles"
type: source
provenance: docs
source_doc: docs/capturing_profiles.md
updated: 2026-07-03
status: fresh
---
# Capturing profiles

The central how-to for getting a profile out of a running workload — programmatic capture, on-demand (manual) capture, continuous snapshots, remote-machine and Google Cloud profiling, and multi-session runs.

## Key points
- **Programmatic capture** instruments the workload to emit a profile over a code region ([programmatic-capture](src:docs/capturing_profiles.md#programmatic-capture))
- **On-demand / manual capture** triggers a capture from the XProf UI against a running server ([on-demand-capture-aka-manual-capture](src:docs/capturing_profiles.md#on-demand-capture-aka-manual-capture))
- **Continuous profiling snapshots** periodically capture during a long run ([continuous-profiling-snapshots](src:docs/capturing_profiles.md#continuous-profiling-snapshots))
- Profiling on a remote machine and on Google Cloud; framework-specific instructions link to the JAX/PyTorch-XLA/TensorFlow guides ([framework-specific-instructions](src:docs/capturing_profiles.md#framework-specific-instructions))
- Multiple sessions per run and troubleshooting/privilege issues ([multiple-sessions-per-run](src:docs/capturing_profiles.md#multiple-sessions-per-run))

## Feeds topics
- [Capturing profiles](../topics/capturing-profiles.md)
- [Framework-specific profiling](../topics/framework-profiling.md)

## Sources
- raw/code/xprof/docs/capturing_profiles.md
