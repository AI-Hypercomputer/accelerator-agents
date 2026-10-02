---
name: tpu-profiling-and-diagnostics
description: >-
  Detailed instructions for setting up and interpreting profiling tools
  (Xprof, LLO Dumps) to diagnose JAX/Pallas TPU kernels.
---

# TPU Profiling and Diagnostics

This skill provides comprehensive instructions on setting up and using profiling tools to diagnose bottlenecks in JAX/Pallas TPU kernels.

---

## 1. Profiling Tooling Setup & Flags

Use these configurations to capture execution timelines and instruction dumps.

### A. Xprof: Runtime Profiler
Xprof is the primary tool for analyzing dynamic runtime behaviors, such as memory latency (SyncWait), loop execution times, and pipelining overlap.

https://github.com/openxla/xprof/tree/master/plugin/xprof/cli

#### CLI Flags
*   `--xla_tpu_enable_llo_profiling`: Allows visualization of custom `jax.named_scope` regions in the trace.
    *   *Note*: `jax.named_scope` introduces optimization barriers. Use it to annotate coarse-grained blocks rather than fine-grained instructions.
*   `--xprof_tracemode=TRACE_COMPUTE_AND_SYNC`: Visualizes DMA semaphore and sync flag waits (indicating HBM-VMEM or ICI transfers). Must be passed to the session API, not just as a command-line flag if using the session API.

---

## 2. Reading and Interpreting Profile Traces

Use this guide to diagnose problems from traces.

### A. Diagnosing Memory Bottlenecks (SyncWait)
*   **Where to look**: In the **Xprof/Perfetto** timeline, locate the **`Tensor Core Sync Flag`** track.
*   **What to look for**: **`SyncWait`** blocks.
*   **Interpretation**: If `SyncWait` spans a major portion (e.g., $>50\%$) of the timeline, your TPU cores are idling while waiting for HBM-to-VMEM DMA copies to finish. The kernel is **memory-bound**.
*   **Next Steps**: Go to the memory optimizations guide: [tpu-memory-overlapping](tpu_memory_overlapping.md).

### B. Diagnosing Low Compute Occupancy (MXU Idle Blocks)
*   **Where to look**: inspect the **`vmatmul`** track under the **`mxu/xlu/eup 12`** section.
*   **What to look for**:
    1.  **Overall MXU percentage**: If it is low (e.g. 25%), the hardware matrix multiply units are mostly idle.
    2.  **Draining Gaps**: Look for a large block of empty cycles (e.g., 100+ cycles) between the last `vmatmul` instruction and the first `vpop` instruction.
*   **Interpretation**:
    1.  A long gap indicates that the MXU pipeline is draining because there is not enough queued work. Matrix multiplies must be queued frequently to hide the 100+ cycle systolic array latency.
    2.  If the MXU is underutilized and there is no significant `SyncWait`, the kernel may be ALU-bound (spending too much time on vector ALU math) or bottlenecked by register pressure.
*   **Next Steps**: Go to the compute optimization guide: [tpu-mxu-and-register-optimization](tpu_mxu_and_register_optimization.md).

### C. Identifying Register Spills and Re-layouts
*   **Where to look**: register pressure track, or by reading the LLO dump manually.
*   **What to look for**:
    *   Alarms/red markings on the top register pressure track.
    *   **LLO Dumps**: Search for a large number of `vxpose` instructions, or unexpected vector stores and loads (`vst`/`vld`) to VMEM.
*   **Interpretation**:
    *   `vst` and `vld` occurring during compute blocks (and not during the initial prefetch/final store loops) mean registers are spilling to VMEM due to excessive live variables.
    *   `vxpose` and frequent layout changes mean Mosaic is performing expensive physical relayouts of your tensors in VMEM.
*   **Next Steps**: Go to the vector layout and register optimizations guide: [tpu-mxu-and-register-optimization](tpu_mxu_and_register_optimization.md).
