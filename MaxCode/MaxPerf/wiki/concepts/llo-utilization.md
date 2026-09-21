<!-- disableFinding("vice versa") -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "LLO Utilization"
type: concept
tags: [profiling, kernels, bottlenecks, custom-calls]
created: 2026-04-22
updated: 2026-07-13
sources: 1
---

The **LLO (Low-Level Op) Utilization** track is a visualization in the XProf Trace Viewer that displays hardware-resource usage inside TPU custom calls (including Pallas and Mosaic kernels). Since custom calls are natively compiled outside of standard XLA fusion passes, they appear as opaque blocks by default. Inspecting LLO tracks is critical for diagnosing performance bottlenecks inside custom low-level kernels.

---

## 1. Enabling LLO Instrumentation

Because capturing LLO-level debug data increases profile trace sizes and introduces minor collection overhead, it is disabled by default. You must enable it by passing the following flags to the TPU compiler before launching your JAX workload:

```shell
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
python your_serving_workload.py
```

*   `--xla_enable_custom_call_region_trace=true`: Traces regions and steps containing custom calls.
*   `--xla_xprof_register_llo_debug_info=true`: Compiles and registers LLO debug mappings so XProf can link instruction execution to Trace Viewer tracks.

---

## 2. Navigating the LLO Track in Trace Viewer

Once enabled, a new **LLO Utilization** track appears per TPU core/device executing the custom call:

1.  Open the **Trace Viewer** in TensorBoard/XProf.
2.  Locate the device execution timeline. Underneath the main JAX/XLA operation blocks, expand the custom call region.
3.  The **LLO Utilization Line** displays a time-varying utilization percentage, broken down by execution unit (MXU vs. VPU) and memory operations.

---

## 3. Diagnosing Model Bottlenecks Using LLO Data

By analyzing the LLO tracks, you can determine why a kernel is underperforming and how it relates to the hardware roofline:

### A. Memory-Bound Stalls (DMA and VMEM)
*   **Symptom**: LLO track shows low Vector/Matrix execution unit (VPU/MXU) utilization (< 15%) but high DMA/VMEM activity or idle gaps.
*   **Cause**: The kernel is waiting for data to copy from HBM to VMEM, or is blocked by memory bank conflicts inside VMEM.
*   **Resolution**:
    *   Implement double-buffering or pipelining using async copies (`pltpu.async_copy`) to overlap memory transfer with compute execution.
    *   Verify tensor dimension alignments (e.g., ensuring vector loads are aligned to 128-element sublane boundaries).

### B. Compute-Bound Saturation
*   **Symptom**: LLO track shows stable, high MXU utilization (> 70%) for the duration of the custom call.
*   **Cause**: The operation is fully utilizing the systolic matrix unit, matching the predicted hardware compute roofline.
*   **Resolution**: Keep the layout as-is; it is already near-optimal. Any further gains must come from changing numerical precision (e.g., switching from BF16 to FP8).

### C. Pipeline Stage Imbalance (bubbles)
*   **Symptom**: Alternating high utilization peaks between MXU (matrix operations) and VPU (vector calculations, e.g., softmax/activations).
*   **Cause**: Pipeline stages are serialized; the MXU sits idle while the VPU completes vector reductions, or vice versa.
*   **Resolution**: Re-size the thread tile blocks (using algebraic pipeline stage balancing) so that both stages take equal wall-clock execution time per loop iteration.

---

## See also

- [XLA Custom Call](custom-call.md)
- [Trace Viewer](trace-viewer.md)
- [Trace Event Categories](trace-event-categories.md)
- [Pallas Kernel](pallas-kernel.md)
- [Mosaic Kernel](mosaic-kernel.md)
- [Memory Hierarchy](memory-hierarchy.md)
- [Roofline Model](roofline-model.md)

## Sources

- [xprof Custom Call Profiling](../sources/2026-xprof-custom-call-profiling.md) — `raw/code/xprof/docs/custom_call_profiling.md`
- [Ch 9 — Profiling TPU Programs](../sources/2025-scaling-book-ch9-profiling.md) — `raw/code/scaling-book/profiling.md`

