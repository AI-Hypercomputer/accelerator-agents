---
title: "Ghostfish TPU Training Setup"
type: experiment
tags: [experiment, ghostfish, tpu, xmanager]
hypothesis: "Verify that Ghostfish TPUs can be successfully allocated and utilized for training via XManager with GQM parameters."
model: qwen3coder-480b
created: 2026-05-27
updated: 2026-05-27
commit: ""
verdict: ""
agent: ""
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


This experiment sets up the basic infrastructure to launch a training job on Ghostfish TPUs using XManager and GQM.

## Hypothesis

By correctly specifying the GQM parameters (accounting group `mlacc-gqm-dyn`, resource pool `msca-dynamic`, and cell `yulhrp`), we can successfully allocate 4 Ghostfish TPU chips and execute a JAX-based training script.

## Method

1. Created a simple JAX training script (`train.py`) that detects TPUs and performs a matrix multiplication.
2. Created an XManager launch script (`launch.py`) that uses the specified GQM parameters and packages the training script.
3. Prepared the command line flags for launching via `xmanager CLI`.

## Profile

<!-- TODO: Diagnostic vector output from TPUDiagnoseAgent. Include roofline analysis, headroom report, DMA idle %, HBM BW utilization %, VREG spill count, collective latency breakdown. -->

## Results

<!-- TODO: TPS/chip, p50/p99 TPOT, HBM peak, compile time, eval scores. -->

## Numeric Equivalence

<!-- TODO: Pass/fail, max deviation, failed prompts if any. Write "N/A" for hypothesis classes that don't require numeric-equiv. -->

## Verdict

<!-- TODO: Accept/reject/invalid per the class-appropriate template from the hypothesis taxonomy. -->

## Next hypotheses

<!-- TODO: Any new hypotheses filed as a result of this experiment's profiling or analysis. -->

## Lessons

<!-- TODO: Generalizable findings, constraint identifications, or knowledge gained. -->
