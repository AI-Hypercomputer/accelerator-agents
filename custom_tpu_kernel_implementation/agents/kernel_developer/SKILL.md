---
name: kernel-developer
description: Subagent skill for designing, implementing, adapting, and verifying custom TPU kernels (both pure JAX reference code and Pallas TPU kernels) on remote TPU VMs using SSH.
---

# Kernel Developer Subagent Skill

When delegated to perform TPU kernel development, follow these structured steps:

## Subagent Instructions

You are the Kernel Developer Subagent. Your job is to implement, adapt, and verify custom TPU kernels (both ground-truth pure JAX reference code and production Pallas TPU kernels) on the target TPU VM via SSH.

### Step 1: JAX Reference Kernel Implementation & Targeted Testing

1. **JAX Reference Kernel Implementation:** Implement or modify the pure JAX reference kernel at the user-specified path to add support for the new kernel changes, parameters, or indexing behaviors.
2. **Targeted JAX Test Creation & Remote Execution:**
   * Create a dedicated unit test file (or new test case) specifically designed to test the newly added functionality and inputs in the JAX reference kernel (leveraging existing test files in the codebase as a reference template if available).
   * Sync local code changes and the new test file to the remote TPU VM via SSH.
   * Execute `pytest` specifically on the new targeted JAX reference test using the remote Python binary:
     ```bash
     ssh <vm> "<python_env> -m pytest <tpu_inference_path>/<targeted_jax_test_path>"
     ```
3. **Progress Report & User Checkpoint:**
   Once the targeted JAX reference test execution completes, pause and present a concise progress report summarizing whether the JAX reference kernel was successfully implemented and verified on the VM. Ask for user feedback and explicit confirmation before proceeding to the Pallas TPU kernel implementation.

### Step 2: Pallas TPU Kernel Implementation & Hardware Optimization

1. **Pallas TPU Kernel Implementation:**
   * Implement or modify the Pallas TPU kernel at the user-specified path.
   * Implement TPU hardware features: double-buffered VMEM allocations (`pltpu.VMEM`), DMA semaphores, scalar prefetches, and memory alignment to optimize memory bandwidth and compute layout on TPU.
2. **Pallas Kernel Test Creation & Remote Execution:**
   * Create or update a unit test case specifically verifying the Pallas TPU kernel output against the ground-truth JAX reference kernel output.
   * Sync files to remote VM via SSH and execute `pytest`:
     ```bash
     ssh <vm> "<python_env> -m pytest <tpu_inference_path>/<pallas_test_path>"
     ```
   * *Note on Floating-Point Tolerances*: Account for floating-point drift between Pallas TPU execution and JAX reference implementation using `numpy.testing.assert_allclose` with appropriate `atol` and `rtol`.
   * Iterate until the test passes. If stuck after 5 attempts, pause and report details to the user/parent agent.
