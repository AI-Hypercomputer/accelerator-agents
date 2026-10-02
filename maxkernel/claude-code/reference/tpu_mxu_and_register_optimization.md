---
name: tpu-mxu-and-register-optimization
description: >-
  Detailed reference for optimizing compute throughput (MXU/VALU), managing Mosaic vector layouts,
  reordering LLO instructions for scheduling, and preventing register spills in JAX/Pallas.
---

# TPU MXU and Register Optimization

This skill covers how to optimize a compute-bound JAX/Pallas kernel by maximizing Matrix Multiply Unit (MXU) occupancy, managing Mosaic vector layouts, and preventing register spills.

---

## 1. How Matrix Multiplies Work on TPUs

The Matrix Multiply Unit (MXU) is a systolic array. Keeping all MXUs busy is critical for achieving peak FLOPs.

*   **Pipelining & Latency**: The MXU behaves like a pipeline. You can queue one matrix multiplication every ~8 cycles on TPUv5. However, **the latency for a single matmul to complete is over 100+ cycles**.
*   **The Hazard**: If you only execute one matmul at a time, the compiler must wait for the systolic array to drain (100+ cycles of idle time) before fetching the result with `vpop`.
*   **The Solution**: Queue multiple matrix multiplications in succession to keep the pipeline full.
*   **Native Sizes & Instructions**: Matmuls are performed in chunks of $(8 \times 128) \times (128 \times 128)$ and use three LLO assembly instructions:
    1.  `vmatpush`: Pushes Vector Registers (VREGs) representing the RHS ("gains matrix", which has a fixed $128 \times 128$ shape) into specialized registers.
    2.  `vmatmul`: Queues the matrix multiplication of the LHS (read from VREG or LHS register) @ RHS to the MXU.
    3.  `vpop`: Transfers the computed result back from the MXU to a VREG.

---

## 2. Mosaic Vector Layouts

Mosaic compiles arrays by tiling their **last 2 dimensions** into physical $8 \times 128$ vector registers (VREGs).
*   > [!IMPORTANT]
    > **The last two dimensions of an array in Pallas are physical.** Operations like reshapes, transposes, and adding/removing axes along these dimensions are NOT free and can trigger expensive layout re-compilations (showing up as `vxpose` or heavy `vst`/`vld` in traces).

### A. Layout Notation
Vector layouts are denoted as:
$$\langle\text{bitwidth}, \text{offset}, \text{tile\_size}, \text{implicit\_dim}\rangle$$
*   *Example*: $\langle32, (0,0), (8,128), -1\rangle$ indicates a 32-bit layout tiled into $8 \times 128$ VREGs with no offset.
*   **Implicit Dimension**: 1D arrays are mapped to a 2D layout:
    *   `implicit_dim=-1`: Tiles the array as shape $(N, 1)$.
    *   `implicit_dim=-2`: Tiles the array as shape $(1, N)$.

### B. Vector Layout "Code Smells" to Avoid
1.  **Singleton trailing dimensions** (e.g., shape `[128, 1, 1]`):
    *   *The Problem*: Only tiles 1 element out of 1024 inside each physical $8 \times 128$ VREG. This wastefully consumes 128 VREGs.
    *   *The Fix*: Omit the trailing dimensions. Store the array as `[1, 128]` or a 1D array of `[128]`, which fits into a single VREG.
2.  **Reshapes and Transposes involving the last two dimensions**:
    *   *The Problem*: Triggers Mosaic to perform an expensive physical re-layout in VMEM.
    *   *The Fix*: Verify in the LLO dump if layout changes are occurring. Keep the last two dimensions stable.
3.  **Explicit transposes before matmuls** (e.g., `A @ B.T`):
    *   *The Problem*: Copying/transposing in VMEM is slow.
    *   *The Fix*: Use `pl.dot(A, B, trans_b=True)`. The TPU transpose values en-route to the MXU, making this operation virtually free.

---

## 3. Diagnosing Compute Issues

*   **Pacchetto**: Check the `vmatmul` track under `mxu/xlu/eup 12`.
    *   If MXU usage is low or you see large empty cycle blocks (e.g., 100+ cycles) between the last `vmatmul` and `vpop`, you are not queuing enough compute to keep the systolic array busy.
*   **LLO Dumps (`*-packed-bundles-post-ra.txt`)**: Locate where the `vmatmul` instructions are queued. Check if they are bundled consecutively.

---

## 4. Scheduling & Overlapping VALU/MXU

The LLO compiler bundles instructions together. The general rule is: **scheduling follows the MXU**. The compiler will not reorder `pl.dot` operations, so their order in your Python code determines the execution schedule.

### A. Overlap VALU and MXU
Do not run a massive block of Vector ALU (VALU) instructions (such as activations, normalizations) followed by a massive block of MXU matmuls.
*   **Action**: Structure code so that activations/reductions are evaluated immediately as soon as the result of a sub-block matmul is ready. This allows the VALU to work on sub-block $N$ while the MXU processes sub-block $N+1$.
*   **Slicing Matmuls**: Break large matmuls into smaller ones, interleaving the vector core math.

### B. Eliminate Optimization Barriers
The compiler can only reorder instructions within a single basic block. It cannot move operations across control flow splits, loop bounds, or `named_scope` blocks.
*   **Bad Pattern**:
    ```python
    def kernel(...):
      body_compute()
      if is_last_iteration:
        output_compute() # Barrier: Compiler cannot merge body and output compute
    ```
*   **Good Pattern (Output Fusion)**:
    ```python
    def kernel(...):
      if is_last_iteration:
        body_compute()
        output_compute() # Combined in one block; compiler can optimize
      else:
        body_compute()
    ```

### C. Loop Unrolling
Unroll loops to remove branch barriers. Pass `unroll=N` when using JAX loops like `lax.scan` to allow the compiler to schedule instructions across iteration boundaries.

---

## 5. Preventing Register Spills

Register spills occur when your kernel requires more live variables than physical registers available (e.g., TPUv5 has 64 VREGs). Live values are dumped to VMEM and re-read, creating massive latency overhead.

*   **Diagnosis**: Look for register pressure alerts in Pacchetto's top track. In the LLO dump, watch out for excessive vector stores (`vst`) and loads (`vld`) inside the compute loop.
*   **Remediation**:
    1.  **Lower Precision**: Cast arrays to `bf16` or `fp8` to fit more elements per register.
    2.  **Optimize Tiling**: Align shapes to $8 \times 128$ boundaries so registers aren't wastefully underpopulated.
    3.  **Adjust Loop Order**: Play with loop schedules to consume values as soon as they are loaded, minimizing the live range of variables.
