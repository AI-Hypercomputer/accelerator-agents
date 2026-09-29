---
name: pallas-xla-interaction
description: >-
  Detailed reference for managing how JAX/Pallas kernels interact with the XLA compiler,
  covering manual fusion, cost estimation overrides, and asynchronous execution across HLO boundaries.
---

# Pallas and XLA Interaction

Pallas kernels compile to "TPU Custom Calls" in the XLA High-Level Optimizer (HLO) graph. Because these custom calls are opaque to XLA, they block standard graph-level optimizations. This skill describes techniques to integrate Pallas kernels smoothly with XLA.

---

## 1. Manual Fusion API

*   **Problem**: XLA cannot automatically fuse surrounding operations (like subsequent elementwise additions or scalar scaling) into Pallas kernels.
*   **Input Fusion**: While `allow_input_fusion` can be specified in `TPUCompilerParams`, it is not guaranteed.
*   **Output Fusion**: Output fusion is never supported automatically by XLA and must be coded manually.
*   **Solution**: Use the manual fusion API to combine operations within a single compilation scope:
    *   Fuser API module: `https://github.com/jax-ml/jax/tree/main/jax/_src/pallas/fuser/`
---

## 2. Cost Estimates

*   **Problem**: The XLA scheduler defaults to assuming a Pallas custom call has a cost of zero. This discourages it from scheduling other operations (such as HBM communication prefetches) to run concurrently with the Pallas kernel.
*   **Solution**: Pass a `pl.CostEstimate` object to `pl.pallas_call` or `pl.pallas_call_p`. This informs XLA's scheduler of the actual computational and memory weight of the kernel.

### A. How to Construct Cost Estimates
There are three ways to generate cost estimates:

1.  **Manual Calculation**: Manually calculate FLOPs and bytes accessed and fill the object fields.
2.  **XLA Compilation Analysis (Most Accurate)**:
    Compile a JAX reference implementation using standard XLA and retrieve the compiler's cost analysis:
    ```python
    # lower and compile a reference JAX function
    lowered = jax.jit(reference_func).lower(*args)
    compiled = lowered.compile()
    cost_analysis = compiled.cost_analysis()
    
    # Extract metrics to feed pl.CostEstimate
    flops = cost_analysis.get("flops", 0)
    bytes_accessed = cost_analysis.get("bytes_accessed", 0)
    ```
    *   *Warning*: This invokes full compilation and can run out of memory (OOM) for very large input sizes.
3.  **Pallas Cost Estimate Helper**:
    Use Pallas' built-in helper. It is less accurate because it analyzes the code before optimizations, but works on any size:
    ```python
    from jax.experimental.pallas import cost_estimate as pl_cost

    cost = pl_cost.estimate_cost(reference_func, *args_shapes_dtypes)
    # Returns an estimate containing .flops and .bytes_accessed
    ```

---

## 3. Async Kernels (HLO-level Overlapping)

*   **Concept**: Overlap the memory transfers of a Pallas kernel with other unrelated HLO operations in the wider JAX program, rather than overlapping communication and compute *inside* the Pallas kernel itself.
*   **Implementation**:
    1.  Split the kernel execution. Create one kernel to start the DMA communication.
    2.  Return the DMA semaphores back to the main JAX/XLA program.
    3.  Control passes back to XLA, which executes other program blocks concurrently while the DMA completes.
    4.  Invoke a subsequent Pallas kernel that waits on the semaphores and executes the compute block.
*   **Resource**: See *Pallas Async Operations* in JAX documentation for API patterns.
