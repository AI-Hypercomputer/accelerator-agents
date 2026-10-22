PROMPT = """
# General Workflow
The general workflow for optimizing performance of a kernel is to identify bottlenecks and resolve them until the bottleneck is the MXU. As the MXU contains the majority of the FLOP/s capacity of a TPU, once this is achieved it generally means we have good utilization of the hardware. The general steps are:
(1) Determine if the kernel is compute-bound or not. See diagnosis.
    (a) If the kernel is memory-bound, see overlapping communication.
(2) Determine if the kernel is MXU-bound or not. See diagnosis.
    (a) If the kernel is ALU-bound, see optimizing MXU usage.
(3) Optimize with respect to XLA.
    (a) At this point, the kernel itself should be in good shape. The final step is to benchmark the kernel as part of the larger XLA program, and determine if it slows down neighboring ops because XLA cannot perform some optimizations around Pallas kernels. See interaction with XLA for some general guidance.

# Overlapping Communication and Compute
The first step we generally take when optimizing kernels is to make sure that they are compute-bound, so the TPU is not idling while it is waiting for inputs or outputs to be copied. This chapter will cover how to diagnose if this is a problem, and offer guidance on how to make a kernel compute-bound if it is not.

## Diagnosis
Two good choices for diagnosing whether a kernel is memory or compute bound are roofline analysis (theoretical) and Xprof (empirical).

## Using Xprof
Roofline analysis is excellent for rough estimates, but is a purely theoretical tool. Xprof provides a direct, empirical method for inspecting whether the code you wrote is compute bound or not. Fortunately, it’s fairly simple to see when the TPU is idle and waiting for memory transfers to complete. For a guide on how to set up Xprof, see profiling tools.

### Basic Example
Here is an example using the following basic matrix multiplication kernel. Note that in practice it would be a bad idea to insert named_scope operations this frequently inside a kernel, as it introduces optimization barriers, but we do so in this example to illustrate how named scopes appear in the profiler.

```python
def kernel(x_ref, y_ref, o_ref):
  with jax.named_scope("kernel_body"):
    @pl.when(pl.program_id(2) == 0)
    def _():
      o_ref[...] = jnp.zeros_like(o_ref)
    with jax.named_scope("load"):
      x_val = x_ref[...]
      y_val = y_ref[...]
    with jax.named_scope("matmul"):
      result = x_val @ y_val
    with jax.named_scope("store"):
      o_ref[...] += result
```

Let’s run this kernel using a 128x128 block size using fp32 inputs. By roofline analysis, we know that this kernel should be memory-bound. Indeed, the Xprof profile shows a significant amount of time waiting for memory transfer to complete:

We can see that more than half of the time is spent waiting for DMAs/memory transfers (marked as SyncWait under the “Tensor Core Sync Flag” track). This is very poor utilization of the TPU, since it is idle during this time. We can also see our named_scope annotations in the compute section on the XLA TraceMe track.

If we instead run this kernel with a 1024x1024 block size using bf16 inputs, we see that the kernel has become compute-bound as most of the time is spent in the kernel body, and the SyncWait operations are no longer visible.


## How to make a kernel compute-bound
### Pipelining
One of the main optimizations for kernels is to overlap communication and computation, since these can happen asynchronously. The communication here refers generally to one of two things: HBM-VMEM transfers on a single chip, or ICI transfers between chips within a single TPU pod.  The main programming pattern we use to achieve overlap is pipelining - where we “prefetch” the next block of input data while we are computing results on the current block of input data.

Pallas implements an HBM-VMEM pipeline by default which is configured via GridSpec and BlockSpecs. For a primer on HBM-VMEM pipelining, see Pipelining — JAX documentation. Also see Matrix Multiplication — JAX documentation for a basic walkthrough on how to benchmark and optimize a pipelined kernel until it is compute-bound. For distributed pipelining (involving multiple chips), see Distributed Computing in Pallas for TPUs — JAX documentation. 

The most general tips for improving performance of pipelined kernels are:
- **Block sizes** - Increase block sizes for operations which have arithmetic intensity that scales with size, such as matrix multiplication (which we have proved via roofline analysis). Eventually you will be bottlenecked by VMEM limits of the chip or pipeline bubble effects, so you will need to try other tricks.
- **Lower precision** - Use lower-bit data types, such as i8 or bf16 instead of f32. This will reduce memory bandwidth usage by ¼ or ½, respectively, compared to f32, at the cost of precision.
- **Fusion** - perform as many operations as possible while the data is in VMEM before storing a result into HBM, to avoid round-trip memory transfers. This is especially helpful before or after a matrix multiply operation because it easily becomes compute-bound with larger block sizes, and you can “piggyback” other operations around it. 
    - Flash attention (online softmax) is a good example of how to non-trivially rewrite an operation in a way that can be fused.
- **Fetching the same block on consecutive iterations**. The pipeline emitter will skip copying a block if the index_map outputs the same indices as the previous iteration. Try to make sure you access your data in a way that exploits this by setting the grid and index_maps accordingly.
    - An example of this is the output matrix in a matrix multiplication. If your inner loop (minor-most grid dimension) is over the contracting dimension, then the output block index should not change until the entire inner loop is computed. Therefore, we only need to fetch and store each output block once.

Other less common techniques to consider include:
- Increase the number of pipeline stages. On TPUs, double-buffering is generally enough for HBM-VMEM pipelining of matrix multiplication because the latency of a DMA is usually smaller than the amount of time to perform a computation block (as opposed to GPUs where latencies are longer). However, it is possible to manually implement a pipeline using DMAs for both HBM-VMEM pipelining as well as in distributed kernels (where we do not yet have an automatic pipeline emitter).
- For sparse kernels, additional preprocessing may be necessary to help the pipeline emitter prefetch the next block of data. See the sparse kernels tutorial for more details.

### Pipeline Bubbles
Increasing block sizes and additional pipeline stages will not always help performance of a kernel due to pipeline bubbles, unless the input size is also large. In any pipelined kernel, there is always a “bubble” at the beginning and end of a kernel while we are waiting for the initial/final memory transfers to complete.

In shorter pipelines and in pipelines with more stages (although on TPUs we usually only have a fixed 2 stages), the bubble becomes a larger overall percentage of the total runtime. Larger block sizes, which generally help with making a kernel compute-bound, unfortunately shorten the pipeline and could actually make the problem worse. In these cases, the kernel will perform better when the total size of the input is also large in order to minimize the effect of the bubble.
"""
