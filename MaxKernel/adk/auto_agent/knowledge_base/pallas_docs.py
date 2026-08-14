PROMPT = r'''
Pallas: a JAX kernel language Pallas Quickstart .ipynb .pdf 
# Pallas Quickstart 

## Contents 
Hello world in Pallas Pallas programming model Grids by example Grid semantics Block specs by example 
# Pallas Quickstart # 

Pallas is an extension to JAX that enables writing custom kernels for GPU and TPU.
Pallas allows you to use the same JAX functions and APIs but operates at a lower level of abstraction. 

Specifically, Pallas requires users to think about memory access and how to
divide up computations across multiple compute units in a hardware accelerator.
On GPUs, Pallas lowers to Mosaic GPU, and on TPUs, Pallas lowers to Mosaic. 

Let’s dive into some examples. 

Note: Pallas is still an experimental API and you may be broken by changes! 

Note: when using the Mosaic GPU backend, only Hopper and newer GPUs are
supported. 

Note: there also exists a Triton backend on GPU, but it is maintained only on
a best-effort basis, and is not recommended for use. The Triton backend
supports GPUs down to Ampere. 

## Hello world in Pallas # 

```python
from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
```

We’ll first write the “hello world” in Pallas, a kernel that adds two vectors. 

```python
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y
```

Ref types 

Let’s dissect this function a bit. Unlike most JAX functions you’ve probably written,
it does not take in jax.Array s as inputs and doesn’t return any values.
Instead, it takes in Ref objects as inputs, which represent mutable buffers in memory.
Note that we also don’t have any outputs but we are given an o_ref , which corresponds
to the desired output. 

Reading from Ref s 

In the body, we are first reading from x_ref and y_ref , indicated by the [...] (the ellipsis means we are reading the whole Ref ;
alternatively we also could have used x_ref[:] ).
Reading from a Ref like this returns a jax.Array . 

Writing to Ref s 

We then write x + y to o_ref .
Mutation has not historically been supported in JAX – jax.Array s are immutable! Ref s are new (experimental) types that allow mutation under certain circumstances.
We can interpret writing to a Ref as mutating its underlying buffer. 

Indexing and Slicing Ref s with .at 

In addition to accessing the entire underlying buffer through a reference, it
is possible to also access only a slice by using the .at property. Using x_ref.at[slice] does not immediately read or write data; it
creates a new Ref object that points to a slice of the original buffer. For
example ref.at[0:128] creates a view of the first 128 elements; ref.at[::2] creates a strided view. 

Once you have a new Ref that represents a slice you can read it or write to it
with the usual syntax. Here is a simple example: 

```python
def add_sliced_kernel(x_ref, y_ref, o_ref):
  small_mid = x_ref.shape[0] // 2

  x_left = x_ref.at[:small_mid]
  x_right = x_ref.at[small_mid:]
  y_left = y_ref.at[:small_mid]
  y_right = y_ref.at[small_mid:]

  # The output shape is (4*small_mid).
  large_mid = 2*small_mid
  o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
  o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
  o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
  o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]
```

Note that using x_ref.at[slice][...] is equivalent to x_ref[slice] . The .at is useful if you want to compose multiple slices (e.g. x_ref.at[block_slice][thread_slice] ) or if need to pass a slice to a subkernel
function that takes a Ref . 

So we’ve written what we call a “kernel”, which we define as a program that will
run as an atomic unit of execution on an accelerator,
without any interaction with the host.
How do we invoke it from a JAX computation?
We use the pallas_call higher-order function. 

```python
@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
add_vectors(jnp.arange(8), jnp.arange(8))
```

```python
Array([ 0,  2,  4,  6,  8, 10, 12, 14], dtype=int32)
```

pallas_call lifts the Pallas kernel function into an operation that can be called
as part of a larger JAX program. But, to do so, it needs a few more details.
Here we specify out_shape , an object that has a .shape and .dtype (or a list
thereof). out_shape determines the shape/dtype of o_ref in our add_vector_kernel . 

pallas_call returns a function that takes in and returns jax.Array s. 

What’s actually happening here? 

Thus far we’ve described how to think about Pallas kernels but what we’ve actually
accomplished is we’re writing a function that’s executed very close to the compute units
since values are loaded into the innermost (fastest) portion of the memory hierarchy. 

On GPU, x_ref corresponds to a value in high-bandwidth memory (HBM) and when
we do x_ref[...] we are copying the value from HBM into static RAM (SRAM)
(this is a costly operation generally speaking!).
We then use GPU vector compute to execute the addition, then copy the resulting value
in SRAM back to HBM. 

On TPU, we do something slightly different. Before the kernel is ever executed,
we fetch the value from HBM into SRAM. x_ref therefore corresponds to a value in
SRAM and when we do x_ref[...] we are copying the value from SRAM into a register.
We then use TPU vector compute to execute the addition, then copy the resulting
value back into SRAM. After the kernel is executed, the SRAM value is copied back into HBM. 

We are in the process of writing backend-specific Pallas guides. Coming soon! 

## Pallas programming model # 

In our “hello world” example, we wrote a very simple kernel.
It takes advantage of the fact that our 8-sized arrays can comfortably fit inside
the SRAM of hardware accelerators.
In most real-world applications, this will not be the case! 

Part of writing Pallas kernels is thinking about how to take big arrays that
live in high-bandwidth memory (HBM, also known as DRAM) and expressing computations
that operate on “blocks” of those arrays that can fit in SRAM. 

### Grids by example # 

To automatically “carve” up the inputs and outputs, you provide a grid and BlockSpec s to pallas_call . 

A grid is a tuple of integers (e.g. () , (2, 3, 4) , or (8,) ) that specifies
an iteration space.
For example, a grid (4, 5) would have 20 elements: (0, 0), (0, 1), ..., (0, 4), (1, 0), ..., (3, 4) .
We run the kernel function once for each element, a style of single-program
multiple-data (SPMD) programming. 



A 2D grid 

When we provide a grid to pallas_call , the kernel is executed as many times
as prod(grid) . Each of these invocations is referred to as a “program”.
To access which program (i.e. which element of the grid) the kernel is currently
executing, we use program_id(axis=...) .
For example, for invocation (1, 2) , program_id(axis=0) returns 1 and program_id(axis=1) returns 2 . 

Here’s an example kernel that uses a grid and program_id . 

```python
def iota_kernel(o_ref):
  i = pl.program_id(0)
  o_ref[i] = i
```

We now execute it using pallas_call with an additional grid argument.
On GPUs, we can call the kernel directly like so: 

```python
# GPU version
def iota(size: int):
  return pl.pallas_call(iota_kernel,
                        out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                        grid=(size,))()
iota(8)
```

```python
Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
```

TPUs distinguish between vector and scalar memory spaces and in this case the
output must be placed in scalar memory ( MemorySpace.SMEM ) since i is
a scalar. For more details read TPU and its memory spaces .
To call the above kernel on TPU, run: 

```python
# TPU version
from jax.experimental.pallas import tpu as pltpu

def iota(size: int):
  return pl.pallas_call(iota_kernel,
                        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                        out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
                        grid=(size,))()
iota(8)
```

### Grid semantics # 

On GPUs, each program is executed in parallel on separate threads.
Thus, we need to think about race conditions on writes to HBM.
A reasonable approach is to write our kernels in such a way that different
programs write to disjoint locations in HBM to avoid these parallel writes.
On the other hand, parallelizing the computation is how we can execute
operations like matrix multiplications really quickly. 

In contrast, TPUs operate like a very wide SIMD machine.
Some TPU models contain multiple cores, but in many cases a TPU can be
treated as a single-threaded processor. The grid on a TPU can be
specified in a combination of parallel and sequential dimensions, where sequential
dimensions are guaranteed to run serially. 

You can read more details at grid, a.k.a. kernels in a loop and Noteworthy properties and restrictions . 

### Block specs by example # 

With grid and program_id in mind, Pallas provides an abstraction that
takes care of some common indexing patterns seen in a lot of kernels.
To build intuition, let’s try to implement a matrix multiplication. 

A simple strategy for implementing a matrix multiplication in Pallas is to
implement it recursively.
We know our underlying hardware has support for small matrix multiplications
(using GPU and TPU tensorcores), so we just express a big matrix multiplication
in terms of smaller ones. 

Suppose we have input matrices \(X\) and \(Y\) and are computing \(Z = XY\) .
We first express \(X\) and \(Y\) as block matrices. \(X\) will have “row” blocks
and \(Y\) will have “column” blocks. 
\[\begin{split}
\begin{align*}
X = \begin{bmatrix}
X_0 \\ X_1
\end{bmatrix}
\end{align*}
\end{split}\] \[
\begin{align*}
Y = \begin{bmatrix}
Y_0 & Y_1
\end{bmatrix}
\end{align*}
\] \[\begin{split}
\begin{align*}
Z &=
\begin{bmatrix}
X_0 \\ X_1
\end{bmatrix}
\begin{matrix}
\begin{bmatrix}
Y_0 & Y_1
\end{bmatrix}
\\
~
\end{matrix}
\\
&=
\begin{bmatrix}
X_0 Y_0 & X_0 Y_1 \\
X_1 Y_0 & X_1 Y_1
\end{bmatrix}
\end{align*}
\end{split}\] 
Our strategy is that because \(Z\) is also a block matrix, we can assign each of
the programs in our Pallas kernel one of the output blocks.
Computing each output block corresponds to doing a smaller matrix multiply
between a “row” block of \(X\) and a “column” block of \(Y\) . 

To express this pattern, we use BlockSpec s. A BlockSpec specifies a block
shape for each input and output, and an “index map” function, that maps a
set of program indices to a block index. 



A visualization of a BlockSpec 

For a concrete example, let’s say we’d like to multiply two (1024, 1024) matrices x and y together to produce z , and would like to parallelize
the computation 4 ways. We split up z into 4 (512, 512) blocks where
each block is computed with a (512, 1024) x (1024, 512) matrix multiplication.
To express this, we’d first use a (2, 2) grid (one block for each program). 

For x , we use BlockSpec((512, 1024), lambda i, j: (i, 0)) – this
carves x up into “row” blocks.
To see this, see how both program instances (1, 0) and (1, 1) pick the (1, 0) block in x .
For y , we use a transposed version BlockSpec((1024, 512), lambda i, j: (0, j)) .
Finally, for z we use BlockSpec((512, 512), lambda i, j: (i, j)) . 

These BlockSpec s are passed into pallas_call via in_specs and out_specs . 

For more detail on BlockSpec s see BlockSpec, a.k.a. how to chunk up inputs . 

Underneath the hood, pallas_call will automatically carve up your inputs and
outputs into Ref s for each block that will be passed into the kernel. 

```python
def matmul_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]

def matmul(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j),
    )
  )(x, y)
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (1024, 1024))
y = jax.random.normal(k2, (1024, 1024))
z = matmul(x, y)
np.testing.assert_allclose(z, x @ y)
```

Note that this is a very naive implementation of a matrix multiplication but
consider it a starting point for various types of optimizations.
Let’s add an additional feature to our matrix multiply: fused activation.
It’s actually really easy! Just pass a higher-order activation function into the kernel. 

```python
def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
  z_ref[...] = activation(x_ref[...] @ y_ref[...])

def matmul(x: jax.Array, y: jax.Array, *, activation):
  return pl.pallas_call(
    partial(matmul_kernel, activation=activation),
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j)
    ),
  )(x, y)
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (1024, 1024))
y = jax.random.normal(k2, (1024, 1024))
z = matmul(x, y, activation=jax.nn.relu)
np.testing.assert_allclose(z, jax.nn.relu(x @ y))
```

To conclude, let’s highlight a cool feature of Pallas: it composes with jax.vmap !
To turn this matrix multiplication into a batched version, we just need to vmap it. 

```python
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (4, 1024, 1024))
y = jax.random.normal(k2, (4, 1024, 1024))
z = jax.vmap(partial(matmul, activation=jax.nn.relu))(x, y)
np.testing.assert_allclose(z, jax.nn.relu(jax.vmap(jnp.matmul)(x, y)))
```

previous 

Pallas: a JAX kernel language 

next 

Software Pipelining 
Contents Hello world in Pallas Pallas programming model Grids by example Grid semantics Block specs by example 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Software Pipelining .ipynb .pdf 
# Software Pipelining 

## Contents 
Memory Hierarchies Pipelining Basics Deriving a Double-Buffered Pipeline Pallas Pipelining API Grid BlockSpecs Kernel Pallas Call Example - Elementwise Kernel revisited Parameterizing a Kernel Sharp edges Buffer Revisiting Reductions and accumulation Analyzing the performance 
# Software Pipelining # 

Software pipelining is an important technique in performance optimization by overlapping multiple asynchronous operations even if there are data dependencies between them. In the context of kernel writing, the most common form of pipelining involves overlapping communication and memory transfers with compute such that the hardware accelerator never stalls while waiting for data to arrive. Therefore, we will solely focus on the problem of communication-compute pipelining in this tutorial. We will begin by covering the problem conceptually, outlining the Pallas API for writing pipelines, and going over some realistic examples using the API. 

This tutorial only covers the conceptual foundations of pipelining. For platform-specific references, please see TPU Pipelining , or Mosaic GPU Pipelining . 

```python
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
```

## Memory Hierarchies # 

The first step in understanding pipelining conceptually involves understanding the different forms of memory available and the tradeoffs between them. Most hardware architectures (including CPUs, GPUs, and TPUs) utilize a wide variety of memory spaces that tradeoff capacity vs latency/bandwidth. For the purpose of Pallas, we are typically interested in registers, SRAM, DRAM, and potentially network communication: 

Registers are the the memory physically closest to the processor, and typically values must be loaded directly into registers before doing any compute on them. 

SRAM (also known as Shared Memory/L1 and L2 cache on GPUs, or VMEM on TPUs) also lives fairly close to the processor, but has larger capacity than registers.
SRAM on modern ML accelerators typically range in the 10-100MB range (TPU v5p contains 96MB of VMEM, and H100 GPUs contain ~30MB of L1 cache and 50MB of L2).
It’s reasonable to expect the latency to access SRAM to be on the order of 10x longer than accessing a register. 

DRAM (also known as HBM) has much higher capacity than SRAM, typically in the 10-100GB range for modern ML accelerators. However, the latency is roughly on the order of 10x longer to access compared to SRAM. 

Network communication becomes crucial for larger workloads when the size of DRAM on a single device becomes insufficient or when we’d like to take advantage of parallel computations. We do not cover distributed pipelining in this tutorial, but see the distributed TPU kernels guide for writing pipelines across multiple devices. 



In order to perform computation on values X and Y that live in HBM, we need to: 

Copy the values x and y into SRAM. 

Load the values from SRAM into registers. 

Execute the computation and store the result into registers. 

Store the values in the output registers into SRAM. 

Copy the output values in SRAM back to HBM. 

Let’s implement a Pallas function that does just that! 

```python
# Note: This is a TPU example.

def add_matrices_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
  # Load x and y from SRAM into registers
  x_regs = x_sram_ref[:, :]
  y_regs = y_sram_ref[:, :]
  # Execute a vectorized add
  z_regs = x_regs + y_regs
  # Store the output values in registers back into SRAM
  z_sram_ref[:, :] = z_regs


def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  # pallas_call will first allocate scratch buffers for `x` and `y` in SRAM.
  # It will then copy `x` and `y` from HBM into SRAM.
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  # pallas_call will also copy the output from SRAM back into HBM.
  return z


x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices(x, y)
```

```python
Array([[2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       ...,
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
```

We’ve written two functions: add_matrices_kernel and add_matrices . 

add_matrices_kernel operates using Refs that live in SRAM. Loading from a SRAM Ref produces a value that lives in registers. Values in registers behave like jax.Arrays in that we can use jnp and jax.lax operations on them to produce new values that live in registers. When we produce the values we’d like to return, we store them in the output SRAM Ref . 

The add_matrices function acts on jax.Array s and returns a jax.Array . Inside it, we pass x and y into pallas_call. pallas_call is responsible for copying x and y into SRAM and for allocating the SRAM buffers that the kernel operates on (including allocating z_vmem_ref , the output SRAM buffer). After the kernel function is finished running, pallas_call will also copy the value in z_vmem_ref to HBM, resulting in an output jax.Array . 

Pallas exposes access to lower level memory spaces like SRAM but writing performant kernels requires more care in utilizing the various memory spaces. For example, we need to consider both: 

Memory capacity . SRAM is small! If our arrays are too big, the above kernel would not work because we cannot fit the input into SRAM. For reference, an f32[2048, 2048] array is 16MiB, so our above kernel won’t scale beyond moderately sized arrays. 

Memory bandwidth . Copying to/from HBM and SRAM takes a long time, at least compared to most compute instructions. The add_matrices function above will likely spend more time copying between HBM and SRAM than actually performing the addition itself. 

With these two constraints in mind, we’ll have to rethink our strategy for getting performance out of our accelerators. 

## Pipelining Basics # 

How can we take advantage of the strengths of each form of type memory in the hierarchy, and be able to operate on large arrays stored in HBM while still utilizing fast SRAM for compute? Pipelining is a very general programming pattern which will allow us to do exactly this, but it requires transforming your problem into smaller sub-problems that can be overlapped in parallel. 

The first step in pipelining is to divide our problem into smaller subproblems that can fit inside of SRAM. For example, an elementwise operation is can be trivially transformed by operating on one slice of the source array at a time, which results in the following 3 steps (also known as stages): 

copy_in : Copy a slice A[i] from HBM to SRAM X . 

compute : Load X into registers, compute a result, and store in SRAM Y 

copy_out : Copy result Y back into HBM A[i] . 

Note that there is a data-dependence between steps 1-3, and we cannot trivially overlap them since we need step (1) to complete before starting step (2), and so on. However, there is no data dependence across multiple invocations of the subproblem - that is, we can execute step (1) for block A[i+1] while executing step (2) for block A[i] and step (3) for block A[i-1] . 



The diagram above depicts how an idealized pipelined program can be scheduled across time. The key insight is that in the majority of the kernel, the copy operations are executed in parallel with compute operations, meaning we can ideally “hide” the cost of transferring between HBM/SRAM with computation and keep the processor busy with as much uptime as possible. 

The initial startup time and final teardown time known as “bubbles”, where only a subset of the stages are being executed while the pipeline is being “filled” or “drained”. The bulk of the time is spent in the “steady-state” phase of the pipeline, where each pipeline stage is being executed in parallel across different iterations of the subproblem. While with more general pipelining approaches the goal is to achieve N-way parallelism (where N is the number of stages), with kernel pipelining we are usually bottlenecked either by memory bandwidth or processing speed. Therefore, our goal with kernel pipelining is typically to achieve full utilization of the FLOPs/s of our processor, meaning that at any point in time there is always a compute block active. In the figure above, the compute block is active in 6/8 timeslots, and assuming we are fully utilizing the processor in each compute timeslot, we would have achieved 75% utilization of the processor. 

### Deriving a Double-Buffered Pipeline # 

Now lets look at how we could implement a pipeline in pseudocode. Consider the following elementwise program, where we load values from HBM ( A[i] ) with a copy_in instruction, add 1 to the result, and store the result back to HBM with copy_out : 

```python

for i in range(N):
  copy_in(A[i], X)
  Y = X + 1
  copy_out(Y, A[i])
```

The issue with this approach is that copy_in and copy_out are typically blocking operations. So we are forced to wait for the copies to finish while the GPU/TPU is idle, then perform compute while the memory is idle. What we would like to do is to “pre-fetch” the input value that is required on the next iteration of the loop asynchronously while performing the computation for the current loop, so that compute and memory communication are happening simultaneously. 

In order to reason about the code transformation we will make, lets unroll the loop for N=4, and decompose the copy instructions into separate copy_start and copy_wait operations to be able to express asynchrony: 

```python

  # Itr 1
  copy_in_start(A[0], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[0])
  copy_out_wait(Y)

  # Itr 2
  copy_in_start(A[1], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[1])
  copy_out_wait(Y)

  # Itr 3
  copy_in_start(A[2], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[2])
  copy_out_wait(Y)

  # Itr 4
  copy_in_start(A[3], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[3])
  copy_out_wait(Y)
```

Once the loop has been unrolled, the pipelining transformation simply involves issuing copy_start instructions as early as possible, and copy_wait values as late as possible (right before we need the value). However, in the current state of the loop there is a fake data dependency through X - we cannot simultaneously perform an async copy into X while using it for computation or else we may have a race condition. Therefore, we can use a multiple-buffering technique where we keep 2 buffers for each input X and each output Y. With 2 buffers, we can push the copy_in_start one iteration ahead (with 3 buffers you can push 2 iterations, and so on) and we rewrite our loop as follows: 

```python

  # Prologue
  copy_in_start(A[0], X[0])
  
  # Itr 1
  copy_in_start(A[1], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[0])
  copy_out_wait(Y[0])

  # Itr 2 - Steady state
  copy_in_start(A[2], X[0])
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[1])
  copy_out_wait(Y[1])

  # Itr 3 - Steady state
  copy_in_start(A[3], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[2])
  copy_out_wait(Y[0])

  # Itr 4 - No copy-in
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[3])
  copy_out_wait(Y[1])
```

Next, we can push the copy_out_wait as late as possible, right before we need to write into Y on the subsequent loop iteration. 

```python

  # Prologue
  copy_in_start(A[0], X[0])
  
  # Itr 1
  copy_in_start(A[1], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[0])

  # Itr 2 - Steady state
  copy_in_start(A[2], X[0])
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[1])
  copy_out_wait(Y[0])

  # Itr 3 - Steady state
  copy_in_start(A[3], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[2])
  copy_out_wait(Y[1])

  # Itr 4 - No copy-in
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[3])
  copy_out_wait(Y[0])

  # Epilogue
  copy_out_wait(Y[1])
```

Finally, re-rolling our loop back into a for loop, we obtain the following pipelined loop: 

```python
# Prologue
copy_in_start(A[0], X[0])

# Main loop
for i in range(N):
  cur_slot = i % 2
  next_slot = (i + 1) % 2

  if i+1 < N:
    copy_in_start(A[i+1], X[next_slot])
  
  copy_in_wait(X[cur_slot])
  Y[cur_slot] = X[cur_slot] + 1
  copy_out_start(Y[cur_slot], A[i])

  if i > 0:
    copy_out_wait(Y[next_slot])

# Epilogue
copy_out_wait(Y[1])
```

If we want to generalize this loop to handle a broader set of computations, notice that we essentially need to specify 3 pieces of information to the pipeline: 

The grid , or the bounds of the for loop that specifies the number of subproblems to compute. In our example we had a 1-dimensional grid with size (N,) . 

The kernel , or the actual computation happening once the inputs have been loaded into SRAM. In our example we performed an elementwise addition Y = X + 1 . 

The data_slices , which map a subproblem to corresponding slices into the HBM buffer. In our example the data slice was the identity function lambda i: i . 

By allowing the user to specify these pieces of information we can write a wide variety of programs following this pattern: 

```python
def double_buffered_pipeline(
    grid: tuple[int, ...],
    kernel: Callable,
    in_slices: Callable,
    out_slices: Callable):
  # Prologue
  copy_in_start(in_hbm[in_slices(0)], in_sram[0])

  # Main loop
  grid_size = prod(grid)
  for i in range(grid_size):
    cur_slot = i % 2
    next_slot = (i + 1) % 2
    if (i + 1) < grid_size:
      copy_in_start(in_hbm[in_slices(i+1)], in_sram[next_slot])
    copy_in_wait(in_sram[cur_slot])

    kernel(in_sram[cur_slot], out_ram[cur_slot])

    copy_out_start(out_sram[cur_slot], out_hbm[out_slices(i)])
    if i > 0:
      copy_out_wait(out_sram[next_slot])

  # Epilogue
  last_slot = (grid_size - 1) % 2
  copy_out_wait(out_sram[last_slot])
```

Now that we’ve seen how to manually implement a pipelined loop, let’s look into how to use the Pallas API. 

## Pallas Pipelining API # 

Pallas offers a pipelining API that abstracts away the boilerplate of maintaining multiple buffers and overlapping asynchronous communication with computation. The basics of this API are covered in Pallas Quickstart , so we will go over the API briefly here for completeness and discuss some sharp edges that arise from the use of pipelining. 

### Grid # 

The program grid is a tuple of integers specifying the number of subproblems as an array. The structure of the pipeline can be interpreted as a nested for-loop where the bounds of each loop. 

```python
# For grid (N, M, K)
for n in range (N):
  for m in range(M):
    for k in range(K):
      kernel()
```

The kernel will be invoked a total of prod(grid) times. For more details, see grid and blockspecs . 

### BlockSpecs # 

A BlockSpec specifies the size and slice of data copied to the kernel on each subproblem. The basic constructor to pl.BlockSpec involves specifying the block_shape , the size of a slice of data, and index_map , a function that takes in the program ids of the current subproblem and outputs blocked indices into the source buffer. Blocked indices specify which block to copy on each iteration, assuming the source buffer has been carved into blocks of shape as block_shape . The memory_space argument specifies what memory space to copy the inputs to - be default this will be SRAM. 

```python
pl.BlockSpec(
  block_shape: tuple[int, ...],
  index_map: Callable,
  memory_space: pl.MemorySpace
)
```

There should be one BlockSpec for each input and each output to the kernel. For more details, see grid and blockspecs . 

### Kernel # 

The kernel function specifies what compute to perform on each subproblem. The kernel function should return no outputs, and instead all outputs should be written into the output buffers that are passed into the kernel. All inputs and output buffers are SRAM buffers by default (unless the user has overridden the behavior by specifying a memory_space on the corresponding BlockSpec ). 

```python
def kernel(*input_buffers, *output_buffers):
  # ... perform compute
  # ... store result into output buffers
```

The index of the current subproblem can be queried inside the kernel using pl.program_id(grid_axis: int) . 

### Pallas Call # 

The pl.pallas_call function is the main entry point to Pallas and performs pipelined execution when a grid and BlockSpecs are supplied. It has the following signature: 

```python
def pallas_call(
  kernel,
  grid: tuple[int, ...],
  in_specs: Sequence[PyTree[BlockSpec]],
  out_specs: PyTree[BlockSpec],
  out_shape: PyTree[jax.ShapeDtypeStruct],
) -> Callable:
```

pallas_call will return a callable function that when invoked with input values, will return outputs of the same shape as out_shape . 

in_specs , out_specs , and out_shape are PyTrees of their respective element type. The PyTrees for in_specs and the input buffers supplied to the kernel should match, and the PyTrees for out_specs and out_shape should also match. 

### Example - Elementwise Kernel revisited # 

Let’s revisit the initial add_matrices_kernel from the beginning of the tutorial, except using pipelining. We will add two input arrays of shape f32[4096, 4096] that live in HBM. As subproblems, we will carve up the inputs into block_shape=(512, 512) blocks and only add two blocks together at a time in the kernel. Because addition is elementwise, each index_map is identical and selects out the i, j th block on the i, j th iteration. 

```python
# Note: This is a TPU example.

total_shape = (4096, 4096)
block_shape = (512, 512)

def add_matrices_pipelined_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

def add_matrices_pipelined(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    add_matrices_pipelined_kernel,
    grid=tuple(total // block for (total, block) in zip(total_shape, block_shape)),
    in_specs=[
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j))
    ],
    out_specs=pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
    out_shape=jax.ShapeDtypeStruct(total_shape, dtype=jnp.float32),
  )(x, y)

x = jax.random.uniform(jax.random.key(0), total_shape, dtype=jnp.float32)
y = jax.random.uniform(jax.random.key(1), total_shape, dtype=jnp.float32)
result = add_matrices_pipelined(x, y)
np.testing.assert_array_equal(
    result, x + y
)
```

It turns out that with this API, writing a pipelined kernel is not much more lines of code than writing our original naive addition kernel! 

### Parameterizing a Kernel # 

It’s common to parameterize the block shapes in our kernel. Block sizes are perhaps the most important parameter to tune when optimizing the performance of Pallas kernels! They give us control over the pipeline (for example, picking smaller blocks adds more iterations to our pipelined loop where each iteration has less work to do). Let’s write a a function that does so: 

```python
def add_matrices_pipelined_param(
    x: jax.Array, y: jax.Array, *, bm: int = 256, bn: int = 256
) -> jax.Array:
  m, n = x.shape
  block_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(m // bm, n // bn),
  )(x, y)

np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=256, bn=256), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=128, bn=128), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=512, bn=512), x + y
)
```

## Sharp edges # 

While pipelining provides a close approximation to the mental model of simply calling a kernel function in a loop, there are a number of sharp edges that arise from the use of intermediate buffers that are not fully hidden from the user and can result in subtle bugs. 

### Buffer Revisiting # 

In general, a good rule-of-thumb to follow is that the input buffers passed into the kernel function should be interpreted as read-only, and output buffers are write only . 

Writing to inputs and reading from outputs will in most cases result in incorrectness. This is because the SRAM buffers passed to a kernel only contain copies of the data contained in the underlying HBM buffer. If an input SRAM buffer is updated, the updated results will never be written back out to HBM, and if an output buffer is updated, it’s updated value is never read into SRAM. This issue is analogous to staleness issues encountered when using caches in general. 

There are two cases where a buffer supports both reads and writes - accumulation (discussed next), and marking a pair of input and output buffers as input-output aliased by passing in the input_output_aliases argument to pallas_call . 

### Reductions and accumulation # 

Reduction/accumulation should only be performed over the last (innermost) dimensions of the grid, and the buffer should be initialized manually first. 

Reductions are one of the few cases where the pipeline supports both reading and writing to an output buffer, but the reason it works is subtle.
The Pallas pipeline emitter performs an optimization where if the data slices between two consecutive iterations are the same, the pipeline will not issue a copy_in / copy_out on that buffer. This means the same SRAM buffer used in a previous iteration will be passed into the kernel again on the following iteration, and thus any writes that were issued to the output buffer will become visible on the next iteration. Once the data slice changes, the final accumulated SRAM buffer will be written out to HBM. This is also why reductions must be performed over the last dimensions of the grid – we want to finish all of the accumulation while the output buffer is in SRAM in the innermost loop, then write it to HBM and never touch that output block again. 

As a concrete example, let’s consider performing the following computation for reducing an (8, 1024, 1024) array along the first axies into a (1024, 1024) array. 

```python
x = jnp.ones((8, 1024, 1024))
jnp.sum(x, axis=0)
```

```python
Array([[8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       ...,
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.]], dtype=float32)
```

To do this using pallas_call , we could use a grid of size (8,) and in each iteration i load x[i] into SRAM. Then we could add x[i] to an output SRAM buffer. Let’s implement this naively first. 

```python
# Note: This is a TPU example.

# Warning: this implementation is incorrect!
def incorrect_sum_kernel(x_ref, o_ref):
  o_ref[...] += x_ref[...]

def incorrect_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  grid = (reduction_size, *(out // blk for out, blk in zip(out_shape, block_size)))
  return pl.pallas_call(
      incorrect_sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (i, j, k))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (j, k)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)

result = incorrect_sum(x)
print(result)
```

```python
[[65. 65. 65. ... 66. 66. 66.]
 [65. 65. 65. ... 66. 66. 66.]
 [65. 65. 65. ... 66. 66. 66.]
 ...
 [71. 71. 71. ... 72. 72. 72.]
 [71. 71. 71. ... 72. 72. 72.]
 [71. 71. 71. ... 72. 72. 72.]]
```

This result is completely wrong! 

There are two errors inside this kernel. First, we are accumulating along the first grid dimension instead of the last grid dimension. Second, o_ref initially contains garbage values and thus we need to initialize it to zeros before we begin accumulation. 

After fixing these two issues, we obtain the following corrected kernel. In this new kernel, we use @pl.when to create a conditional that checks when the program ID is 0 along the reduction axis, indicating we are beginning to accumulate into a new output block. We have also moved the reduction dimension to the last axis of the grid . 

```python
# Note: This is a TPU example.

def correct_sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
  o_ref[...] += x_ref[...]

def correct_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  # We moved the reduction to the last axis of the grid.
  grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
  return pl.pallas_call(
      correct_sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)

result = correct_sum(x)
print(result)
```

```python
[[8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 ...
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]]
```

## Analyzing the performance # 

What is the performance of a pipelined kernel? This question can vary depending on where the bottleneck in the hardware is. We are typically interested in 3 quantities: 

Memory latency \(α\) , the minimum latency of a memory transfer. 

Memory bandwidth \(β\) , the rate in bytes/second that we can transfer from HBM to SRAM. 

FLOP/s \(F\) , or floating-point-operations per second, the number of calculations per second that the processor can perform. 

We refer to a program as compute-bound if the processing speed FLOPs/s is the bottleneck, and as memory-bound if the bandwidth or latency are the bottleneck. Generally, our goal is to optimize a kernel such that it is compute-bound, meaning we are utilizing all of the available processing power of our hardware. 

Suppose we are running a program that requires \(X\) bytes of memory transfers per kernel iteration, and runs \(Y\) floating-point operations per iteration. The ratio of \(X\) to \(Y\) varies depending on the type of compute – for elementwise operations such as addition or multiplication, they will both scale equally. However, for operations such as matrix multiplication, compute scales cubically with the size of the problem while memory scales quadratically. 

In a compute-bound regime, a pipeline running \(N\) iterations would take \((\alpha + X/\beta) + N (Y/F)\) seconds, where the first term represents the cost of the initial bubble (multiply by a factor of 2 if there is also a bubble at the end), and the second term represents the total time of the steady-state of the pipeline. Assuming that N is large and there is enough work to produce a long pipeline, the dominating term in the runtime is \(F\) , the processing speed of the accelerator. 



In a memory-bound regime it is useful to identify if the problem is the latency versus the bandwidth. If the bandwidth is the bottleneck, then the total runtime would take \(\alpha + N(X / \beta)\) seconds. In contrast with a latency-bound regime, the memory copies happen serially because the bandwidth is already saturated. Being memory-bound is generally not ideal as there will be gaps in time where the processor is idle, and in most hardware configurations the memory bandwidth \(\beta\) is orders of magnitude slower than the processing speed \(F\) . 



If the bottleneck is specifically the latency and not the bandwidth, it is possible to fix the problem by inserting additional pipeline stages at the cost of additional SRAM required to store more buffers. With sufficient stages, the problem will either become compute or bandwidth bound again depending on which bottleneck we hit first during the steady-stage stage of the pipeline. The downside, however, of a multi-stage pipeline is that the size of the bubble is proportional to the number of stages so it is important to make sure the pipeline is long enough such that the bubble does not take up a substantial amount of the total runtime. 



Pallas on TPU only supports double-buffering, as TPU programs can operate on larger block sizes and double-buffering is typically enough to cover the latency. On GPU, the number of pipeline stages can be specified in both the Triton (via CompilerParams ) and Mosaic GPU backends (via argument to the pipeline emitter). See the platform-specific pipelining documentation for more details. 

previous 

Pallas Quickstart 

next 

Grids and BlockSpecs 
Contents Memory Hierarchies Pipelining Basics Deriving a Double-Buffered Pipeline Pallas Pipelining API Grid BlockSpecs Kernel Pallas Call Example - Elementwise Kernel revisited Parameterizing a Kernel Sharp edges Buffer Revisiting Reductions and accumulation Analyzing the performance 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Grids and BlockSpecs .md .pdf 
# Grids and BlockSpecs 

## Contents 
grid , a.k.a. kernels in a loop BlockSpec , a.k.a. how to chunk up inputs The “element” indexing mode 
# Grids and BlockSpecs # 

## grid , a.k.a. kernels in a loop # 

When using jax.experimental.pallas.pallas_call() the kernel function
is executed multiple times on different inputs, as specified via the grid argument to pallas_call . Conceptually: 

```python
pl.pallas_call(some_kernel, grid=(n,))(...)
```

maps to 

```python
for i in range(n):
  some_kernel(...)
```

Grids can be generalized to be multi-dimensional, corresponding to nested
loops. For example, 

```python
pl.pallas_call(some_kernel, grid=(n, m))(...)
```

is equivalent to 

```python
for i in range(n):
  for j in range(m):
    some_kernel(...)
```

This generalizes to any tuple of integers (a length d grid will correspond
to d nested loops).
The kernel is executed as many times
as prod(grid) .
The default grid value () results in one
kernel invocation.
Each of these invocations is referred to as a “program”.
To access which program (i.e. which element of the grid) the kernel is currently
executing, we use jax.experimental.pallas.program_id() .
For example, for invocation (1, 2) , program_id(axis=0) returns 1 and program_id(axis=1) returns 2 .
You can also use jax.experimental.pallas.num_programs() to get the
grid size for a given axis. 

See Grids by example for a simple kernel that uses this API. 

## BlockSpec , a.k.a. how to chunk up inputs # 

In conjunction with the grid argument, we need to provide Pallas
the information on how to slice up the input for each invocation.
Specifically, we need to provide a mapping between the iteration of the loop to which block of our inputs and outputs to be operated on .
This is provided via jax.experimental.pallas.BlockSpec objects. 

Before we get into the details of BlockSpec s, you may want
to revisit Block specs by example in Pallas Quickstart. 

BlockSpec s are provided to pallas_call via the in_specs and out_specs , one for each input and output respectively. 

First, we discuss the semantics of BlockSpec when indexing_mode == pl.Blocked() . 

Informally, the index_map of the BlockSpec takes as arguments
the invocation indices (as many as the length of the grid tuple),
and returns block indices (one block index for each axis of
the overall array). Each block index is then multiplied by the
corresponding axis size from block_shape to get the actual element index on the corresponding array axis. 

Note 

Not all block shapes are supported. 

On TPU, only blocks with rank at least 1 are supported.
Furthermore, the last two dimensions of your block shape must be equal to
the respective dimension of the overall array, or be divisible
by 8 and 128 respectively. For blocks of rank 1, the block dimension
must be equal to the array dimension, or be a multiple of 1024, or be a
power of 2 and at least 128 * (32 / bitwidth(dtype)) . 

On GPU, when using the Mosaic GPU backend, the size of the blocks is
unrestricted. However, due to hardware limitations, the size of the minormost
array dimension must by such that it is a multiple of 16 bytes. For example,
it must be a multiple of 8 if the input is jnp.float16 . 

On GPU, when using the Triton backend, the size of the blocks themselves is
unrestricted, but each operation (including a load or store) must operate
on arrays whose size is a power of 2. 

If the block shape does not divide evenly the overall shape then the
last iteration on each axis will still receive references to blocks
of block_shape but the elements that are out-of-bounds are padded
on input and discarded on output. The values of the padding are unspecified, and
you should assume they are garbage. In the interpret=True mode, we
pad with NaN for floating-point values, to give users a chance to
spot accessing out-of-bounds elements, but this behavior should not
be depended upon. Note that at least one of the
elements in each block must be within bounds. 

More precisely, the slices for each axis of the input x of
shape x_shape are computed as in the function slice_for_invocation below: 

```python
>>> import jax
>>> from jax.experimental import pallas as pl
>>> def slices_for_invocation(x_shape: tuple[int, ...],
...                           x_spec: pl.BlockSpec,
...                           grid: tuple[int, ...],
...                           invocation_indices: tuple[int, ...]) -> tuple[slice, ...]:
...   assert len(invocation_indices) == len(grid)
...   assert all(0 <= i < grid_size for i, grid_size in zip(invocation_indices, grid))
...   block_indices = x_spec.index_map(*invocation_indices)
...   assert len(x_shape) == len(x_spec.block_shape) == len(block_indices)
...   elem_indices = []
...   for x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):
...     start_idx = block_idx * block_size
...     # At least one element of the block must be within bounds
...     assert start_idx < x_size
...     elem_indices.append(slice(start_idx, start_idx + block_size))
...   return elem_indices

```

For example: 

```python
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]

>>> # Same shape of the array and blocks, but we iterate over each block 4 times
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j, k: (i, j)),
...                       grid = (10, 5, 4),
...                       invocation_indices = (2, 4, 0))
[slice(20, 30, None), slice(80, 100, None)]

>>> # An example when the block is partially out-of-bounds in the 2nd axis.
>>> slices_for_invocation(x_shape=(100, 90),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]

```

The function show_program_ids defined below uses Pallas to show the
invocation indices. The iota_2D_kernel will fill each output block
with a decimal number where the first digit represents the invocation
index over the first axis, and the second the invocation index
over the second axis: 

```python
>>> def show_program_ids(x_shape, block_shape, grid,
...                      index_map=lambda i, j: (i, j)):
...   def program_ids_kernel(o_ref):  # Fill the output block with 10*program_id(1) + program_id(0)
...     axes = 0
...     for axis in range(len(grid)):
...       axes += pl.program_id(axis) * 10**(len(grid) - 1 - axis)
...     o_ref[...] = jnp.full(o_ref.shape, axes)
...   res = pl.pallas_call(program_ids_kernel,
...                        out_shape=jax.ShapeDtypeStruct(x_shape, dtype=np.int32),
...                        grid=grid,
...                        in_specs=[],
...                        out_specs=pl.BlockSpec(block_shape, index_map),
...                        interpret=True)()
...   print(res)

```

For example: 

```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1  1]
 [ 0  0  0  1  1  1]
 [10 10 10 11 11 11]
 [10 10 10 11 11 11]
 [20 20 20 21 21 21]
 [20 20 20 21 21 21]
 [30 30 30 31 31 31]
 [30 30 30 31 31 31]]

>>> # An example with out-of-bounds accesses
>>> show_program_ids(x_shape=(7, 5), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1]
 [ 0  0  0  1  1]
 [10 10 10 11 11]
 [10 10 10 11 11]
 [20 20 20 21 21]
 [20 20 20 21 21]
 [30 30 30 31 31]]

>>> # It is allowed for the shape to be smaller than block_shape
>>> show_program_ids(x_shape=(1, 2), block_shape=(2, 3), grid=(1, 1),
...                  index_map=lambda i, j: (i, j))
[[0 0]]

```

When multiple invocations write to the same elements of the output
array the result is platform dependent. 

In the example below, we have a 3D grid with the last grid dimension
not used in the block selection ( index_map=lambda i, j, k: (i, j) ).
Hence, we iterate over the same output block 10 times.
The output shown below was generated on CPU using interpret=True mode, which at the moment executes the invocation sequentially.
On TPUs, programs are executed in a combination of parallel and sequential,
and this function generates the output shown.
See Noteworthy properties and restrictions . 

```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2, 10),
...                  index_map=lambda i, j, k: (i, j))
[[  9   9   9  19  19  19]
 [  9   9   9  19  19  19]
 [109 109 109 119 119 119]
 [109 109 109 119 119 119]
 [209 209 209 219 219 219]
 [209 209 209 219 219 219]
 [309 309 309 319 319 319]
 [309 309 309 319 319 319]]

```

A None value appearing as a dimension value in the block_shape behaves
as the value 1 , except that the corresponding
block axis is squeezed (you could also pass in pl.Squeezed() instead of None ). In the example below, observe that the
shape of the o_ref is (2,) when the block shape was specified as (None, 2) (the leading dimension was squeezed). 

```python
>>> def kernel(o_ref):
...   assert o_ref.shape == (2,)
...   o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))
>>> pl.pallas_call(kernel,
...                jax.ShapeDtypeStruct((3, 4), dtype=np.int32),
...                out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),
...                grid=(3, 2), interpret=True)()
Array([[ 0,  0, 10, 10],
       [ 1,  1, 11, 11],
       [ 2,  2, 12, 12]], dtype=int32)

```

When we construct a BlockSpec we can use the value None for the block_shape parameter, in which case the shape of the overall array
is used as block_shape .
And if we use the value None for the index_map parameter
then a default index map function that returns a tuple of zeros is
used: index_map=lambda *invocation_indices: (0,) * len(block_shape) . 

```python
>>> show_program_ids(x_shape=(4, 4), block_shape=None, grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]

>>> show_program_ids(x_shape=(4, 4), block_shape=(4, 4), grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]

```

### The “element” indexing mode # 

The behavior documented above applies to the default “blocked” indexing mode.
When integers are used in the block_shape tuple e.g. (4, 8) , it is
equivalent to passing in a pl.Blocked(block_size) object instead, e.g. (pl.Blocked(4), pl.Blocked(8)) . Blocked indexing mode means the indices
returned by index_map are block indices . We can pass in objects other than pl.Blocked to change the semantics of index_map , most notably, pl.Element(block_size) ..
When using the pl.Element indexing mode the values returned by the
index map function are used directly as the array indices, without first
scaling them by the block size.
When using the pl.Element mode you can specify virtual padding
of the array as a tuple of low-high paddings for the dimension: the
behavior is as if the overall array is padded on input. No guarantees
are made for the padding values in element mode, similarly to the padding
values for the blocked indexing mode when the block shape does not divide the
overall array shape. 

The Element mode is currently supported only on TPUs. 

```python
>>> # element without padding
>>> show_program_ids(x_shape=(8, 6), block_shape=(pl.Element(2), pl.Element(3)),
...                  grid=(4, 2),
...                  index_map=lambda i, j: (2*i, 3*j))
    [[ 0  0  0  1  1  1]
     [ 0  0  0  1  1  1]
     [10 10 10 11 11 11]
     [10 10 10 11 11 11]
     [20 20 20 21 21 21]
     [20 20 20 21 21 21]
     [30 30 30 31 31 31]
     [30 30 30 31 31 31]]

>>> # element, first pad the array with 1 row and 2 columns.
>>> show_program_ids(x_shape=(7, 7),
...                  block_shape=(pl.Element(2, (1, 0)),
...                               pl.Element(3, (2, 0))),
...                  grid=(4, 3),
...                  index_map=lambda i, j: (2*i, 3*j))
    [[ 0  1  1  1  2  2  2]
     [10 11 11 11 12 12 12]
     [10 11 11 11 12 12 12]
     [20 21 21 21 22 22 22]
     [20 21 21 21 22 22 22]
     [30 31 31 31 32 32 32]
     [30 31 31 31 32 32 32]]

```

previous 

Software Pipelining 

next 

Pallas TPU 
Contents grid , a.k.a. kernels in a loop BlockSpec , a.k.a. how to chunk up inputs The “element” indexing mode 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Pallas TPU Writing TPU kernels with Pallas .rst .pdf 
# Writing TPU kernels with Pallas 

## Contents 
What is a TPU? Noteworthy properties and restrictions BlockSpec s and grid iteration Array Layouts Multicore TPU configurations Placing operands in SMEM Supported data types Computation placement Supported operations Matrix multiplication Precision control Transposition Accessing memory Elementwise operations Array constructors Reductions Broadcasting Reshapes Random Number Generation Control flow 
# Writing TPU kernels with Pallas # 

This page focuses on the details that are important when attempting to run
Pallas kernels on Google TPUs. For one, the TPU backend is still in an
experimental phase, and only a subset of JAX NumPy will be accepted.
Furthermore, writing performant code for TPUs might require thinking carefully
about the native capabilities of the hardware. While many patterns that are
unnatural to the hardware will be accepted, they might end up requiring
software emulation, and can slow down the computation. 

Warning 

This feature should still be considered experimental as work is still in
progress (in particular on improving the error messages). 

Note 

While all the features described here are experimental, we remain very serious
about maintaining their correctness. As such, it might not be uncommon to
see a “not implemented” error while attempting to write TPU kernels. But, if
a kernel is accepted by the compiler, it must return the expected results. 

If you see unexpected outputs, please compare them against a kernel run with interpret=True passed in to pallas_call . If the results diverge,
please file a bug report . 

## What is a TPU? # 

TPU is a hardware accelerator developed at Google. You can think of TPUs as
GPUs, but specialized for machine learning workloads specifically. As such,
their architecture differs quite significantly. However, we believe that Pallas
can make it easy to start writing TPU kernels, even without having a full
understanding of the underlying hardware. Having said that, understanding the
hardware well will certainly make it easier to write performant kernels. 

In a nutshell, the main difference between TPUs and GPUs is that TPUs are
sequential machines with a very wide vector register (kind of like a CPU!).
At the same time, they allow the software to schedule certain operations in the
background, making them execute asynchronously with respect to the main
instruction stream. This includes things like HBM memory accesses
(which cannot be issued directly, but instead have to be prefetched to
lower levels of the memory hierarchy by the DMA subunits), matrix multiplies
(supported by the MXU unit) or matrix transpositions and permutes (supported by
the XLU unit). 

If you’re interested in learning more about the TPU architecture
in detail, we recommend reading a collection of papers published over the
years. While many of them talk about specific TPU generations, many of the
ideas described transfer to later generations as well. 

A Domain-Specific Supercomputer for Training Deep Neural Networks 

The Design Process for Google’s Training Chips: TPUv2 and TPUv3 

Ten Lessons From Three Generations Shaped Google’s TPUv4i : Industrial Product 

TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings 

## Noteworthy properties and restrictions # 

### BlockSpec s and grid iteration # 

BlockSpec s (see BlockSpec, a.k.a. how to chunk up inputs ) generally behave as expected
in Pallas — every invocation of
the kernel body gets access to slices of the inputs and is meant to initialize a slice
of the output. 

Note 
Not all block shapes are supported. On TPU, only blocks with rank at least 1 
are supported. Furthermore, the last two dimensions of your block shape
must be divisible by 8 and 128 respectively, or be equal to the respective
dimensions of the overall array. 

One interesting aspect of Pallas TPU kernels is the way they handle memory spaces:
While the inputs to pallas_call will often reside in HBM (the main TPU
memory), the references passed in to the kernel body will point to buffers in
lower levels of memory hierarchy (VMEM or SMEM). This enables the kernel body
to write and read them at very high speeds, while all the communication with
HBM (which has very high latency) is handled by the compiler and overlapped
with compute. 

What’s more, compared to GPUs, TPUs are actually highly sequential machines.
Ergo, the grid is generally not processed in parallel, but sequentially,
in lexicographic order (though see the Multicore TPU configurations section
for exceptions). This unlocks some interesting capabilities: 

When two (lexicographically) consecutive grid indices use the same slice of
an input, the HBM transfer for the second iteration is skipped, as the data is
already available. 

Multiple invocations of the kernel body can write to the same slice of the
output, without any risk of race conditions. However, we do require that all
invocations that write to a particular slice are consecutive. 

The “consecutive” restriction on the output usually means that some prefix
of the grid dimensions always varies the slice of the output an invocation needs
to access, while the output window remains constant for the remaining suffix. 

For example, when implementing a Pallas TPU kernel for matrix multiplication,
one would generally use a 3 dimensional grid: the first two dimensions would
correspond to slicing along the first axis of the left operand and the second
axis of the second operand. The third and last grid axis would tile the
reduction dimension. The grid axis corresponding to the reduction dimension has
to be the last one, since the output window does not vary along this axis.
The output reference can be then used as an accumulator for partial results. 

Note 

VMEM is fairly large for such a low-level memory hierarchy (16MB+), making it
possible to use large window sizes. And, oftentimes, the larger the window
size, the better the eventual hardware utilization will be. However, it is possible to
specify a window size that (together with space necessary to hold
spilled vector registers) exceeds the size of VMEM. In this case, you will likely see a
low-level compiler error message complaining about an out-of-memory error. 

### Array Layouts # 

Dimension ordering of arrays is meaningful in Pallas.
In JAX programs, the ordering of intermediate arrays inside jax.jit usually
has no impact on performance, as the compiler is free to rearrange them.
However, as Pallas is meant to expose lower-level capabilities, the dimension
order can have great impact on the quality of generated code. 

TPUs perform the bulk of the computation on 2D vector registers, which are typically of
size 8x128 for 32-bit values (as of TPU v6).
When a vector value is loaded from VMEM into registers (e.g. x = x_ref[...] ),
the last two dimensions of the array will be tiled into the registers.
Pallas will only ever consider mapping the last two dimensions of
intermediate arrays to the 8x128 vector register dimensions (sublanes and lanes
respectively). 

Here is a graphical example of how a 12x320 array can be tiled using 6 8x128
tiles: 

Tiled layouts have several import ramifications for kernel writers: 

The last two axes of an array are treated differently than other
axes. For example, reductions, reshapes, and transposes are generally
more expensive when involving the last two axes. Some reshapes
involving the last two dimensions are not supported and will result in a compiler
error, but are “free” and performed at compile time for other dimensions. 

While sometimes unavoidable, it is generally wasteful to have singleton
dimensions in the last two axes, since they will occupy 1 element out of
the entire tile dimension. Consuming too many registers can
also potentially cause register spills into VMEM which degrades kernel
performance. 

Related to the above point, all vector computation is padded up to the tile
size. Adding a two 1x1 arrays costs as much as adding two 8x128 arrays, and
adding two 8x128x1x1 arrays will be 1024 times as expensive as adding two
8x128 arrays, since the 8x128x1x1 array will be padded to 8x128x8x128. 

### Multicore TPU configurations # 

In newer TPU generations, the two cores on a chip are often abstracted as a
single device. To take advantage of multiple cores, Pallas has to break the
sequential grid execution guarantees, and will need to parallelize one of the
grid axes over cores. This is an opt-in procedure. To allow that, pallas_call requires an extra parameter named dimension_semantics : 

```python
pallas_call(
    ...,
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=["parallel", "parallel", "arbitrary"]
    ),
  )
```

That parameter is a list, with as many entries as many axes there are in the
grid. Only parallel dimensions can be partitioned over cores. As a rule of
thumb, the dimensions are parallel, unless the output window does not vary.
As such, dimension_semantics is always a number of parallel axes
followed by a number of arbitrary axes. 

While partitioning a kernel over a 2-core TPU device often leads to a 2x
speedup, it can be in fact significantly smaller. This is especially true if
different instances of the body have highly varying cost. If all of the expensive
steps get mapped to one core, but all cheap steps are assigned to the other, the
second core will be sitting idle until the first one completes its tasks. 

Pallas TPU generally favors partitioning axes of a size that is a multiple of the
number of TPU cores, and prefers to partition leading grid axes. 

### Placing operands in SMEM # 

Most of the compute on the TPU will happen on the vector unit. Still, there are
many cases where it is useful to perform a number of scalar operations, e.g., to
carry out control-flow. For that reason, TPUs come with a separate
scalar unit, and a separate scalar memory (SMEM) attached to it.
As a rule of thumb, any data used to perform control-flow decisions should
be placed in SMEM. 

SMEM is a low-latency memory that supports random access, but lets you only
read and write 32-bit values with a single instruction (very small compared to
the 4KBi granularity of VMEM transactions, but much more flexible due to lack
of alignment requirements!). 

The scalar memory is also very useful when implementing kernels that do not
access the tiles of inputs in a regular pattern, such as when writing
block-sparse kernels. In Pallas, this can be achieved by replacing the grid argument to pallas_call with a grid_spec of PrefetchScalarGridSpec with a non-zero num_scalar_prefetch argument.
If num_scalar_prefetch is n , then the first n arguments to pallas_call will be placed in SMEM. No BlockSpec s should be specified
for those arguments. But, the BlockSpec s for all subsequent arguments will
receive not only the grid indices, but also the SMEM references to the leading
operands. 

See Scalar Prefetch and Block-Sparse Computation for examples on using this
feature. 

### Supported data types # 

At the moment Pallas TPU supports the following data types: 

jnp.float32 

jnp.bfloat16 

jnp.int* (all precisions, except for jnp.int4 ) 

jnp.uint* (all precisions) 

jnp.bool_ 

### Computation placement # 

All scalar (i.e. 0D) arrays will be stored in scalar registers, and operations
on then will be executed on the scalar core.  All other operations (even on
single-element, but 1D+ arrays) will be executed on the vector core. 

## Supported operations # 

### Matrix multiplication # 

Matrix multiplication always produces results in the float32 format.
If your inputs are not float32, we recommend using lax.dot with preferred_element_type set to jnp.float32 . 

When using lax.dot_general , it is possible to fuse transpositions of
the last two dimensions of matrix multiplication operands into the operation,
which can improve overall kernel performance. 

#### Precision control # 

Pallas TPU lowering is aware of jax.default_matmul_precision . For best
performance (and lowest precision), use bfloat16 . If you care about
numerical accuracy, you might want to set the precision to float32 . 

Warning 

Even if you pass in 32-bit operands to a matrix multiplication, they will be
rounded to bfloat16 unless float32 precision is requested. 

### Transposition # 

If the value has at least 4 dimensions, arbitrary transpositions of all but
the last two axes are free.
Otherwise, only the transposition of the last two axes is implemented.
Note that some transpositions of the last two dimensions can be fused into
matrix multiplication. 

### Accessing memory # 

Arbitrary slices of references can be read or updated, subject to implementation
constraints. Currently, no restrictions are placed on inputs that are 32-bit wide,
but only some slicing patterns are supported for narrower types. Reads and
writes that are aligned to multiples of, and have a length that is a multiple
of 8 and 128 respectively in the last two dimensions are always supported. 

Reads and writes to vector memory generally happen on tiles of shape (8, 128) .
As such, when reading or writing to references that have at least two dimensions,
the best performance is achieved when the base offset of the memory access
has indices divisible by the tiling, and the size of the read region is a
multiple of the tile size. 

### Elementwise operations # 

Many elementwise operations are supported. It is worth noting that the hardware
generally only supports elementwise computation using 32-bit types. When loading
operands that use lower-precision types, they should generally be upcast to a
32-bit type before applying elementwise ops. 

It is worth noting that they can vary significantly in their cost. As such, we
outline three categories of supported operations: cheap (🟢), medium (🌕) and
expensive (🔴). 

Operation 

Cost 

jnp.add , + 

🟢 

jnp.sub , - 

🟢 

jnp.mul , * 

🟢 

/ , // , % 

🌕 

jnp.max , jnp.min 

🟢 

jnp.where (select) 

🟢 

jnp.abs 

🟢 

| , ^ , & , ~ 

🟢 

<< , >> 

🟢 

Comparisons ( == , …) 

🟢 

Type casts ( .astype ) 

🟢 

jnp.exp 

🌕 

jnp.tanh 

🌕 

jnp.pow 

🌕 

jnp.sin 

🔴 

jnp.cos 

🔴 

Many JAX functions are implemented in terms of other JAX primitives, so this
list might not be comprehensive. For example, jax.nn.relu is implemented
in terms of comparisons and jnp.where will work in Pallas kernels too. 

### Array constructors # 

All constant array constructors are supported ( jnp.ones , jnp.zeros , jnp.full ). 

### Reductions # 

sum , max , min (for floating point values) reductions are supported, as well
as any and all for boolean values. Integer reductions are not supported. 

Reductions over the last array dimension are generally the slowest.
Reductions over the second last dimension are faster, but still slower than
over the leading dimensions. 

### Broadcasting # 

The performance characteristics of broadcasting are very similar to those
of reductions. Broadcasting along all but the two trailing dimensions is
always supported and free. Broadcasting along the second to last dimension is
slower, while broadcasting along the last dimension is the slowest. 

### Reshapes # 

As usual, reshapes in all dimensions but the last two dimensions are supported
and free. 

The only two supported cases when a reshape can modify the last two dimensions
of an array is when (1) some leading dimensions are flattened onto the second
to last dimension, or (2) it adds a dimension that was just removed by a
reduction. 

### Random Number Generation # 

Pallas supports the most commonly used functions from the jax.random module,
such as uniform , normal , and bernoulli . The key should be a threefry2x32 key,
which is the default setting in JAX. Keys can be directly passed into a kernel,
or generated inside of a kernel. 

### Control flow # 

The TPU backend features limited support for control flow at the moment. The
currently supported functions are cond , fori_loop and for_loop .
However, loop primitives get fully unrolled during the compilation at the
moment, so try to keep the loop trip count reasonably small. 

Overusing control flow can lead to significant regressions in low-level code
generation, and it is recommended to try to squeeze as many computationally
expensive operations into a single basic block as possible. 

previous 

Pallas TPU 

next 

TPU Pipelining 
Contents What is a TPU? Noteworthy properties and restrictions BlockSpec s and grid iteration Array Layouts Multicore TPU configurations Placing operands in SMEM Supported data types Computation placement Supported operations Matrix multiplication Precision control Transposition Accessing memory Elementwise operations Array constructors Reductions Broadcasting Reshapes Random Number Generation Control flow 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Pallas TPU TPU Pipelining .ipynb .pdf 
# TPU Pipelining 

## Contents 
TPU and its memory spaces TPU-specific Pipelining Features TPU Memory Spaces Multiple Buffering pltpu.emit_pipeline Lookahead Prefetch Dynamic Block Shapes TPUs in Megacore configuration 
# TPU Pipelining # 

This guide serves as a reference for TPU-specific pipelining concerns.
We’ll review the memory hierarchy and compute units on TPUs, and TPU-specific features of the pipelining API. For a more general-purpose overview of pipelining, see the Software Pipelining . 

```python
#@title Imports

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
```

## TPU and its memory spaces # 

A TPU and its TensorCore consist of memory spaces (where arrays can reside),
registers (which temporarily store scalar and array values) and compute units
(that do computation with values in registers).
Below is a diagram of a TPU in which x and y are arrays that live in
high-bandwidth memory (HBM): 



Let’s talk about the components of this diagram in more detail: 

Memory spaces : A TPU has high-bandwidth memory (HBM) which is what we
often think of as “device memory”.
There is also vector memory (VMEM),
a cache meant for storing vector and array values, and scalar memory (SMEM),
a cache designed to store scalar values. 

Registers : A TensorCore has two main types of registers: vector
registers (VREGs) store array values, and scalar registers (SREGs) store
scalar values.
Values can be loaded into memory from their respective caches (VMEM for
VREGs and SMEM for SREGs). 

Compute units : A TensorCore has a scalar unit, vector unit (VPU) and
matrix unit (MXU) that can do numerical computation. Each of these compute units can operate asynchronously, but this is managed by the TPU compiler and thus from the programmer’s perspective a TPU program is single-threaded.
Compute units operate on values that live in SREGs and VREGs and output
values into those registers as well. 

## TPU-specific Pipelining Features # 

Pallas TPU supports the following platform-specific features. 

### TPU Memory Spaces # 

Pallas exposes all levels of the TPU memory hierarchy to users. The following table maps from Pallas TPU memory spaces to their standard memory types (DRAM/SRAM): 

Pallas Enum 

TPU Memory Space 

Type (DRAM/SRAM) 

pl.ANY 

HBM (usually) or VMEM 

DRAM 

pltpu.VMEM 

VMEM 

SRAM 

pltpu.SMEM 

SMEM 

SRAM 

pltpu.SEMAPHORE 

Semaphore 

SRAM 

MemorySpace.VMEM denotes vector SRAM. It is the default memory space if nothing is specified. 

MemorySpace.SMEM denotes scalar SRAM. Only scalar loads and stores can be performed to/from SMEM. 

MemorySpace.ANY is a hint to the compiler that the memory space is unconstrained. In most cases, XLA will place this buffer in HBM. A buffer assigned to the ANY memory space cannot be dereferenced normally using array indexing syntax (e.g. x[...] ). Instead, we must first copy the values into a VMEM or SMEM buffer using pltpu.sync_copy or pltpu.async_copy . 

MemorySpace.SEMAPHORE is used to allocate semaphores for constructing barriers or tracking asynchronous operations. It is also possible to return semaphores from the kernel for building asynchronous kernels - this is an experimental feature; see Pallas Async Operations for more details. 

Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The default behavior for pallas_call on TPU is that arguments to pallas_call are assumed to live in HBM, and inputs to the user kernel body are stored in VMEM. 

While not specific to pipelining, it is possible to gain manual control over the memory space of input and output buffers, you can specify the memory_space argument on a BlockSpec . Note that pipelining is not allowed unless the memory_space is marked as VMEM . Memory spaces can also be used to specify scratch arguments to a kernel via the scratch_shapes argument on pallas_call . Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in VMEM , SMEM , or SEMAPHORE . 

As an example for using multiple manual memory space assignments in a kernel, the following program copies a slice of an HBM buffer x_hbm_ref into a scratch VMEM buffer scratch_vmem_ref before using it for arithmetic and storing the result into an output VMEM buffer: 

```python
def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
  pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
  out_vmem_ref[...] = scratch_vmem_ref[...] + 1

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
out = pl.pallas_call(hbm_vmem_kernel,
  in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
  out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
  scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),)
)(x)

np.testing.assert_allclose(out, x[0:1] + 1)
```

### Multiple Buffering # 

Multiple buffering can be specified on a per-argument basis to the pipeline via the pipeline_mode option on pl.BlockSpec . To do so, pass a pl.Buffered object to pl.BlockSpec specifying the number of buffers to allocate for this particular argument: 

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count)
)
```

The default buffer count is 2 for all inputs and outputs. 

### pltpu.emit_pipeline # 

pltpu.emit_pipeline is a pipelining API implemented in Pallas that allows you to construct pipelines inside of a kernel rather than only on kernel entry. This several use-cases over using pl.pallas_call , such as: 

For constructing nested pipelines. For example, an outer pipeline that communicates between chips, and an inner pipeline that performs HBM-VMEM pipelining. 

For using emit_pipeline specific features such as lookahead prefetch and dynamic block shapes (covered below). 

pltpu.emit_pipeline follows a similar signature to pl.pallas_call and requires you to specify a body kernel , a grid, and block specs for inputs and outputs: 

```python
def emit_pipeline(
    kernel: Callable,
    grid: tuple[int],
    in_specs: PyTree[BlockSpec] = None,
    out_specs: PyTree[BlockSpec] = None,
    dimension_semantics: tuple[GridDimensionSemantics] = None,
    core_axis: int | None = None,
) -> Callable:
  ... # Returns a custom pipeline given an inner kernel and BlockSpecs.
```

The dimension_semantics and core_axis arguments are used for partitioning the kernel grid over Megacore (see below). 

### Lookahead Prefetch # 

Lookahead prefetch is a pipelining feature where the pipeline will attempt to prefetch the next input block as soon as a buffering slot is available, rather than the iteration directly before it would be used. For example, if the kernel had a grid of (8,) and the block indices to fetch on each iteration were 0, 0, 0, 0, 1, 1, 1, 1 , then lookahead prefetch will begin fetching both blocks 0 and 1 on iteration 0, whereas the standard pipeline schedule would fetch block 0 on iteration 0 but not begin fetching block 1 until iteration 3. There is a small amount of control flow overhead in performing lookahead so it is disabled by default. 

Lookahead is primarily useful when there is a variable amount of compute work in each block, such as when some blocks contain skipped or a reduced amount of work. In these cases, there may not be enough compute work in the iteration immediately preceding the step when the block is needed to fully overlap with the memory transfer. Therefore, we would like to begin fetching blocks earlier in the pipeline. 

Lookahead prefetch can be used in conjunction with multiple buffering and can likewise be enabled by passing pl.Buffered into the pipeline_mode argument: 

```python
pl.BlockSpec(
  pipeline_mode=pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
)
```

### Dynamic Block Shapes # 

pltpu.emit_pipeline supports pipelining over blocks with dynamic but bounded shapes. In order to specify such an block shape, the dynamic-sized dimension in the block should be marked with pl.BoundedSlice(max_size) rather than a static integer size, where max_size is the maximum size of the block. In addition, the corresponding index returned by index_map should be a dynamic slice constructed via pl.ds(start, size) where both start and size are element indices (not block indices) and can be dynamic. 

The following is an example for a block spec with a dynamic first dimension: 

```python
pl.BlockSpec(
   block_shape=(pl.BoundedSlice(32), 256),
   index_map=lambda *grid_idxs: (pl.ds(start, end), 0),
)
```

```python
# The following kernel copies `x` to the output in dynamic-sized chunks
# passed in via `slices`.

def dynamic_block_example_kernel(x_hbm, slices_hbm, o_hbm, slices_smem):
    pltpu.sync_copy(slices_hbm, slices_smem)  # Copy slices into SMEM.
    def pipeline_body(x_vmem, o_vmem):
        o_vmem[...] = x_vmem[...]
    def index_map(i):
        start = slices_smem[i, 0]
        size = slices_smem[i, 1] - slices_smem[i, 0]
        return (pl.ds(start, size), 0)
    block_spec = pl.BlockSpec(block_shape=(pl.BoundedSlice(8), 128),
                              index_map=index_map)
    pltpu.emit_pipeline(
        pipeline_body,
        grid=(slices.shape[0],),
        in_specs=[block_spec],
        out_specs=block_spec
    )(x_hbm, o_hbm)

x = jax.random.uniform(jax.random.key(0), (8, 128), jnp.float32)
slices = jnp.array([[0, 2], [2, 3], [3, 5], [5, 8]], dtype=jnp.int32)

hbm_block_spec = pl.BlockSpec(memory_space=pl.ANY)
out = pl.pallas_call(dynamic_block_example_kernel,
               in_specs=[hbm_block_spec, hbm_block_spec],
               out_specs=hbm_block_spec,
               out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
               scratch_shapes=(pltpu.SMEM(slices.shape, jnp.int32),)
              )(x, slices)

np.testing.assert_allclose(x, out)
```

### TPUs in Megacore configuration # 

Some TPU chips have two TensorCores but appear as one device to JAX users.
This is called “megacore”.
The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs
and compute units but share HBM . 



Conceptually, TPUs in Megacore behave like very simple GPUs, i.e. they have
only two threads.
How do we modify our kernels to utilize both TensorCores simultaneously? 

The basic idea is that if we have embarrassingly parallel dimensions in our
computation, we can split up those dimensions across the TensorCores.
We can indicate which dimensions are parallelizable by providing an
annotation to pallas_call called dimension_semantics . 

```python
def add_matrices_kernel(x_vmem_ref, y_vmem_ref, z_vmem_ref):
  # Load x and y from VMEM into VREGs
  x_vregs = x_vmem_ref[:, :]
  y_vregs = y_vmem_ref[:, :]
  # Execute a vectorized add
  z_vregs = x_vregs + y_vregs
  # Store the output values in VREGs back into VMEM
  z_vmem_ref[:, :] = z_vregs

def add_matrices_pipelined_megacore(x: jax.Array, y: jax.Array) -> jax.Array:
  block_spec = pl.BlockSpec((256, 512), lambda i: (i, 0))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(2,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )(x, y)

x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices_pipelined_megacore(x, y)
```

```python
Array([[2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       ...,
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
```

dimension_semantics should be a tuple of same length as grid where each
entry is either "parallel" or "arbitrary" . "parallel" indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. "arbitrary" indicates to Pallas that there can be no assumptions made about this grid dimension and it therefore cannot be parallelized. 

By specifying dimension_semantics , we now execute the kernel
simultaneously on each TensorCore. Pallas will handle splitting up the grid
automatically. 

Note that Megacore is only currently available on TPU v4 and TPU v5p . Supplying dimension_semantics annotations is a no-op on other platforms, but not specifying it will result in only one TensorCore being used (even if there are more than one available). 

When using pltpu.emit_pipeline , core_axis should be passed into emit_pipeline . core_axis should be the index of a parallel grid axis to partition the grid on. For example, the following template can be used to partition the kernel over a leading parallel grid dimension: 

```python
def kernel_body(...):
  def inner_pipeline_body(...):
    ...
  pltpu.emit_pipeline(inner_pipeline_body,
                      grid=(4, 4), 
                      core_axis=0,
                      dimension_semantics=("parallel", "sequential"))

pl.pallas_call(
      kernel_body,
      grid=(num_cores,),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",))
  )
```

previous 

Writing TPU kernels with Pallas 

next 

Matrix Multiplication 
Contents TPU and its memory spaces TPU-specific Pipelining Features TPU Memory Spaces Multiple Buffering pltpu.emit_pipeline Lookahead Prefetch Dynamic Block Shapes TPUs in Megacore configuration 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Pallas TPU Matrix Multiplication .ipynb .pdf 
# Matrix Multiplication 

## Contents 
Background Block Matrix Multiplication Tiling and Pipelining Your first matrix multiplication kernel Matrix multiplication performance bfloat16 matrix multiplication Performance of pipelined kernels Templating the matrix multiplication Fused right-hand-side transpose Fused activation function Conclusion 
# Matrix Multiplication # 

In this guide, we’ll write a matrix multiplication routine using Pallas. We’ll also go over how to think about matmul performance on TPU and how to template a matmul kernel to fuse in operations. 

```python
#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
```

## Background # 

Matrix multiplication is a fundamental linear algebra operation at heart of modern deep learning and language modeling. We’d like to make matmuls as speedy as possible using specialized accelerators like TPUs and GPUs, which both have specialized units for fast matrix multiplication. 

To effectively utilize TPUs for matrix multiplication, we’ll need to cover a few background concepts: block matrix multiplication, tiling and pipelining. 

### Block Matrix Multiplication # 

Let’s say we want to implement matmul(x, y) which generically multiplies an (m, k) array with a (k, n) array, but with a twist. We’re only allowed to use the primitive matmul_small which multiples small matrices (say m, k, n <= 256 ). How could we do it? 

A nice property of matrix multiplication is that each block of the output can be expressed as the sum of several smaller matrix multiplications of row blocks and column blocks of the inputs.
Formally, if we have input arrays \(x \in \mathbb{R}^{m \times k}\) and \(y \in \mathbb{R}^{k \times n}\) and output \(z \in \mathbb{R}^{m \times n}\) , we decompose them into blocks along the dimensions of size \(b_m, b_k, b_n\) . 

For example, \(x\) would be decomposed as: 
\[\begin{split}
\begin{bmatrix}
x_{0, 0} & \cdots & x_{0, i_k} \\
x_{1, 0} & \cdots & x_{1, i_k} \\
\vdots & \ddots & \vdots \\
x_{i_m, 0} & \cdots & x_{i_m, i_k} \\
\end{bmatrix}
\end{split}\] 
where \(x_{ik} \in \mathbb{R}^{b_m \times b_k}\) . (We can similarly decompose \(y\) and \(z\) .) 

For a particular output block \(z_{ij}\) , we can compute it as 
\[
z_{ij} = \sum_k x_{ik} y_{kj}
\] 
Therefore, each output block \(z_{ij}\) is the sum of several smaller block matrix multiplications \(x_{ik} y_{kj}\) . Here’s how we’d implement this algorithm in NumPy: 

```python
def matmul_small(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  m, k, n = x.shape[0], x.shape[1], y.shape[0]
  assert m <= 256
  assert k <= 256
  assert n <= 256
  return np.matmul(x, y)

def block_matmul(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bm: int = 256,
    bk: int = 256,
    bn: int = 256,
) -> np.ndarray:
  m, k = x.shape
  _, n = y.shape

  z = np.zeros((m, n), dtype=x.dtype)
  for m_i in range(m // bm):
    for n_i in range(n // bn):
      for k_i in range(k // bk):
        m_slice = slice(m_i * bm, (m_i + 1) * bm)
        k_slice = slice(k_i * bk, (k_i + 1) * bk)
        n_slice = slice(n_i * bn, (n_i + 1) * bn)
        x_block = x[m_slice, k_slice]
        y_block = y[k_slice, n_slice]
        z[m_slice, n_slice] += matmul_small(x_block, y_block)
  return z
```

Our block_matmul function should now work on inputs larger than 256 (though we assume that our input dimensions evenly divide 256). 

```python
m, k, n = 4096, 4096, 4096
x = np.random.uniform(size=(m, k)).astype(np.float32)
y = np.random.uniform(size=(k, n)).astype(np.float32)
np.testing.assert_allclose(x @ y, block_matmul(x, y), atol=1e-6, rtol=1e-6)
```

block_matmul decomposes a matrix multiplication into many smaller ones by observing that each output chunk of size (bm, bn) can be computed by accumulating several (bm, bk) x (bk, bn) size matrix multiplications. 

TPUs and GPUs do matmuls just like this! They natively support small matrix multiplication akin to matmul_small , so to utilize this hardware when doing bigger matrix multiplications, we will apply the block_matmul decomposition. 

### Tiling and Pipelining # 

In the previous guide , we covered how tiling up computations and pipelining in Pallas works. To make sure our compute units are always working and never stalled by memory transfers, we overlap the memory transfers for the next iteration of a kernel with the current one. 

In Pallas, we specify that via BlockSpec s and a grid . Note that we already have a nested for loop in the block matrix multiplication algorithm. We can specify that in Pallas via a grid . The slices in the block matrix multiplication can also be specified via BlockSpec s. 

## Your first matrix multiplication kernel # 

Putting it all together, here’s an implementation of a block matrix multiplication kernel that pipelines the memory transfers with the compute. We create a 3-d grid, corresponding to the 3-nested loop in the NumPy code. Note that while MXUs are only capable of multiplying small blocks, Pallas will automatically take bigger blocks and automatically tile them over the MXUs. 

The last dimension of the grid corresponds to the contraction dimension of the matrix multiply and is a reduction dimension, so we need to be sure to initialize the accumulator. 

```python
def matmul_kernel(x_ref, y_ref, z_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    z_ref[...] = jnp.zeros_like(z_ref)

  z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      grid=(m // bm, n // bn, k // bk),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```python
m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float32)
y = random.normal(k2, (k, n), dtype=jnp.float32)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

## Matrix multiplication performance # 

Let’s think about how to analyze matrix multiplication performance. When we think about matmul performance, we typically care about two things: the total number of floating point operations (FLOPs) and the amount of memory bandwidth usage. From the guide on TPUs and pipelining , we see that in order to use the efficient compute units on TPUs (and on ML accelerators on general), we need to copy our inputs from HBM into VMEM, closer to the compute units. This copying to and from HBM takes time and an efficient kernel hopefully spends most of its time actually computing, as opposed to waiting for these transfers. Memory bandwidth measures the rate of this data transfer. 

Quick note: in this guide, we’ll be discussing floating point operations, but want to make the distinction between FLOPs vs FLOP/s.
When we say “FLOPs” we mean “floating point operations”, as in a number of operations. When we say “FLOP/s”, we refer to “floating point operations per second ”, as in a rate of performing floating point operations. 

The number of FLOPs in a (m, k) x (k, n) matrix multiplication are (approximately) 2 * m * k * n . (Technically it is n * m * (2k - 1) but for large enough k our approximation is sufficient.) 

The minimum amount of memory bandwidth usage for a matrix multiply (assuming float32) is the total size of the inputs (copying into VMEM) plus the size of the output (copying into HBM). Thus the minimum bandwidth usage is (m * k + k * n + m * n) * 4 bytes/float32 . Memory usage can be greater if we re-read the inputs multiple times, which is often the case. 

One observation is that the number of matmul FLOPs is cubic in its inputs whereas the minimum bandwidth usage is quadratic in its inputs. Intuitively, this means that FLOPs grow faster than bandwidth usage, meaning that the bigger our matmul is, the more compute we have relative to copying. 

```python
def matmul_flops(m: int, k: int, n: int):
  return 2 * m * k * n

def matmul_membw(m: int, k: int, n: int, dtype: jnp.dtype):
  return (m * k + k * n + m * n) * np.dtype(dtype).itemsize

print(matmul_flops(1024, 1024, 1024))
print(matmul_membw(1024, 1024, 1024, jnp.float32))
```

```python
2147483648
12582912
```

Now that we can calculate the total number of FLOPs and (minimum) memory bandwidth usage of a matrix multiplication, let’s see what a real TPU can handle. 

This notebook was run on a TPU v5e chip so we’ll use the v5e numbers (if you are running this notebook, your numbers may differ). TPU v5es have 197 TFLOP/s of bf16/f32 compute and 819 GB/s of memory bandwidth . By looking at the ratio of these numbers (called the arithmetic intensity), we can get a bound on how low this “FLOPs / memory bandwidth usage” ratio can get before we become IO bound (about 240 FLOPs/byte on TPU v5e). 

```python
v5e_flops = 197e12
v5e_membw = 819e9
v5e_op_intensity = v5e_flops / v5e_membw  # ~240.5
```

Roughly, these numbers tell us the FLOPs of a matmul should take 2 * m * k * n / (197 TFLOP/s) seconds and the copies to/from VMEM should take (m*k + k*n + m*n) * 4 bytes / 819GB/s seconds. 

```python
def matmul_flops_intensity(m: int, k: int, n: int, dtype: jnp.dtype):
  flops = matmul_flops(m, k, n)
  membw = matmul_membw(m, k, n, dtype)
  return flops / membw
```

This basic calculation tells us roughly how efficiently we’ll be able to use our MXUs. If our matmul op intensity is below what our chip is capable of, then our computation will be memory bound , i.e. our compute units will be idling while waiting for values to be transferred. If the matmul intensity is higher than what the chip is capable, then we will be compute bound . 

Because matmul FLOPs are cubic in their input sizes and memory bandwidth usage is quadratic, we expect that we will get compute bound as we get bigger and bigger, but this crossing over point is really important! Let’s say we are doing a (1024, 1024) x (1024, 1024) float32 matrix multiplication. 

```python
print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.float32)} flops/byte")
```

```python
170.66666666666666 flops/byte
```

Our matmul flops intensity is less than what our chip is capable of. That’s not good! We are likely going to be memory bound with this type of matrix multiplication. However, what if our inputs and outputs were bigger instead?  At some point when our matmuls get big enough, we will cross over from memory bound into compute bound. For example, if we have a matmul where m = k = n , we will cross over (on TPU v5e) when 2m**3 / 12m**2 > 240 or when m = k = n > 1440 . 

### bfloat16 matrix multiplication # 

To make it easier for matrix multiplication to be compute bound on TPU, we could also use a smaller dtype for our inputs and outputs. Our previous example used float32 inputs and outputs but TPU v5e also supports the bfloat16 data type (a 16-bit floating point format, also called bf16 ) for matrix multiplication as well. On TPU v5e, we will have the same FLOP/s but will halve our memory bandwidth usage . This makes it way easier to be compute bound for smaller matrices. Let’s see what our intensity is with a 1024 x 1024 x 1024 bf16 matrix multiply: 

```python
print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.bfloat16)} flops/byte")
```

```python
341.3333333333333 flops/byte
```

We now have a matmul that is compute bound! 

Let’s add bf16 support to our matrix multiplication kernel. 

The native MXU bf16 matmul routine takes two input bf16 matrices and accumulates it in f32 . We will trigger this routine by passing preferred_element_type=jnp.float32 into jnp.matmul . We will also need a accumulator Ref that is in f32 . We will then downcast the output back to bf16 before writing it back to HBM. This way we don’t lose any precision, don’t do any extra casting, and still retain the bf16 memory bandwidth savings. 

Note that the only way of allocating scratch spaces right now is via pltpu.PrefetchScalarGridSpec . Don’t worry about exactly what it does for now – all you need to know for now is that it allows you to allocate scratch spaces in VMEM. 

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```python
m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.bfloat16)
y = random.normal(k2, (k, n), dtype=jnp.bfloat16)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

## Performance of pipelined kernels # 

Our above analysis about FLOPs vs memory usage applies at a coarse scale i.e. when we are looking at the the size of a the total matrix multiplication. However, remember that in practice, we are pipelining the execution of a blocked matrix multiplication, meaning we have a loop in which we are doing matrix multiplication with smaller blocks. 

This means that we actually care about the FLOPs vs memory bandwidth usage of each individual instance of the kernel, not the global FLOPs vs memory bandwidth usage. 

In addition, when tiling the matmul operation, the same values could be read multiple times from memory.
Specifically the memory bandwidth for the first operand of the kernel is (bm * bk) , which needs to be multiplied by the grid dimensions, that is (bm * bk) * m // bm * n // bn * k // bk = m * k * n // bn .
Similarly for the second operand, yielding a total bandwidth usage (m * k * n // bn + k * n * m // bm + m * n) * element_size . 

Therefore, the block sizes bm , bk , bn are extremely important for performance.
Even if we have the largest matrices in the world, if we pick very small bm , bk , and bn , we will be memory bound because each time we invoke the kernel we will have too few FLOPs to hide the memory transfers happening in the background. 

The intuition should therefore be: to be compute bound, make the blocks as big as possible! There are two main constraints: 

VMEM usage: The bigger our blocks, the more VMEM we use. With large enough blocks, we will run out. 

Pipeline bubbles: The larger our blocks are relative to the matrix size, the fewer loop iterations we will have in our pipeline. This will make the size of the bubbles at the beginning and end of the pipeline larger relative to the total pipeline and this overhead can be nontrivial. 

Getting good matrix multiplication performance in Pallas boils down to picking good block sizes to balance this optimization problem. In practice, we often sweep over a large set of candidate block sizes, profile the kernel, and pick the best one. 

For now, let’s do some very simple timing experiments. We’ll use timeit to measure the amount of time running each kernel takes. Note that this is a upper bound on the actual runtime of the kernel since we are measuring Python dispatch and other overheads using timeit . We’ll compute the amount of FLOP/s we obtained this way and compute the percentage utilization we get compared to what the chip offers and we’ll use some reasonable block sizes to verify our intuition. 

```python
import timeit

def benchmark(f, ntrials: int = 100):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    # print(f"Time: {time}")
    return time
  return run

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func):
  x = jnp.ones((m, k), dtype=dtype)
  y = jnp.ones((k, n), dtype=dtype)
  time = benchmark(mm_func)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

```python
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00029766598949208854
Matmul FLOP/s:  7214407167121.377
FLOP/s utilization: 3.6621%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.011771515250438824
Matmul FLOP/s:  11675553278230.387
FLOP/s utilization: 5.9267%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09183577066054567
Matmul FLOP/s:  11972585626140.668
FLOP/s utilization: 6.0775%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012708659982308746
Matmul FLOP/s:  16897797651282.135
FLOP/s utilization: 8.5776%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.00088908776990138
Matmul FLOP/s:  154584235803001.88
FLOP/s utilization: 78.4692%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006099433819763363
Matmul FLOP/s:  180264539343531.62
FLOP/s utilization: 91.5048%
```

Bigger block sizes help a lot! We get pretty good utilization (80-90%) in the bigger matmuls, but the smallest matmul seems pretty hard to get good performance with. 

Let’s compare this with XLA’s matmuls. We don’t expect Pallas to do better than XLA because XLA is very good at generating matmuls but hopefully we are close.
With more careful block size tuning (left as future work), we can also reach XLA performance. 

```python
print("================ XLA matmul ===================")
mm = jnp.matmul
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

```python
================ XLA matmul ===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00011943008983507753
Matmul FLOP/s:  17981093801113.996
FLOP/s utilization: 9.1275%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.0008272899803705514
Matmul FLOP/s:  166131533963991.34
FLOP/s utilization: 84.3307%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006047147869830951
Matmul FLOP/s:  181823175395037.44
FLOP/s utilization: 92.2960%
```

Pallas, with some very basic tuning, gets pretty close to XLA’s performance numbers! By trying out more block sizes, we should expect to close the gap entirely. 

## Templating the matrix multiplication # 

Now that we have a basic matrix multiplication kernel, we can now try fusing operations into it. 

### Fused right-hand-side transpose # 

A common first thing to do is to fuse a transpose. What do we mean by that? Suppose we wanted to compute x @ y.T instead of x @ y . Naively we could first compute y.T and then pass it into our efficient matrix multiply kernel. However, the operation y.T is not free on its own – it involves copying O(n^2) data. Ideally, we could compute the transpose while doing the matrix multiply in just one kernel, i.e. “fusing” it with the matmul. 

Accelerators often support native matrix multiplication routine that fuse a RHS transpose. For instance TPU v5e, the MXU allows us to do x @ y.T for small arrays. We can invoke this routine with jax.lax.dot_general , which will be more efficient than doing a transpose then a matmul separately. 

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  # dot_general expects a data structure (contraction_dims, batch_dims),
  # where contraction_dims are the set of dimensions for LHS and RHS that will
  # be contracted (reduced) in the matmul; batch_dims, on the other hand, are
  # looped over. The remaining dimensions will be the input and output dimension
  # of the matmul.
  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...], y_ref[...], dims, preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'transpose_rhs'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk, transpose_rhs=transpose_rhs),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            y_block_spec,
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

We do a transpose inside of the matmul function ( y = y.swapaxes(0, 1) ). This is because inside of a JIT-ted JAX computation, dimension ordering is purely logical , not physical, so rearranging dimensions does not imply a
physical layout difference. However, when we pass an array into a pallas_call , we do enforce a major-to-minor dimension ordering constraint. By transposing y inside of the matmul function, we are requesting that y be in a
transposed layout (n, k) instead of the usual (k, n) . The user will still pass in the array in the (logical) (k, n) dimension, however. 

Note: to benchmark the transpose, we actually want y to be in the physical transposed layout when we pass it into the kernel, so we don’t measure relayout time. In the wrapper function, we will (logically) transpose it back to (k, n) before passing it into matmul because matmul expects a logical (k, n) dimension ordering. 

```python
def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False):
  x = jnp.ones((m, k), dtype=dtype)
  if transpose_rhs:
    y = jnp.ones((n, k), dtype=dtype)
    @jax.jit
    def _wrapper(x, y):
      y = y.swapaxes(0, 1)
      return mm_func(x, y, transpose_rhs=True)
  else:
    y = jnp.ones((k, n), dtype=dtype)
    _wrapper = mm_func
  time = benchmark(_wrapper)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)
```

```python
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.0003029372810851783
Matmul FLOP/s:  7088872126624.065
FLOP/s utilization: 3.5984%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.012017967159627005
Matmul FLOP/s:  11436123235026.848
FLOP/s utilization: 5.8051%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09500920018996112
Matmul FLOP/s:  11572685861765.383
FLOP/s utilization: 5.8745%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012131539988331496
Matmul FLOP/s:  17701657415839.363
FLOP/s utilization: 8.9856%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.0008790623804088682
Matmul FLOP/s:  156347213275211.03
FLOP/s utilization: 79.3641%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006107717020204291
Matmul FLOP/s:  180020067095253.78
FLOP/s utilization: 91.3807%
```

See how we get the same utilization despite the extra transpose! 

### Fused activation function # 

Fusing in an activation is also really common. This makes sure we don’t follow an efficient, compute bound matmul kernel with a slow memory bound activation kernel. 

```python
def matmul_kernel(
    x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs, activation
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...],
      y_ref[...],
      dims,
      preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'activation'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(
          matmul_kernel,
          nsteps=k // bk,
          transpose_rhs=transpose_rhs,
          activation=activation,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              y_block_spec,
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```python
def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False,
                   activation = lambda x: x):
  x = jnp.ones((m, k), dtype=dtype)
  if transpose_rhs:
    y = jnp.ones((n, k), dtype=dtype)
    @jax.jit
    def _wrapper(x, y):
      y = y.swapaxes(0, 1)
      return mm_func(x, y, transpose_rhs=True, activation=activation)
  else:
    y = jnp.ones((k, n), dtype=dtype)
    _wrapper = functools.partial(mm_func, activation=activation)
  time = benchmark(_wrapper)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()


activation = jax.nn.relu
print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)
```

```python
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00030103540048003196
Matmul FLOP/s:  7133658182976.541
FLOP/s utilization: 3.6211%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.011807117109419778
Matmul FLOP/s:  11640348122095.826
FLOP/s utilization: 5.9088%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09181861146935262
Matmul FLOP/s:  11974823079773.941
FLOP/s utilization: 6.0786%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012622540001757442
Matmul FLOP/s:  17013086492108.6
FLOP/s utilization: 8.6361%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.000896632740041241
Matmul FLOP/s:  153283442968721.44
FLOP/s utilization: 77.8089%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006130605939542875
Matmul FLOP/s:  179347953304919.88
FLOP/s utilization: 91.0396%
```

The additional fused activation barely affects our utilization at all! 

## Conclusion # 

In this guide, we covered how to write efficient matrix multiplications on TPU using Pallas. We discussed blocked matrix multiplication and pipelining, how to analyze the performance of a TPU matmul, and how to write an efficient bf16 matrix multiplication. We concluded with templating the matrix multiplication to support a fused transpose and fused activation functions. 

Exercises left to the reader: 

Add support for input fusions. Sometimes we want to fuse an operation into the inputs of the matmul. Try templating the matrix multiplication even more to support this. 

Add support for int8 matrix multiplication. TPU v5 supports native int8 matrix multiplication at twice the FLOPs of bf16 . Try adding support for that and see what utilization is possible. 

Add backwards pass support for the matmul function. You can do this with jax.custom_vjp . 

previous 

TPU Pipelining 

next 

Scalar Prefetch and Block-Sparse Computation 
Contents Background Block Matrix Multiplication Tiling and Pipelining Your first matrix multiplication kernel Matrix multiplication performance bfloat16 matrix multiplication Performance of pipelined kernels Templating the matrix multiplication Fused right-hand-side transpose Fused activation function Conclusion 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



Pallas: a JAX kernel language Pallas TPU Scalar Prefetch and Block-Sparse Computation .ipynb .pdf 
# Scalar Prefetch and Block-Sparse Computation 

## Contents 
Dynamic Block Indexing with Scalar Prefetch Example: Block Dynamic Slice with Scalar Prefetch Sparse Kernels: Representing Sparse Data Example: Sparse @ Dense Matrix Multiplication Sparse Access Patterns on Dense Data Example: Dense @ Dense Matrix Multiplication with a Block-Sparse Output Mask 
# Scalar Prefetch and Block-Sparse Computation # 

In this tutorial, we will cover the basics of block-sparse computing in Pallas. Sparse computation is a major reason to write custom Pallas kernels over simply using JAX/XLA, since it is generally difficult to express programs that perform a dynamic amount of computation in XLA due to static array shapes. In this tutorial we will learn how to use the scalar prefetch feature of Pallas in order to write block-sparse kernels that can dynamically skip over computation and blocks of memory. 

```python
import functools
import timeit
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import checkify
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print("Running on", jax.devices()[0].device_kind)
```

```python
Running on TPU v5 lite
```

## Dynamic Block Indexing with Scalar Prefetch # 

We will be exploiting the “scalar prefetch” feature of Pallas to enable us to write sparse kernels. Scalar prefetch allows you to pass in a small amount of data into SMEM (“scalar memory”) that is loaded before the start of the pipeline (“prefetch”). Because this data is loaded before the pipeline, it is available for use in the index_map for each BlockSpec, allowing you to perform data-dependent indexing calculations. The main goal of this tutorial is to go over common programming patterns that utilize this feature. 

To use scalar prefetch, use pltpu.PrefetchScalarGridSpec in place of the standard pl.GridSpec : 

```python
class PrefetchScalarGridSpec:
  def __init__(self,
    num_scalar_prefetch: int,
    grid: tuple[int, ...],
    in_specs: PyTree[BlockSpec],
    out_specs: PyTree[BlockSpec],
    scratch_shapes: tuple[MemorySpace, ...]):
      ...
```

The num_scalar_prefetch parameter indicates the number of scalar prefetch values. When this is set to a non-zero value, it changes the call signature of the kernel and index maps to expect additional prefetch values. The prefetch Ref s passed in to the index_map and kernel are all allocated in SMEM and are not partitioned into blocks as they do not have a BlockSpec defined. Moreover, the order of arguments to both index_map and kernel are always fixed and described below: 

Each BlockSpec ’s index_map now expects the prefetch Ref s to come after the grid indices: 

```python
def index_map(*grid_indices, *prefetch_refs):
    ...
```

The user-defined kernel expects prefetch Ref s to come before the input Ref s. Additionally, the scratch refs come after the output Ref s. 

```python
def kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs):
    ...
```

When calling a new kernel using pallas_call , the function returned by pallas_call also expects the scalar prefetch arguments to come before the inputs, e.g. 

```python
kernel = pl.pallas_call(...)
result = kernel(*prefetch_args, *input_args)
```

## Example: Block Dynamic Slice with Scalar Prefetch # 

Let’s begin with a basic example that demonstrates how to use the scalar prefetch feature. We will implement a block-aligned dynamic slice kernel which simply extracts a block out of larger array based on user-specified indices: 

Outside of the kernel, we compute the block index to extract as: block_idx = (start[0] // size[0], start[1] // size[1]) 

We pass block_idx as a scalar prefetch argument into pallas_call . 

In our index map, we use the block index to select the corresponding block by returning (block_idx[0], block_idx[1]) . 

Of course, this kernel is limited in that our slice sizes must fit inside of a kernel block (limited by VMEM size) and we can only start on size-aligned indices. A more advanced kernel would decouple the kernel block size with the slice size and allow non-aligned start indices. 

```python
def dynamic_slice_kernel(indices, x_ref, o_ref):
  del indices
  o_ref[...] = x_ref[...]

@checkify.checkify
@functools.partial(jax.jit, static_argnums=(2,))
def block_dynamic_slice(x, starts, sizes):
  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(1, 1),
      in_specs=[pl.BlockSpec(
          sizes,
          lambda i, j, block_idx: (block_idx[0], block_idx[1]))],
      out_specs=pl.BlockSpec(sizes, lambda *_: (0, 0)),
  )

  kernel = pl.pallas_call(
    dynamic_slice_kernel,
    grid_spec=grid_spec,
    out_shape=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
  )
  # Checkify inserts a runtime assert that starts are divisible by block size.
  checkify.check(starts[0] % sizes[0] == 0, "Starts must be divisible by size.")
  checkify.check(starts[1] % sizes[1] == 0, "Starts must be divisible by size.")
  block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]])
  return kernel(block_idx, x)

shape = (512, 512)
x = jnp.reshape(jnp.arange(np.prod(shape), dtype=jnp.int32), shape)
err, result = block_dynamic_slice(x, starts=(128, 256), sizes=(128, 128))
err.throw()
ref = lax.dynamic_slice(x, start_indices=(128, 256), slice_sizes=(128, 128))
diff = jnp.max(jnp.abs(result - ref))
print("Error |result - lax.dynamic_slice| =", diff)
```

```python
Error |result - lax.dynamic_slice| = 0
```

## Sparse Kernels: Representing Sparse Data # 

Before we dive into implementing sparse kernels, let’s first review how sparse matrices are represented. While there are several popular formats for storing sparse matrices, we will be following a blocked variant of the coordinate-list format (COO) in which we will store a matrix as a list of (block_index, block_data) pairs. All blocks that are not explicitly stored in the list are assumed to be zero, meaning we can save a significant amount of memory if there are many zero blocks in the matrix. 

The following figure demonstrates how we convert a 4x4 dense matrix (left) into a block-COO format (right) with a block size of 2x2. Note that in the sparse format, we can avoid explicitly storing the upper-right block which consists of all zero elements. 



We will use the following helper function to sample a block-sparse matrix. It returns a dense matrix used for checking our results, as well as a list of block data and indices for each axis. 

```python
def generate_block_sparse_mat(key, M, N, blk_M, blk_N, p=0.2, dtype=jnp.float32):
  """Returns a sampled matrix and its block-sparse representation.

  Args:
    key: RNG Key.
    M: Major array dimension.
    N: Minor array dimension.
    blk_M: Block size along M dimension.
    blk_N: Block size along N dimension.
    p: Probability that a block will be non-zero.
    dtype: dtype of the sampled matrix.

  Returns:
    dense_mat: A (M, N) dense sampled array.
    block_data: A (num_blocks, blk_M, blk_N) array of data blocks representing
      the non-zero blocks of the matrix.
    indices_i: A (num_blocks,) array of block indices for the first axis.
    indices_j: A (num_blocks,) array of block indices for the second axis.
  """
  mask_key, blocks_key = jax.random.split(key)
  num_blocks = (M // blk_M, N // blk_N)
  # We first sample a block mask, denoting which blocks are nonzero.
  block_mask = jax.random.bernoulli(mask_key, p=p, shape=num_blocks)
  num_blocks = jnp.sum(block_mask)
  indices = jnp.where(block_mask)
  # For each non-zero block, we sample a block of random values.
  block_data = jax.random.uniform(blocks_key,
                                  shape=(num_blocks, blk_M, blk_N),
                                  dtype=dtype)
  # For checking purposes, create the dense version of the sparse matrix.
  dense_mat = jnp.zeros((M, N), dtype=dtype)
  for blk in range(num_blocks):
    idx_i = indices[0][blk]
    idx_j = indices[1][blk]
    slice_i = slice(idx_i * blk_M, (idx_i + 1) * blk_M)
    slice_j = slice(idx_j * blk_N, (idx_j + 1) * blk_N)
    dense_mat = dense_mat.at[slice_i, slice_j].set(block_data[blk])
  return dense_mat, block_data, indices[0], indices[1]
```

## Example: Sparse @ Dense Matrix Multiplication # 

In our first example, we will multiply a sparse LHS matrix with a dense RHS matrix to produce a dense output. 

We will structure our kernel grid with 2 loops - the outer loop over the columns of the RHS/output, and inner loop over the sparse blocks of the LHS. During each inner loop iteration, we load one block from the LHS and lookup the corresponding block on in the RHS using the block index of the contracting dimension (K). We multiply the two blocks together and accumulate into the correct output block. One outer loop iteration will compute a result for an entire column as depicted by the following diagram: 



It is important that we group the block indices by row (e.g. [0, 0, 1, 2, 3, 3] ) before we pass them into the kernel for two reasons. First, in our kernel we need to know when to initially zero-out the accumulator in the output ref, and it is easy to do so if the row indices are grouped. Second, the pipelining logic for Pallas does not allow us to re-visit blocks in the output Ref on non-consecutive iterations, and therefore we need to do all accumulation into an output block in consecutive kernel iterations. This is because the pipeline emitter will realize that we are loading the same output block on consecutive iterations and keep the block in VMEM. When we change output block Pallas will finally store the output into HBM and assume we never touch it again. Failure to access output blocks consecutively will result in incorrect values even though the kernel is otherwise logically correct. 

```python
M = N = K = 16384
blk_M = blk_N = blk_K = 512


def dsd_kernel(idxs_i_ref, idxs_k_ref, # Scalar prefetch inputs.
               x_ref, y_ref, _, o_ref, # Kernel inputs.
               accum_scratch,
               ):
  """A DSD (Dense = Sparse @ Dense) matmul kernel."""
  del idxs_k_ref
  blk_idx = pl.program_id(1)
  is_start = blk_idx == 0
  changed_blocks = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx-1, 0)])
  @pl.when(is_start | changed_blocks)
  def _():
    accum_scratch[...] = jnp.zeros_like(accum_scratch)
  accum_scratch[...] += jnp.dot(x_ref[0, :, :], y_ref[...], preferred_element_type=jnp.float32)

  next_block_change = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.minimum(blk_idx+1, num_blocks)])
  is_end = blk_idx == (num_blocks - 1)
  @pl.when(is_end | next_block_change)
  def _():
    o_ref[...] = accum_scratch[...].astype(o_ref.dtype)


def x_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del j, blk_idxs_i, blk_idxs_k
  return (blk_idx, 0, 0)
def y_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del blk_idxs_i
  return (blk_idxs_k[blk_idx], j)
def o_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  del blk_idxs_k
  return (blk_idxs_i[blk_idx], j)

(X_dense, X_blocks, indices_i, indices_k) = generate_block_sparse_mat(
    jax.random.key(0), M, K, blk_M, blk_K, p=0.1, dtype=jnp.bfloat16)
num_blocks = X_blocks.shape[0]
Y = jax.random.uniform(jax.random.key(1), shape=(K, N), dtype=jnp.bfloat16)
zeros = jnp.zeros((M, N), dtype=jnp.bfloat16)
out_shape = jax.ShapeDtypeStruct((M, N), dtype=jnp.bfloat16)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=2,
    # Note that while num_blocks is static here, Pallas does support
    # dynamic grid sizes.
    grid=(N // blk_N, num_blocks),
    in_specs=[pl.BlockSpec((1, blk_M, blk_K), x_map),
              pl.BlockSpec((blk_K, blk_N), y_map),
              # Placeholder for a zeros-array used by input_output_aliases.
              pl.BlockSpec((blk_M, blk_N), o_map),
              ],
    out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
    scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)]
)
kernel = pl.pallas_call(
  dsd_kernel,
  grid_spec=grid_spec,
  out_shape=out_shape,
  # We use input-output aliases to zero-out o_ref for blocks that we never
  # visit. By passing in an array of zeros we avoid having o_ref start with
  # uninitialized values.
  input_output_aliases={4: 0},  # Map zeros to o_ref.
)
args = (indices_i, indices_k, X_blocks, Y, zeros)
result = kernel(*args)

ref = X_dense @ Y
diff = jnp.abs(ref - result)
print('mean |result - ref|:', jnp.mean(diff))
```

```python
mean |result - ref|: 0
```

We can do a quick benchmark to compare the performance of our sparse kernel compared to a dense matmul in JAX. On a TPU v5e chip, this kernel achieves a roughly ~6x speed increase compared to the theoretical 10x from the sparsity factor. 

There are a few main tips for performance here, mainly centered around reducing the communication overhead between HBM/VMEM: 

Using dtype=jnp.bfloat16 is critical for performance since it reduces memory bandwidth by half. 

Using larger block sizes also helps, since matrix multiply is an \(O(N^3)\) compute and \(O(N^2)\) memory operation. As \(N\) grows larger, the kernel becomes compute-bound. However, a counter-argument to this in practice is that smaller block sizes also enables data to be more sparse, so this is a parameter that should be selected carefully. 

```python
# Benchmark Sparse Pallas kernel vs reference JAX implementation

def benchmark(f, ntrials: int = 100):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    return time
  return run


n_trials = 100

pallas_impl = lambda *args: kernel(*args)
time = benchmark(pallas_impl, n_trials)(indices_i, indices_k, X_blocks, Y, zeros)
print("Sparse Kernel: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))

ref_impl = jax.jit(lambda x, y: x @ y)
time = benchmark(ref_impl, n_trials)(X_dense, Y)
print("Reference: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))
```

```python
Sparse Kernel: 8.136 ms (avg over 100 trials)
Reference: 46.953 ms (avg over 100 trials)
```

## Sparse Access Patterns on Dense Data # 

In our previous example we considered the case when the data itself is sparse. This manifested itself in the kernel structure as a dimension in the kernel grid that was dynamic and looped over the number of nonzero blocks ( num_blocks ). 

A second useful programming pattern emerges when the underlying data is dense, but we wish to perform sparse computation over it. Our kernel grid in this case will be dense, but we wish to skip over some blocks in the grid as indicated by a block-sparse mask. This type of programming pattern commonly arises when using masks in many machine learning applications, such as causal or local masks in self-attention. In these cases, we can entirely skip over computation in blocks where the mask is zeroed-out. Examples of this programming pattern can be found in the Splash Attention and Grouped Matrix Multiplication kernels located in jax/experimental/pallas/ops/tpu , or in PyTorch’s FlexAttention . 

The main performance consideration with dealing with a sparse access pattern on dense data is the interaction with pipelining. On any given kernel iteration, the Pallas pipeline emitter will attempt to prefetch the next block of data by calling the index_map for each BlockSpec on the next iteration of the grid. However, if our computation is sparse we may be skipping the computation for the next block in the grid, so we need some method to tell the pipeline instead begin fetching the next block that we are not skipping . In order to do this, we need to construct prefetch maps which contains indices to the next non-skipped block of data for each kernel input. The following diagram illustrates how a prefetch map could be constructed for a block-sparse mask that is stored in a COO-like format. 



Left: A sparse access pattern, where the color blue denotes blocks with non-zero masks that we need to compute. Right: The prefetch map, where each element of the array contains the index of the next non-zero block data. 

Once the prefetch map has been constructed, we can pass the map as a scalar prefetch argument and query it in the index_map function of the BlockSpec. 

```python
def mask_index_map(prefetch_map, i, j, ...):
  next_nonzero_block = prefetch_map[i, j]
  return (next_nonzero_block, 0, 0)
```

We can construct similar index maps for the other inputs to the kernel. For dense inputs you will most likely need to construct prefetch maps which point to the next non-zero block index in the grid. Our next example will provide an example of using these prefetch maps. 

## Example: Dense @ Dense Matrix Multiplication with a Block-Sparse Output Mask # 

In our next example we will cover dense matrix multiplication fused with a sparse output mask using a prefetch map to improve pipelining performance. We will use the mask to selectively skip computing output blocks that are zeroed-out, therefore saving on computation costs. 

As we will be working with a sparse mask, we will begin by implementing a function that converts an N x M mask stored in dense format into a block-sparse format. We additionally need to compute prefetch maps to help the pipeline emitter know which block to fetch next. In total, our sparsify_mask function computes: 

A block_mask of shape (num_N_blocks, num_M_blocks) indicating if a block is all-zeros (value 0 ) or contains non-zero elements (value 1 ). If the block_mask has a value of 0 we can skip computing the block in the kernel. 

A prefetch_mask array of shape (num_N_blocks, num_M_blocks) consisting of indices into mask_data for the next non-zero block. 

A prefetch_i array of shape (num_N_blocks, num_M_blocks) consisting of the next non-masked i index of the mask. 

A prefetch_j array of shape (num_N_blocks, num_M_blocks) consisting of the next non-masked j index of the mask. 

A mask_data array of shape (num_blocks, blk_N, blk_M) containing data for non-zero blocks of the mask. 

```python
def sparsify_mask(mask: jax.Array,
                  block_shape: tuple[int, int]):
  """Preprocesses a mask into a sparse representation.

  Args:
    mask: A boolean array of shape [M, N]
    block_shape: The size of a single block.

  Returns:
    block_mask: A block_shape array of booleans indicating whether a block
      is all-zeros (0) or contains non-zero elements (1).
    prefetch_mask: A block_shape array of integers indicating the index of the
      next non-zero block.
    mask_data: A (num_blocks, block_shape) array containing
      the data for non-zero blocks of the mask.
  """
  M, N = mask.shape
  bm, bn = block_shape

  block_mask = jnp.zeros((M // bm, N // bn), dtype=mask.dtype)
  mask_types_finder = []
  mask_data = []

  next_mask_type_idx = 0
  prefetch_mask = jnp.zeros_like(block_mask)
  next_i = (M // bm) - 1
  next_j = (N // bn) - 1
  prefetch_i = jnp.zeros_like(block_mask)
  prefetch_j = jnp.zeros_like(block_mask)
  for i in range(M // bm, -1, -1):
    for j in range(N // bn, -1, -1):
      mask_block = mask[i * bm :(i + 1) * bm,
                        j * bn :(j + 1) * bn]
      is_nonzero = jnp.any(mask_block)
      if is_nonzero:
        try:
          type_index = mask_types_finder.index(str(mask_block))
        except ValueError:
          type_index = len(mask_types_finder)
          mask_types_finder.append(str(mask_block))
          mask_data.append(mask_block)
        next_mask_type_idx = type_index
        next_i = i
        next_j = j
      else:
        type_index = -1
      block_mask = block_mask.at[i, j].set(is_nonzero)
      prefetch_mask = prefetch_mask.at[i, j].set(next_mask_type_idx)
      prefetch_i = prefetch_i.at[i, j].set(next_i)
      prefetch_j = prefetch_j.at[i, j].set(next_j)
  return block_mask, prefetch_mask, prefetch_i, prefetch_j, jnp.stack(mask_data)
```

In terms of the structure of the kernel, we use the same grid pattern as the standard matrix multiplication kernel we covered in previous tutorials with a 3 loops over the N , M , and K dimensions. Within the kernel itself, we first check the block_mask to see if the mask for the current output block was all zeros. If the mask is all zeros, we can skip computation and move onto the next block; otherwise we need to compute the matrix multiplication and then mask the result. 

```python
M = N = K = 16384
blk_M = blk_N = 512
blk_K = 1024

def sparse_mask_matmul(
    block_mask_ref, prefetch_mask, prefetch_i, prefetch_j, # Scalar prefetch inputs.
    x_ref, y_ref, mask_ref, o_ref,  # Kernel inputs.
    accum_scratch
    ):
  del prefetch_mask, prefetch_i, prefetch_j
  i, j, k = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  should_compute = block_mask_ref[i, j] != 0
  @pl.when(k == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
    accum_scratch[...] = jnp.zeros_like(accum_scratch[...])

  # We only compute the output for blocks with non-zero masks.
  # Otherwise we skip the computation entirely.
  @pl.when(should_compute)
  def _():
    result = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    accum_scratch[...] += result
    @pl.when(k == pl.num_programs(2) - 1)
    def _():
      o_ref[...] = (mask_ref[0, ...] * accum_scratch[...]).astype(o_ref.dtype)

X = jax.random.normal(jax.random.key(0), shape=(M, K), dtype=jnp.bfloat16)
Y = jax.random.normal(jax.random.key(1), shape=(K, N), dtype=jnp.bfloat16)
mask = jnp.ones((M, N), dtype=jnp.int32)
mask = jnp.tril(mask)
block_mask, prefetch_mask, prefetch_i, prefetch_j, sparse_mask_data = sparsify_mask(mask, (blk_M, blk_N))

def x_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
  del prefetch_mask, prefetch_j
  # Zero-out the k index if the mask is zero, to avoid constantly fetching
  # new blocks in the inner loop for blocks we are skipping.
  k_fetch = (block_mask[i, j] != 0) * k
  return (prefetch_i[i, j], k_fetch)

def y_map(i, j, k, block_mask, prefetch_mask, prefetch_i, prefetch_j):
  del prefetch_mask, prefetch_i
  k_fetch = (block_mask[i, j] != 0) * k
  return (k_fetch, prefetch_j[i, j])

def mask_map(i, j, k, block_mask, prefetch_mask, *_):
  del k, block_mask
  return (prefetch_mask[i, j], 0, 0)

def o_map(i, j, k, *_):
  del k
  return (i, j)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=4,
    grid=(M // blk_M, N // blk_N, K // blk_K),
    in_specs=[pl.BlockSpec((blk_M, blk_K), x_map),
              pl.BlockSpec((blk_K, blk_N), y_map),
              pl.BlockSpec((1, blk_M, blk_N), mask_map)],
    out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
    scratch_shapes=[pltpu.VMEM((blk_M, blk_N), dtype=jnp.float32)]
)
kernel = pl.pallas_call(
  sparse_mask_matmul,
  grid_spec=grid_spec,
  out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
)
args = (block_mask, prefetch_mask, prefetch_i, prefetch_j, X, Y, sparse_mask_data)
result = kernel(*args)

ref = mask * (X @ Y)
diff = jnp.abs(ref - result)
print('mean |result - ref|:', jnp.mean(diff))
```

```python
mean |result - ref|: 1.0252e-05
```

Now let’s compare performance versus a naive dense implementation. On TPU v5e, we achieve around a ~1.8x speed increase with the sparse kernel, compared to a theoretical best-case of 2x from using a lower triangular mask and only visiting half of the possible outputs. 

We would generally expect performance to get closer to the theoretical peak as our inputs get larger, since a few of the main reasons why we don’t exactly reach theoretical performance are: 

We skip slightly less than half of computation since the blocks along the diagonal are mixed 0s and 1s, and for mixed blocks we need to compute the entire block. With larger inputs, our overhead for mixed blocks becomes smaller relative to the overall computation. 

The pipeline bubble also accounts for a less percentage of the overall runtime as inputs become larger. 

```python
n_trials = 100

pallas_impl = lambda *args: kernel(*args)
time = benchmark(pallas_impl, n_trials)(block_mask, prefetch_mask, prefetch_i, prefetch_j, X, Y, sparse_mask_data)
print("Sparse Kernel: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))

ref_impl = jax.jit(lambda mask, x, y: mask * (x @ y))
time = benchmark(ref_impl, n_trials)(mask, X, Y)
print("Reference: %.3f ms (avg over %d trials)" % (time * 1000, n_trials))
```

```python
Sparse Kernel: 28.648 ms (avg over 100 trials)
Reference: 49.988 ms (avg over 100 trials)
```

previous 

Matrix Multiplication 

next 

Distributed Computing in Pallas for TPUs 
Contents Dynamic Block Indexing with Scalar Prefetch Example: Block Dynamic Slice with Scalar Prefetch Sparse Kernels: Representing Sparse Data Example: Sparse @ Dense Matrix Multiplication Sparse Access Patterns on Dense Data Example: Dense @ Dense Matrix Multiplication with a Block-Sparse Output Mask 
By The JAX authors 

© Copyright 2024, The JAX Authors. 



'''
