# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 4096
key = random.PRNGKey(0)
A = random.normal(key, (M, N))
# The scalar must be a JAX type for the `in_specs=None` to work correctly.
s = jnp.float32(3.14)
BLOCK_M = 128
BLOCK_N = 128
grid = (M // BLOCK_M, N // BLOCK_N)


# Computation
def kernel(a_ref, s, c_ref):
  """Pallas kernel for element-wise scalar multiplication.

  Args:
    a_ref: A reference to a block of the input matrix A.
    s: The scalar value to multiply by. Note this is passed by value.
    c_ref: A reference to the corresponding block of the output matrix C.
  """
  # Perform element-wise multiplication of the input block by the scalar `s`
  # and write the result to the output block.
  # a_ref[...] loads the input block from SRAM.
  # The result of the multiplication is written directly to c_ref in SRAM.
  c_ref[...] = a_ref[...] * s


# The pallas_call replaces the C = A * s computation.
C = pl.pallas_call(
  kernel,
  # The output C has the same shape and dtype as the input A.
  out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
  grid=grid,
  # in_specs describes how to chunk the inputs. The scalar `s` is passed
  # to all kernel instances and does not need a BlockSpec. We use None to
  # indicate it's passed as-is.
  in_specs=[
    # Each kernel instance (i, j) gets a (BLOCK_M, BLOCK_N) block of A.
    pl.BlockSpec(block_shape=(BLOCK_M, BLOCK_N), index_map=lambda i, j: (i, j)),
    None,
  ],
  # out_specs describes how the output is chunked. It mirrors the input chunking.
  out_specs=pl.BlockSpec(block_shape=(BLOCK_M, BLOCK_N), index_map=lambda i, j: (i, j)),
)(A, s).block_until_ready()
