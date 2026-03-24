# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 256
N = 256
K = 131072
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128
bK = 128


def kernel(a_ref, b_ref, c_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  Args:
    a_ref: A reference to a block of matrix A.
    b_ref: A reference to a block of matrix B.
    c_ref: A reference to a block of the output matrix C.
  """
  acc = jnp.zeros((bM, bN), dtype=A.dtype)

  def body(k, acc_val):
    # Correctly load chunks of A and B from HBM into SRAM.
    # The shapes of a_block and b_block will be (bM, bK) and (bK, bN).
    a_block = pl.load(a_ref, (slice(None), pl.dslice(k * bK, bK)))
    b_block = pl.load(b_ref, (pl.dslice(k * bK, bK), slice(None)))
    return acc_val + jnp.matmul(a_block, b_block)

  # Loop over the K dimension in chunks of bK.
  acc = jax.lax.fori_loop(0, K // bK, body, acc)
  c_ref[...] = acc


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  # The in_specs define the shape of the *entire* slice of the input
  # arrays that each kernel instance can access (a_ref and b_ref).
  # The kernel then loads smaller chunks from these slices.
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
  # The memory_space argument tells Pallas where the Refs live.
  # By default they live in HBM, which is what we want here to avoid
  # the RESOURCE_EXHAUSTED error.
  interpret=True,  # interpret is needed for fori_loop with dynamic slicing
)(A, B).block_until_ready()
