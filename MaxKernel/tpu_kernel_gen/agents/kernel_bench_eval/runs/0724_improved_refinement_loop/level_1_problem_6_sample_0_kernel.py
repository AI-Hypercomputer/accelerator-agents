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
# Introduce a block size for the K dimension to fit slices into SRAM.
bK = 128


def kernel(x_ref, y_ref, z_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to the full input matrix A in HBM.
    y_ref: A reference to the full input matrix B in HBM.
    z_ref: A reference to a (bM, bN) block of the output matrix C.
  """
  # Get the program IDs for the M and N dimensions.
  i = pl.program_id(0)
  j = pl.program_id(1)

  # Initialize an accumulator in registers.
  acc = jnp.zeros((bM, bN), dtype=x_ref.dtype)

  # Iterate over the K dimension in blocks of bK.
  for k in range(K // bK):
    # Load a (bM, bK) block of A and a (bK, bN) block of B from HBM
    # into SRAM.
    a_block = pl.load(x_ref, (pl.dslice(i * bM, bM), pl.dslice(k * bK, bK)))
    b_block = pl.load(y_ref, (pl.dslice(k * bK, bK), pl.dslice(j * bN, bN)))
    # Perform matrix multiplication on the blocks and accumulate the result.
    acc += a_block @ b_block
  # Store the final accumulated result.
  z_ref[...] = acc


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(),
    pl.BlockSpec(),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()
