# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 256
K = 131072
bM = 8
bK = 1024  # Block size for the K dimension
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, 1))


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Computes a block of the matrix-vector product with an inner loop."""
  # Each program computes one block of the output vector.
  # The index of the program determines which block it is.
  i = pl.program_id(axis=0)

  # Initialize accumulator
  acc = jnp.zeros((bM, 1), dtype=A.dtype)

  # Loop over the K dimension in chunks of bK
  for k in range(K // bK):
    # Load a (bM, bK) tile of A and a (bK, 1) tile of B
    a_tile = pl.load(x_ref, (i * bM, k * bK), block_shape=(bM, bK))
    b_tile = pl.load(y_ref, (k * bK, 0), block_shape=(bK, 1))
    # Perform the matmul on the smaller tiles and accumulate
    acc += jnp.matmul(a_tile, b_tile)
  # Store the final result block
  pl.store(out_ref, (i * bM, 0), acc, block_shape=(bM, 1))


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, 1), A.dtype),
  grid=(M // bM,),
  # By removing in_specs and out_specs, we pass the full arrays by reference.
  # The kernel now handles loading tiles from HBM into VMEM manually.
)(A, B).block_until_ready()
