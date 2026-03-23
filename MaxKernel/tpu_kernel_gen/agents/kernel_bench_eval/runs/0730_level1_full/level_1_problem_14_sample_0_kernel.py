# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = jnp.triu(random.normal(key_A, (N, N)))
B = jnp.triu(random.normal(key_B, (N, N)))

bN = 128


# Computation
def kernel(A_ref, B_ref, C_ref):
  """Pallas kernel for triangular matrix multiplication C = triu(A @ B)."""
  # Get the block's row and column index from the program IDs.
  i, j = pl.program_id(0), pl.program_id(1)

  # The output matrix is upper-triangular, so blocks in the lower triangle
  # (where row index > column index) are zero. Pallas initializes the output
  # to zero, so we only need to compute the non-zero blocks.
  def compute_block(_):
    # This kernel computes a bN x bN block of the output matrix C.
    # C_ij = sum_k A_ik @ B_kj
    # Because A and B are upper-triangular, this sum is equivalent to
    # sum_{k=i to j} A_ik @ B_kj. We can implement this by looping
    # over all k and relying on the zero blocks to cancel out.
    acc = jnp.zeros((bN, bN), dtype=C_ref.dtype)

    def body(k, current_acc):
      # Load the k-th block of the i-th row of A
      a_block = pl.load(A_ref, (i * bN, k * bN), block_shape=(bN, bN))
      # Load the j-th block of the k-th row of B
      b_block = pl.load(B_ref, (k * bN, j * bN), block_shape=(bN, bN))
      return current_acc + a_block @ b_block

    result = lax.fori_loop(0, N // bN, body, acc)

    # The diagonal blocks of the output C_ii = A_ii @ B_ii are already
    # upper-triangular because A_ii and B_ii are upper-triangular.
    # However, to match the reference computation triu(A @ B) exactly,
    # especially for potentially non-triangular inputs in a general case,
    # we apply jnp.triu to the diagonal blocks.
    output_block = lax.cond(i == j, lambda res: jnp.triu(res), lambda res: res, result)
    C_ref[...] = output_block

  # We use lax.cond to conditionally execute the computation.
  lax.cond(i <= j, compute_block, lambda _: None, None)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(N // bN, N // bN),
  in_specs=[
    pl.BlockSpec(memory_space=pl.TPUMemorySpace.ANY),
    pl.BlockSpec(memory_space=pl.TPUMemorySpace.ANY),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()
