# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def normalize_l2(x: jax.Array) -> jax.Array:
  """Computes L2 normalization for each row of a matrix using a Pallas kernel."""
  # Create an array to store the sum of squares for each row, initialized to 0.
  sum_sq_shape = jax.ShapeDtypeStruct((x.shape[0],), x.dtype)
  sum_sq = jnp.zeros(x.shape[0], dtype=x.dtype)

  def sum_sq_kernel(x_ref, sum_sq_ref):
    # This kernel computes the sum of squares for one tile of the input
    # and atomically adds the result to the appropriate row's total.
    # x_ref is a tile of shape (8, 128).
    # sum_sq_ref is a corresponding block of 8 elements from the sum_sq array.

    # Calculate sum of squares for each row within the tile.
    partial_sum_sq = jnp.sum(x_ref[...] * x_ref[...], axis=1)

    # Atomically add each partial sum to the main sum_sq array.
    for i in range(x_ref.shape[0]):
      pl.atomic_add(sum_sq_ref, i, partial_sum_sq[i])

  # Use pallas_call to execute the sum_sq_kernel over the entire input array.
  grid_rows = x.shape[0] // 8
  grid_cols = x.shape[1] // 128
  sum_sq = pl.pallas_call(
    sum_sq_kernel,
    out_shape=sum_sq_shape,
    # The grid iterates over blocks of rows and tiles along the 'dim' axis.
    grid=(grid_rows, grid_cols),
    in_specs=[
      # For grid index (i, j), map to the input tile at row i*8, col j*128.
      pl.BlockSpec((8, 128), lambda i, j: (i * 8, j * 128))
    ],
    out_specs=[
      # Map to the block of 8 rows in the sum_sq array. All tiles `j` for a
      # given row block `i` will update the same output block.
      pl.BlockSpec((8,), lambda i, j: (i * 8,))
    ],
  )(x, sum_sq)

  # The rest of the normalization can be done efficiently in plain JAX.
  # Calculate the L2 norm from the sum of squares.
  norm = jnp.sqrt(sum_sq)

  # Add a new axis to allow broadcasting, e.g., (16,) -> (16, 1).
  norm = norm[:, None]

  # Avoid division by zero by replacing zero norms with 1.
  safe_norm = jnp.where(norm == 0, 1.0, norm)

  return x / safe_norm


# Computation
result = normalize_l2(x).block_until_ready()
