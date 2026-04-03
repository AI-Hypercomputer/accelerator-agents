# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
eps = 1e-5
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for RMS normalization."""
  # Epsilon value for numerical stability, same as in the source.
  eps = 1e-5

  # The kernel computes the mean over the `features` dimension.
  # We can process each (dim1, dim2) location independently.
  # We create a program that computes the RMS norm for a single vector
  # of size `features`.

  # Load a single vector of size `features`
  x = x_ref[...]

  # 1. Square the input values.
  x_sq = x * x

  # 2. Compute the mean of the squared values.
  # The reduction is now over the features dimension, which is axis=1.
  mean_x_sq = jnp.mean(x_sq, axis=1, keepdims=True)

  # 3. Calculate the RMS value.
  rms = jnp.sqrt(mean_x_sq + eps)

  # 4. Normalize the input by dividing by the RMS value.
  output = x / rms

  # 5. Write the result to the output buffer.
  out_ref[...] = output


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # We parallelize over all dimensions except the one we are reducing.
  grid=(batch_size, dim1, dim2),
  in_specs=[
    # Each program is responsible for a single feature vector.
    pl.BlockSpec(index_map=lambda i, j, k: (i, 0, j, k), block_shape=(1, features, 1, 1))
  ],
  out_specs=pl.BlockSpec(index_map=lambda i, j, k: (i, 0, j, k), block_shape=(1, features, 1, 1)),
  input_output_aliases={0: 0},
)(x).block_until_ready()
