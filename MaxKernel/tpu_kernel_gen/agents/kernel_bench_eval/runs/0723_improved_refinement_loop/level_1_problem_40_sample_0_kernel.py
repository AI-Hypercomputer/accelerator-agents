# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
# In Flax, LayerNorm normalizes over specified axes. For an input of
# (batch, features, dim1, dim2), PyTorch's normalized_shape=(features, dim1, dim2)
# is equivalent to normalizing over axes (1, 2, 3).
feature_axes = (1, 2, 3)

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, features, dim1, dim2))
# The `feature_axes` argument is deprecated in favor of `reduction_axes`.
# Using the old name could lead to incorrect parameter shapes if the JAX
# version has removed the alias.
ln = nn.LayerNorm(reduction_axes=feature_axes)
params = ln.init(key_params, x)["params"]


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """Pallas kernel for Layer Normalization.

  This kernel normalizes a single slice of the input tensor `x` from a batch.
  The normalization is performed across the feature dimensions in a tiled manner
  to fit within TPU VMEM constraints.
  """
  # Flax's LayerNorm uses a default epsilon of 1e-6 for numerical stability.
  epsilon = 1e-6
  num_elements = features * dim1 * dim2
  # A tile for this computation will be a single feature plane.
  # Its size is 256 * 256 * 4 bytes = 256KB, which fits comfortably in VMEM.

  # Pass 1: Compute sum and sum of squares for variance calculation.
  def pass1_body(f, carry):
    sum_val, sum_sq_val = carry
    # Load one feature plane from the input tensor.
    x_tile = pl.load(x_ref, (0, f, 0, 0), block_shape=(1, 1, dim1, dim2))
    sum_val += jnp.sum(x_tile)
    sum_sq_val += jnp.sum(jnp.square(x_tile))
    return sum_val, sum_sq_val

  # Iterate over the feature dimension to calculate statistics for the whole layer.
  sum_val, sum_sq_val = jax.lax.fori_loop(0, features, pass1_body, (0.0, 0.0))

  mean = sum_val / num_elements
  var = sum_sq_val / num_elements - jnp.square(mean)
  inv_stddev = jax.lax.rsqrt(var + epsilon)

  # Pass 2: Apply normalization and store the result.
  def pass2_body(f, _):
    # Load tiles of input, scale, and bias for the current feature.
    x_tile = pl.load(x_ref, (0, f, 0, 0), block_shape=(1, 1, dim1, dim2))
    scale_tile = pl.load(scale_ref, (f, 0, 0), block_shape=(1, dim1, dim2))
    bias_tile = pl.load(bias_ref, (f, 0, 0), block_shape=(1, dim1, dim2))

    normalized_x = (x_tile - mean) * inv_stddev
    result_tile = normalized_x * scale_tile + bias_tile

    # Store the resulting tile in the output buffer.
    pl.store(out_ref, (0, f, 0, 0), result_tile)

  # Iterate over the feature dimension to apply the normalization to all tiles.
  jax.lax.fori_loop(0, features, pass2_body, None)


# The LayerNorm operation is parallelized across the batch dimension.
# Each kernel instance is responsible for normalizing one element of the batch.
# The BlockSpecs now define the shape of the Refs passed to the kernel. The kernel
# then uses pl.load/pl.store to manage memory I/O in a tiled fashion.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    # For input 'x', the Ref passed to the kernel will be a slice of shape
    # (1, features, dim1, dim2).
    pl.BlockSpec(block_shape=(1, features, dim1, dim2), index_map=lambda i: (i, 0, 0, 0)),
    # The 'scale' and 'bias' Refs point to the full parameter tensors.
    # The index_map should return an empty tuple to indicate that the entire
    # tensor is passed to each kernel instance without slicing.
    pl.BlockSpec(block_shape=(features, dim1, dim2), index_map=lambda _: ()),
    pl.BlockSpec(block_shape=(features, dim1, dim2), index_map=lambda _: ()),
  ],
  # The output Ref has the same shape as the input 'x' slice.
  out_specs=pl.BlockSpec(block_shape=(1, features, dim1, dim2), index_map=lambda i: (i, 0, 0, 0)),
)(x, params["scale"], params["bias"]).block_until_ready()
