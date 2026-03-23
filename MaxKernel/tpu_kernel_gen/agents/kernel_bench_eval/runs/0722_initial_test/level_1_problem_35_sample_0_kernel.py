# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

x = random.normal(key_x, (batch_size, features, dim1, dim2))
gn = nn.GroupNorm(num_groups=num_groups)
params = gn.init(key_init, x)["params"]


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref, *, num_groups: int, epsilon: float = 1e-5):
  """Pallas kernel for a group normalization-like operation.

  This kernel is designed to work with the provided `pallas_call` invocation,
  where each kernel instance processes a slice of the input tensor with shape
  (1, features, 1, dim2). This slice corresponds to `x[b, :, d1, :]`.

  The normalization statistics (mean and variance) are computed for each group
  across the channels within that group and the `dim2` spatial dimension.

  Note: This differs from a standard `nn.GroupNorm` as statistics are not
  computed over the `dim1` spatial dimension, due to the data slicing pattern
  of the invocation.

  Args:
    x_ref: Input tensor block of shape (1, features, 1, dim2).
    scale_ref: Scaling factor (gamma) of shape (features,).
    bias_ref: Bias term (beta) of shape (features,).
    out_ref: Output tensor block, to be written in-place.
    num_groups: The number of groups to split the channel dimension into.
    epsilon: Small float added to variance for numerical stability.
  """
  # Load data from HBM into SRAM.
  # x_slice has shape (1, features, 1, dim2).
  x_slice = x_ref[...]
  # scale and bias have shape (features,).
  scale = scale_ref[...]
  bias = bias_ref[...]

  # Squeeze singleton dimensions for 2D processing. Shape becomes (features, dim2).
  x_2d = jnp.squeeze(x_slice, axis=(0, 2))
  features, dim2 = x_2d.shape
  group_size = features // num_groups

  # Reshape to (num_groups, group_size, dim2) to compute statistics per group.
  x_grouped = x_2d.reshape(num_groups, group_size, dim2)

  # Calculate mean and variance over the group's channels (axis=1) and the
  # second spatial dimension (axis=2).
  mean = jnp.mean(x_grouped, axis=(1, 2), keepdims=True)
  var = jnp.var(x_grouped, axis=(1, 2), keepdims=True)

  # Normalize the data within each group. `mean` and `var` broadcast correctly.
  x_norm_grouped = (x_grouped - mean) / jnp.sqrt(var + epsilon)

  # Reshape back to (features, dim2).
  x_norm_2d = x_norm_grouped.reshape(features, dim2)

  # Apply the learned affine transformation (scale and bias).
  # Reshape scale and bias to (features, 1) to broadcast across the `dim2` dimension.
  result_2d = x_norm_2d * scale[:, None] + bias[:, None]

  # Reshape the final result to the output block's shape and write to memory.
  out_ref[...] = result_2d.reshape(1, features, 1, dim2)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # Grid iterates over each item in the batch and each spatial row.
  # Each kernel instance processes a (features, dim2) slice.
  grid=(x.shape[0], x.shape[2]),
  in_specs=[
    # Input x: Read a (1, features, 1, dim2) block for each kernel instance.
    pl.BlockSpec(block_shape=(1, x.shape[1], 1, x.shape[3]), index_map=lambda b, d1: (b, 0, d1, 0)),
    # Scale (gamma): Read the entire (features,) vector.
    pl.BlockSpec(block_shape=(x.shape[1],), index_map=lambda b, d1: ()),
    # Bias (beta): Read the entire (features,) vector.
    pl.BlockSpec(block_shape=(x.shape[1],), index_map=lambda b, d1: ()),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, x.shape[1], 1, x.shape[3]), index_map=lambda b, d1: (b, 0, d1, 0)),
  num_groups=num_groups,
)(x, params["scale"], params["bias"]).block_until_ready()
