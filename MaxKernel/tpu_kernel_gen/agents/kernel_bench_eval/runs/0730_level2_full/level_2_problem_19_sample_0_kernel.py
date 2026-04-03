# Imports
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
num_groups = 8

key = random.PRNGKey(0)
key, x_key, init_key = random.split(key, 3)

# Note: JAX uses channels-last convention by default
x = random.normal(x_key, (batch_size, height, width, in_channels))


class Model(nn.Module):
  def setup(self):
    self.conv_transpose = nn.ConvTranspose(
      features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride)
    )
    self.group_norm = nn.GroupNorm(num_groups=num_groups)

  def __call__(self, x):
    x = self.conv_transpose(x)
    x = nn.gelu(x)
    x = self.group_norm(x)
    return x


model = Model()
params = model.init(init_key, x)["params"]


# Computation
def kernel(x_ref, conv_bias_ref, conv_kernel_ref, gn_bias_ref, gn_scale_ref, out_ref):
  """
  Pallas kernel that applies a ConvTranspose, GELU, and GroupNorm sequence.

  This kernel processes a single item from a batch. The logic is a direct
  translation of the equivalent Flax model's apply method.

  Args:
    x_ref: Reference to the input data for a single batch item.
    conv_bias_ref: Reference to the ConvTranspose bias.
    conv_kernel_ref: Reference to the ConvTranspose kernel weights.
    gn_bias_ref: Reference to the GroupNorm bias (beta).
    gn_scale_ref: Reference to the GroupNorm scale (gamma).
    out_ref: Reference to the output buffer for the single batch item.
  """
  # Hardcoded constants from the model definition
  stride = 2
  num_groups = 8
  epsilon = 1e-5

  # Step 1: Transposed Convolution
  # The Flax kernel is stored in (KH, KW, C_in, C_out) format.
  # For lax.conv_transpose, the kernel layout is specified by dimension numbers.
  # We use 'HWOI', where for a transposed convolution, 'O' corresponds to
  # input features and 'I' to output features. This matches the Flax kernel
  # layout, so no transpose is needed.
  dn = jax.lax.ConvDimensionNumbers(lhs_spec=("NHWC"), rhs_spec=("HWOI"), out_spec=("NHWC"))
  conv_out = jax.lax.conv_transpose(
    x_ref[...], conv_kernel_ref[...], strides=(stride, stride), padding="SAME", dimension_numbers=dn
  )
  # Add the bias.
  conv_out = conv_out + conv_bias_ref[...]

  # Step 2: GELU Activation
  gelu_out = nn.gelu(conv_out)

  # Step 3: Group Normalization
  # The input to GroupNorm has shape (1, H', W', C_out).
  # We squeeze the batch dimension for easier processing within the kernel.
  gelu_out_squeezed = jnp.squeeze(gelu_out, axis=0)

  # Get the shape of the feature map after convolution.
  output_shape = gelu_out_squeezed.shape
  out_channels = output_shape[-1]

  # Reshape for group normalization: (H', W', num_groups, channels_per_group)
  reshaped_for_norm = jnp.reshape(gelu_out_squeezed, (*output_shape[:-1], num_groups, out_channels // num_groups))

  # Calculate mean and variance across the spatial and channels-per-group dimensions.
  # The axes for reduction are (0, 1, 3) corresponding to (H', W', channels_per_group).
  group_mean = jnp.mean(reshaped_for_norm, axis=(0, 1, 3), keepdims=True)
  group_var = jnp.var(reshaped_for_norm, axis=(0, 1, 3), keepdims=True)

  # Normalize the reshaped data.
  normalized = (reshaped_for_norm - group_mean) / jnp.sqrt(group_var + epsilon)

  # Reshape back to the original feature map shape (H', W', C_out).
  normalized = jnp.reshape(normalized, output_shape)

  # Apply the learned scale and bias. These have shape (C_out,) and will
  # broadcast correctly over the spatial dimensions.
  group_norm_out = normalized * gn_scale_ref[...] + gn_bias_ref[...]

  # Add the batch dimension back and write the final result to the output buffer.
  out_ref[...] = jnp.expand_dims(group_norm_out, axis=0)


# Determine the output shape and flatten the parameters for the kernel call
out_struct = jax.eval_shape(model.apply, {"params": params}, x)
flat_params, _ = jtu.tree_flatten(params)

# The kernel is parallelized over the batch dimension. Each kernel instance
# processes one item from the batch.
# - grid: A 1D grid of size `batch_size`. The grid index `i` corresponds to the i-th batch item.
# - in_specs:
#   - For the input `x`, `BlockSpec` selects the i-th slice along the batch dimension.
#   - For the model parameters, each kernel instance needs the full parameter arrays.
#     `BlockSpec` is configured with the full parameter shape and an index_map
#     that ignores the grid index, providing the full array to each instance.
# - out_specs: `BlockSpec` maps the output of the i-th kernel instance to the i-th slice of the full output array.
x = pl.pallas_call(
  kernel,
  out_shape=out_struct,
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    *[pl.BlockSpec(p.shape, lambda i, ndim=p.ndim: (0,) * ndim) for p in flat_params],
  ],
  out_specs=pl.BlockSpec(block_shape=(1, *out_struct.shape[1:]), index_map=lambda i: (i, 0, 0, 0)),
)(x, *flat_params).block_until_ready()
