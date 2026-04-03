# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

key = random.PRNGKey(0)
key_x, key_params, key_dropout = random.split(key, 3)
key_conv, key_norm = random.split(key_params)

# JAX uses a channels-last convention: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

# In Flax, layer definitions are stateless. We initialize them to get parameters.
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
conv_params = conv.init(key_conv, x)["params"]

# To initialize GroupNorm, we need a dummy input with the correct channel dimension.
dummy_input_for_norm = jnp.ones((batch_size, depth, height, width, out_channels))
norm = nn.GroupNorm(num_groups=groups, epsilon=1e-5)
norm_params = norm.init(key_norm, dummy_input_for_norm)["params"]

# Perform the convolution operation outside of the Pallas kernel.
conv_out = conv.apply({"params": conv_params}, x)

# Reshape the convolution output to prepare for group normalization.
# This avoids a reshape operation inside the kernel that is not supported on TPU.
reshaped_conv_out = conv_out.reshape(batch_size, depth, height, width, groups, out_channels // groups)


# Computation
def kernel(
  x_ref,
  norm_scale_ref,
  norm_bias_ref,
  dropout_key_ref,
  out_ref,
):
  """
  Pallas kernel that applies group normalization, clipping, and dropout.
  It assumes the input is the result of a preceding convolution and has
  already been reshaped for group normalization.

  This kernel processes a single item from a batch. The batch dimension is
  handled by the grid mapping in the pallas_call.

  Args:
    x_ref: Reference to the input tensor for a single batch item.
      Shape: (1, depth, height, width, groups, out_channels // groups)
    norm_scale_ref: Reference to the GroupNorm scale parameter.
      Shape: (out_channels,)
    norm_bias_ref: Reference to the GroupNorm bias parameter.
      Shape: (out_channels,)
    dropout_key_ref: Reference to the base PRNG key for dropout.
    out_ref: Reference to the output tensor.
  """
  # Hardcoded constants from the source computation
  groups = 8
  min_value = 0.0
  max_value = 1.0
  dropout_p = 0.2
  epsilon = 1e-5
  out_channels = 16

  # The input `x_ref` is already grouped for normalization.
  x = x_ref[...]

  # 1. Apply Group Normalization
  # Calculate mean and variance over spatial and channel-group dimensions.
  # axes=(1, 2, 3, 5) corresponds to (D, H, W, C // num_groups)
  reduction_axes = (1, 2, 3, 5)
  mean = jnp.mean(x, axis=reduction_axes, keepdims=True)
  var = jnp.var(x, axis=reduction_axes, keepdims=True)

  # Normalize.
  norm_out = (x - mean) / jnp.sqrt(var + epsilon)

  # Reshape scale and bias to apply them to the grouped tensor.
  scale_reshaped = norm_scale_ref[...].reshape(1, 1, 1, 1, groups, out_channels // groups)
  bias_reshaped = norm_bias_ref[...].reshape(1, 1, 1, 1, groups, out_channels // groups)
  x = norm_out * scale_reshaped + bias_reshaped

  # 2. Apply minimum operation (to be faithful to the original code)
  x = jnp.minimum(x, min_value)

  # 3. Apply clipping
  x = jnp.clip(x, a_min=min_value, a_max=max_value)

  # 4. Apply dropout
  # Derive a unique dropout key for this specific kernel instance using its
  # program ID. This ensures that each item in the batch gets a different
  # dropout mask.
  key = random.fold_in(dropout_key_ref[...], pl.program_id(0))
  keep_prob = 1.0 - dropout_p
  mask = random.bernoulli(key, p=keep_prob, shape=x.shape)
  x = jnp.where(mask, x / keep_prob, 0.0)

  # 5. Write the final result to the output buffer
  out_ref[...] = x


pallas_output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(reshaped_conv_out.shape, conv_out.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input tensor is the reshaped output of the convolution, chunked along
    # the batch dimension.
    pl.BlockSpec(
      block_shape=(1, depth, height, width, groups, out_channels // groups),
      index_map=lambda i: (i, 0, 0, 0, 0, 0),
    ),
    # The GroupNorm scale is a shared parameter.
    pl.BlockSpec(
      block_shape=norm_params["scale"].shape,
      index_map=lambda i: (0,) * norm_params["scale"].ndim,
    ),
    # The GroupNorm bias is a shared parameter.
    pl.BlockSpec(
      block_shape=norm_params["bias"].shape,
      index_map=lambda i: (0,) * norm_params["bias"].ndim,
    ),
    # The dropout key is passed as a shared resource.
    pl.BlockSpec(
      block_shape=key_dropout.shape,
      index_map=lambda i: (0,) * key_dropout.ndim,
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, depth, height, width, groups, out_channels // groups),
    index_map=lambda i: (i, 0, 0, 0, 0, 0),
  ),
)(
  reshaped_conv_out,
  norm_params["scale"],
  norm_params["bias"],
  key_dropout,
).block_until_ready()

# Reshape the final output back to the standard channel layout.
x = pallas_output.reshape(batch_size, depth, height, width, out_channels)
