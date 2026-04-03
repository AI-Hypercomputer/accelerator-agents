# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
num_groups = 8

key = random.PRNGKey(0)
key, key_x, key_conv, key_gn = random.split(key, 4)

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
group_norm = nn.GroupNorm(num_groups=num_groups)

# Note: Flax uses channel-last (N, ..., C) convention by default
x = random.normal(key_x, (batch_size, D, H, W, in_channels))

# Initialize parameters for each layer separately
conv_variables = conv.init(key_conv, x)
# To initialize the next layer, we need an input of the correct shape
dummy_conv_output = conv.apply(conv_variables, x)
gn_variables = group_norm.init(key_gn, dummy_conv_output)

# Computation
# --- 1. Convolution Layer (Standard JAX/Flax) ---
# This is executed outside the Pallas kernel as conv is not a supported primitive.
conv_out = conv.apply(conv_variables, x)


def group_norm_kernel(x_ref, gn_scale_ref, gn_bias_ref, out_ref):
  """
  Pallas kernel that performs Group Normalization.
  """
  # Constants for Group Normalization
  epsilon = 1e-5
  num_groups = 8

  x_val = x_ref[...]
  C = x_val.shape[-1]
  channels_per_group = C // num_groups

  # Create an empty array to hold the results, which we fill group by group.
  final_output = jnp.empty_like(x_val)

  # Load the full scale and bias vectors
  scale = gn_scale_ref[...]
  bias = gn_bias_ref[...]

  for g in range(num_groups):
    start_c = g * channels_per_group

    # Slice the input data for the current group
    group_data = jax.lax.dynamic_slice_in_dim(x_val, start_c, slice_size=channels_per_group, axis=4)

    # Calculate mean and variance over all axes of the slice
    group_mean = jnp.mean(group_data)
    group_var = jnp.var(group_data)

    # Normalize the data
    normalized_data = (group_data - group_mean) / jnp.sqrt(group_var + epsilon)

    # Slice the scale and bias for the current group
    scale_slice = jax.lax.dynamic_slice_in_dim(scale, start_c, slice_size=channels_per_group, axis=0)
    bias_slice = jax.lax.dynamic_slice_in_dim(bias, start_c, slice_size=channels_per_group, axis=0)

    # Apply scale and bias. Broadcasting handles the shapes.
    scaled_data = normalized_data * scale_slice + bias_slice

    # Place the processed group into the final output tensor
    final_output = jax.lax.dynamic_update_slice_in_dim(final_output, scaled_data, start_c, axis=4)

  out_ref[...] = final_output


# The Pallas kernel performs the group normalization.
# The final reduction is handled by a separate JAX call.
x_intermediate = pl.pallas_call(
  group_norm_kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, D, H, W, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input tensor 'conv_out', sliced by batch index.
    pl.BlockSpec(block_shape=(1, D, H, W, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    # GroupNorm scale, not sliced.
    pl.BlockSpec(block_shape=gn_variables["params"]["scale"].shape, index_map=lambda i: (0,)),
    # GroupNorm bias, not sliced.
    pl.BlockSpec(block_shape=gn_variables["params"]["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, D, H, W, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
)(
  conv_out,
  gn_variables["params"]["scale"],
  gn_variables["params"]["bias"],
).block_until_ready()

# The final reduction is performed outside the Pallas kernel.
x = x_intermediate.mean(axis=(1, 2, 3, 4))
