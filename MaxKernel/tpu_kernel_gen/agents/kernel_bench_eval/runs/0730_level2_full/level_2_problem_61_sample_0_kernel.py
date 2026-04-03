# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention
x = random.normal(key_x, (batch_size, D, H, W, in_channels))


class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), use_bias=bias)(x)
    x = nn.relu(x)
    x = nn.GroupNorm(num_groups=groups)(x)
    return x


model = Model()
params = model.init(key_params, x)["params"]

# Determine output shape
# With default stride=1 and padding='VALID', the output spatial dimensions are:
# output_dim = input_dim + kernel_dim - 1
D_out = D + kernel_size - 1
H_out = H + kernel_size - 1
W_out = W + kernel_size - 1


def kernel(x_ref, conv_kernel_ref, gn_scale_ref, gn_bias_ref, out_ref):
  # Implement transposed convolution as a regular convolution.
  # This is done by flipping the kernel and padding the input.
  # 1. Flip the kernel spatially.
  k_flipped = jnp.flip(conv_kernel_ref[...], axis=(0, 1, 2))

  # 2. Pad the input. Padding on each side is kernel_size - 1.
  pad_width = kernel_size - 1
  padding_config = [
    (0, 0),
    (pad_width, pad_width),
    (pad_width, pad_width),
    (pad_width, pad_width),
    (0, 0),
  ]
  x_padded = jnp.pad(x_ref[...], padding_config)

  # 3. Perform standard 'VALID' convolution.
  conv_out = jax.lax.conv(
    lhs=x_padded,
    rhs=k_flipped,
    window_strides=(1, 1, 1),
    padding="VALID",
    dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
  )

  # Apply ReLU activation
  relu_out = jax.nn.relu(conv_out)

  # Perform Group Normalization.
  # Flax's default epsilon for GroupNorm is 1e-5.
  epsilon = 1e-5
  # Reshape to (N, D, H, W, G, C/G) to compute stats over groups.
  # relu_out shape is (1, D_out, H_out, W_out, out_channels)
  reshaped_for_gn = relu_out.reshape(1, D_out, H_out, W_out, groups, out_channels // groups)
  # Calculate mean and variance over spatial and channel-group dimensions.
  mean = jnp.mean(reshaped_for_gn, axis=(1, 2, 3, 5), keepdims=True)
  var = jnp.var(reshaped_for_gn, axis=(1, 2, 3, 5), keepdims=True)

  # Normalize
  normalized_reshaped = (reshaped_for_gn - mean) / jnp.sqrt(var + epsilon)

  # Reshape back to original channel dimension
  normalized_out = normalized_reshaped.reshape(1, D_out, H_out, W_out, out_channels)

  # Apply the learned scale and bias.
  # These will broadcast over the spatial dimensions.
  final_out = normalized_out * gn_scale_ref[...] + gn_bias_ref[...]

  # Write the final result to the output buffer.
  out_ref[...] = final_out


out_shape_struct = jax.ShapeDtypeStruct((batch_size, D_out, H_out, W_out, out_channels), x.dtype)

# Flatten params for kernel invocation
conv_kernel = params["ConvTranspose_0"]["kernel"]
gn_scale = params["GroupNorm_0"]["scale"]
gn_bias = params["GroupNorm_0"]["bias"]

# Computation
x = pl.pallas_call(
  kernel,
  out_shape=out_shape_struct,
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, D, H, W, in_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, kernel_size, in_channels, out_channels),
      index_map=lambda i: (0, 0, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, D_out, H_out, W_out, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
)(x, conv_kernel, gn_scale, gn_bias).block_until_ready()
