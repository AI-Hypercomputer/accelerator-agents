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
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

key = random.PRNGKey(0)
key, x_key, params_key, bias_key = random.split(key, 4)

# The kernel expects NCHW input format.
x = random.normal(x_key, (batch_size, in_channels, height, width))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
# Flax Conv expects NHWC input, so we provide a dummy input with that shape for initialization
dummy_x_for_init = jnp.empty((batch_size, height, width, in_channels))
params = conv.init(params_key, dummy_x_for_init)["params"]

bias = random.normal(bias_key, bias_shape)


# Computation
def kernel(x_ref, kernel_weights_ref, conv_bias_ref, add_bias_ref, out_ref, constant_value, scaling_factor):
  """Pallas kernel for a sequence of convolution and element-wise operations.

  This kernel performs the following steps for each output channel:
  1. 2D convolution on the input data using the corresponding filter.
  2. Adds a convolution bias.
  3. Applies an element-wise minimum operation (clipping).
  4. Adds a second, channel-wise bias.
  5. Scales the result by a constant factor.

  Args:
    x_ref: Reference to the input data tile (NCHW format).
    kernel_weights_ref: Reference to the convolution kernel weights (HWIO format).
    conv_bias_ref: Reference to the convolution bias vector.
    add_bias_ref: Reference to the second bias tensor (channel-wise).
    out_ref: Reference to the output data tile to be written to.
    constant_value: Scalar value for the minimum operation.
    scaling_factor: Scalar value for the final multiplication.
  """
  # The grid is (batch_size, out_channels). pl.program_id(1) gives the
  # index for the current output channel this kernel instance is responsible for.
  channel_idx = pl.program_id(1)

  # Load the input data slice. The in_spec ensures each kernel instance
  # gets one full batch item.
  x_in = x_ref[...]

  # --- 1. Convolution ---
  # Original weights are in HWIO format. We need a kernel for a single
  # output channel in OIHW format, which for one channel is (1, I, H, W).

  # Slice the weights for the current output channel.
  # kernel_weights_ref has shape (H, W, I, O).
  # Slicing on axis 3 gives a (H, W, I, 1) tensor.
  kernel_slice_hwio = jax.lax.dynamic_slice_in_dim(kernel_weights_ref[...], channel_idx, 1, axis=3)

  # Transpose the (H, W, I, 1) slice to (1, I, H, W) for OIHW format.
  # (H, W, I, O) -> (O, I, H, W) corresponds to a (3, 2, 0, 1) transpose.
  kernel_filter = jnp.transpose(kernel_slice_hwio, (3, 2, 0, 1))

  # Perform the 2D convolution.
  conv_output = jax.lax.conv_general_dilated(
    lhs=x_in, rhs=kernel_filter, window_strides=(1, 1), padding="SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
  )

  # Add the corresponding convolution bias.
  conv_bias_val = conv_bias_ref[channel_idx]
  # Reshape bias to be broadcastable with the (1, 1, H, W) conv output.
  y = conv_output + conv_bias_val.reshape(1, 1, 1, 1)

  # --- 2. Element-wise Minimum ---
  y = jnp.minimum(y, constant_value)

  # --- 3. Element-wise Addition ---
  # Slice the second bias tensor for the current channel.
  add_bias_slice = jax.lax.dynamic_slice_in_dim(add_bias_ref[...], channel_idx, 1, axis=0)
  # The slice has shape (1, 1, 1) and broadcasts correctly.
  y = y + add_bias_slice

  # --- 4. Element-wise Scaling ---
  y = y * scaling_factor

  # --- 5. Store Result ---
  # Write the final result to the output buffer.
  out_ref[...] = y


# The final output shape is (batch_size, out_channels, height, width).
final_out_shape = (
  batch_size,
  out_channels,
  x.shape[2],  # height
  x.shape[3],  # width
)

# The pallas_call replaces the original computation block.
# It takes the initial `x` and all parameters as input.
x = pl.pallas_call(
  lambda x_ref, kernel_weights_ref, conv_bias_ref, add_bias_ref, out_ref: kernel(
    x_ref, kernel_weights_ref, conv_bias_ref, add_bias_ref, out_ref, constant_value, scaling_factor
  ),
  out_shape=jax.ShapeDtypeStruct(final_out_shape, x.dtype),
  grid=(batch_size, out_channels),
  in_specs=[
    # x: Each kernel instance (i, j) gets the i-th batch item.
    pl.BlockSpec(block_shape=(1, in_channels, height, width), index_map=lambda i, j: (i, 0, 0, 0)),
    # kernel_weights_ref: Pass the entire weights tensor to each kernel.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j: (0, 0, 0, 0)),
    # conv_bias_ref: Pass the entire bias vector to each kernel.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j: (0,)),
    # add_bias_ref: Pass the entire bias tensor to each kernel.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i, j: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    # out: Each kernel instance (i, j) writes to the (i, j)-th output slice.
    block_shape=(1, 1, height, width),
    index_map=lambda i, j: (i, j, 0, 0),
  ),
)(x, params["kernel"], params["bias"], bias).block_until_ready()
