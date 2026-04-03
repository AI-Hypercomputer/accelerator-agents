# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
# Note: JAX uses channels-last (NHWC) convention
input_shape = (batch_size, height, width, in_channels)
bias_shape = (1, 1, 1, out_channels)

key = random.PRNGKey(0)
key_params, key_bias, key_x = random.split(key, 3)

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, jnp.ones(input_shape))["params"]
bias = random.normal(key_bias, bias_shape)
x = random.normal(key_x, input_shape)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a fused Conv-ReLU-Bias operation.

  Args:
    x_ref: A reference to the input feature map slice.
    kernel_ref: A reference to the convolution kernel weights.
    bias_ref: A reference to the bias terms.
    out_ref: A reference to the output buffer to store the result.
  """
  # Perform the 2D convolution.
  # The input `x_ref` is a slice of the input feature map that represents the
  # receptive field for the output row. The padding is 'VALID' because the
  # slicing in the `pallas_call` invocation already handles the sliding window,
  # so no additional padding is needed within the kernel itself.
  conv_out = pltpu.conv(
    x_ref[...],
    kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # Apply the ReLU activation function element-wise.
  relu_out = jnp.maximum(conv_out, 0)

  # Add the bias term. The bias_ref is broadcast to match the shape.
  final_out = relu_out + bias_ref[...]

  # Write the final result to the output buffer.
  out_ref[...] = final_out


# Add padding to handle convolution boundaries correctly.
pad_size = kernel_size // 2
x_padded = jnp.pad(
  x,
  pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
  mode="constant",
)

width_padded = width + 2 * pad_size

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size, height),
  in_specs=[
    # Input feature map 'x'. Each kernel instance processes a slice from the
    # padded input. The slice is wide enough to compute a full output row.
    pl.BlockSpec(
      (1, kernel_size, width_padded, in_channels),
      lambda b, h: (b, h, 0, 0),
    ),
    # Convolution kernel weights. The full kernel is passed to each instance.
    pl.BlockSpec(
      (kernel_size, kernel_size, in_channels, out_channels),
      lambda b, h: (0, 0, 0, 0),
    ),
    # Bias. The full bias array is passed to each instance.
    pl.BlockSpec((1, 1, 1, out_channels), lambda b, h: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec((1, 1, width, out_channels), lambda b, h: (b, h, 0, 0)),
)(x_padded, params["kernel"], bias).block_until_ready()
