# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
stride = 2

key = random.PRNGKey(0)
key, params_key, bias_key, x_key = random.split(key, 4)

# JAX uses NHWC data format, so the input shape is (N, H, W, C)
x_shape = (batch_size, height, width, in_channels)
x = random.normal(x_key, x_shape)

# Using 'SAME' padding results in an output shape of (batch_size, 32, 32, 16),
# which is calculated as input_size * stride. This avoids complex padding
# calculations that can be problematic in Pallas kernels.
conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding="SAME"
)
params = conv_transpose.init(params_key, x)["params"]

# Bias shape is (out_channels,) for broadcasting over the NHWC output
bias = random.normal(bias_key, (out_channels,))


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a transposed convolution, bias subtraction, and tanh activation.

  Args:
    x_ref: Input tensor with shape (1, 16, 16, 32).
    kernel_ref: Convolution kernel tensor with shape (4, 4, 32, 16).
    bias_ref: Bias tensor with shape (16,).
    out_ref: Output tensor with shape (1, 32, 32, 16).
  """
  # Perform the transposed convolution.
  # The dimension numbers specify the layout of the input, kernel, and output.
  # 'NHWC' for input/output: (batch, height, width, channels)
  # 'HWIO' for kernel: (height, width, in_channels, out_channels)
  y = jax.lax.conv_transpose(
    x_ref[...], kernel_ref[...], strides=(2, 2), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
  )

  # Subtract the bias. JAX handles the broadcasting of the bias tensor.
  y = y - bias_ref[...]

  # Apply the tanh activation function.
  y = jnp.tanh(y)

  # Write the final result to the output buffer.
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 32, 32, 16), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 16, 16, 32), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(4, 4, 32, 16), index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=(16,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 32, 32, 16), index_map=lambda i: (i, 0, 0, 0)),
)(x, params["kernel"], bias).block_until_ready()
