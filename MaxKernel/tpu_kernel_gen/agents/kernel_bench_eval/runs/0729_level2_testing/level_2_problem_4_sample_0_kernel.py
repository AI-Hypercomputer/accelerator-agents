# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
x_key, params_key = random.split(key)

# JAX uses (N, H, W, C) convention, so we create the input accordingly.
x = random.normal(x_key, (batch_size, height, width, in_channels))

# PyTorch's default padding is 0, which corresponds to 'VALID' in JAX.
# We use the default NHWC layout, so no dimension_numbers are needed.
conv = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  padding="VALID",
)
variables = conv.init(params_key, x)

# Calculate output spatial dimensions for 'VALID' padding
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of convolution, bias addition, and two Mish activations.

  This kernel processes a single image from a batch. The batching is handled
  by the grid specification in the pallas_call.

  Args:
    x_ref: A reference to the input image tensor with shape
      (1, height, width, in_channels).
    kernel_ref: A reference to the convolution kernel weights with shape
      (kernel_size, kernel_size, in_channels, out_channels).
    bias_ref: A reference to the bias tensor with shape (out_channels,).
    out_ref: A reference to the output tensor to store the result, with shape
      (1, out_height, out_width, out_channels).
  """

  # Define the Mish activation function.
  def mish(x):
    return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))

  # Perform the 2D convolution using 'VALID' padding.
  # The dimension numbers specify the layout of the input ('NHWC'),
  # the kernel ('HWIO'), and the output ('NHWC').
  conv_out = lax.conv_general_dilated(
    lhs=x_ref[...],
    rhs=kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # Add the bias to the convolution output. Broadcasting applies the
  # bias vector across all spatial dimensions of the output.
  with_bias = conv_out + bias_ref[...]

  # Apply the Mish activation function twice in sequence.
  activated_out = mish(with_bias)
  final_out = mish(activated_out)

  # Write the final result to the output buffer.
  out_ref[...] = final_out


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input images: Each kernel instance gets one full image.
    pl.BlockSpec(
      block_shape=(1, height, width, in_channels),
      index_map=lambda n: (n, 0, 0, 0),
    ),
    # Kernel weights: Broadcast the same weights to all instances.
    pl.BlockSpec(
      block_shape=variables["params"]["kernel"].shape,
      index_map=lambda n: (0, 0, 0, 0),
    ),
    # Bias: Broadcast the same bias to all instances.
    pl.BlockSpec(block_shape=variables["params"]["bias"].shape, index_map=lambda n: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_height, out_width, out_channels),
    index_map=lambda n: (n, 0, 0, 0),
  ),
)(x, variables["params"]["kernel"], variables["params"]["bias"])
result.block_until_ready()
