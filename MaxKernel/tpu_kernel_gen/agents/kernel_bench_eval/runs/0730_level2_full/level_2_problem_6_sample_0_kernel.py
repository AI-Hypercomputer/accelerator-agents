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
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX uses channels-last convention (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of 3D convolution, softmax, and max pooling.

  This kernel processes a single batch item. The mapping over the entire batch
  is handled by the `grid` argument in the `pallas_call`.

  Args:
    x_ref: A reference to the input tensor for a single batch item, with shape
      (1, depth, height, width, in_channels).
    kernel_ref: A reference to the convolution kernel tensor, with shape
      (kernel_size, kernel_size, kernel_size, in_channels, out_channels).
    bias_ref: A reference to the convolution bias vector, with shape
      (out_channels,).
    out_ref: A reference to the output tensor, which will be populated by this
      kernel. Its shape is (1, depth/4, height/4, width/4, out_channels).
  """
  # The input x_ref already has a leading dimension of 1 due to the
  # block_spec, so it's ready for use in `conv_general_dilated` which
  # expects a batch dimension.
  x = x_ref[...]
  conv_kernel = kernel_ref[...]
  conv_bias = bias_ref[...]

  # Perform 3D convolution using the Pallas-specific TPU convolution primitive.
  # The dimension numbers are specified for channels-last data ('NDHWC').
  # 'SAME' padding is used to maintain the spatial dimensions of the input.
  conv_out_biased = pltpu.convolution(
    x,
    conv_kernel,
    conv_bias,
    window_strides=(1, 1, 1),
    padding="SAME",
    dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
  )

  # Apply the softmax activation function along the channel axis (-1).
  softmax_out = jax.nn.softmax(conv_out_biased, axis=-1)

  # Define the window shape and strides for the max pooling operations.
  # The pooling is 3D, so we specify a window and stride of 2 for the
  # depth, height, and width dimensions.
  pool_window_shape = (1, 2, 2, 2, 1)
  pool_strides = (1, 2, 2, 2, 1)

  # First max pooling operation. This reduces the spatial dimensions by half.
  pool1_out = jax.lax.reduce_window(
    operand=softmax_out,
    init_value=-jnp.inf,
    computation=jnp.maximum,
    window_dimensions=pool_window_shape,
    window_strides=pool_strides,
    padding="VALID",
  )

  # Second max pooling operation. This further reduces the spatial dimensions.
  pool2_out = jax.lax.reduce_window(
    operand=pool1_out,
    init_value=-jnp.inf,
    computation=jnp.maximum,
    window_dimensions=pool_window_shape,
    window_strides=pool_strides,
    padding="VALID",
  )

  # Write the final result to the output reference.
  out_ref[...] = pool2_out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, depth // 4, height // 4, width // 4, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, depth, height, width, in_channels),
      index_map=lambda i: (i, 0, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, depth // 4, height // 4, width // 4, out_channels),
    index_map=lambda i: (i, 0, 0, 0, 0),
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()
