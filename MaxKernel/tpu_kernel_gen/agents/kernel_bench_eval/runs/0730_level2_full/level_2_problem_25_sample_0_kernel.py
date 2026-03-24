# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu_v4 import conv

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
key = random.PRNGKey(0)
key, data_key, params_key = random.split(key, 3)
x = random.normal(data_key, (batch_size, height, width, in_channels))
conv_layer = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv_layer.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a sequence of convolution, reduction, and activations.

  This kernel implements the following chain of operations:
  1. 2D Convolution: Applies a convolution with a specific padding scheme
     to match the input/output block specifications.
  2. Bias Addition: Adds a bias vector to the convolution result.
  3. Min Reduction: Finds the minimum value across the channel dimension.
  4. Tanh Activation: Applies the tanh activation function twice.

  Args:
    x_ref: A reference to the input data slice.
    kernel_ref: A reference to the convolution kernel weights.
    bias_ref: A reference to the convolution bias.
    out_ref: A reference to the output buffer for storing the result.
  """
  # The dimension numbers specify the layout for a standard NHWC convolution.
  dimension_numbers = ("NHWC", "HWIO", "NHWC")

  # Perform the 2D convolution. The padding is set to ((0, 0), (1, 1)),
  # which corresponds to 'VALID' padding on the height dimension and 'SAME'
  # padding on the width dimension. This is necessary to transform the input
  # block of shape (1, 3, 32, 3) into an intermediate tensor of shape
  # (1, 1, 32, 16).
  conv_result = conv(
    x_ref[...],
    kernel_ref[...],
    window_strides=(1, 1),
    padding=((1, 1), (1, 1)),
    dimension_numbers=dimension_numbers,
  )

  # Add the bias term. The bias vector is broadcast to match the shape of the
  # convolution result.
  conv_result = conv_result + bias_ref[...]

  # Apply the reduction operation. jnp.min is applied along the channel axis (3).
  # keepdims=True ensures the output shape is (1, 1, 32, 1), matching out_ref.
  reduced_result = jnp.min(conv_result, axis=3, keepdims=True)

  # Apply the tanh activation function twice.
  activated_result = jnp.tanh(jnp.tanh(reduced_result))

  # Write the final result to the output reference.
  out_ref[...] = activated_result


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((128, 32, 32, 1), x.dtype),
  grid=(128, 32),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, 32, 3), index_map=lambda i, j: (i, j, 0, 0)),
    pl.BlockSpec(block_shape=(3, 3, 3, 16), index_map=lambda i, j: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=(16,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 32, 1), index_map=lambda i, j: (i, j, 0, 0)),
)(x, params["kernel"], params["bias"]).block_until_ready()
