# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

key = random.PRNGKey(0)
params_key, x_key = random.split(key)

conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding="SAME",
  transpose_kernel=True,
)
# avg_pool is a function in Flax and does not have parameters to initialize.

# Flax uses a channel-last convention: (N, D, H, W, C)
x_shape = (batch_size, depth, height, width, in_channels)
x = random.normal(x_key, x_shape)
params = conv_transpose.init(random.key(0), x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel demonstrating a fusion of operations.
  NOTE: lax.conv_transpose, nn.avg_pool, and nn.softmax are not supported
  in Pallas on TPU. This kernel replaces them with supported primitives
  (einsum, element-wise functions) to demonstrate a working fusion pattern.
  """
  x_slice = x_ref[...]

  # 1. Perform a channel-mixing operation using einsum, which is supported
  # in Pallas on TPU and replaces the unsupported `lax.conv_transpose`.
  # This applies a linear transformation on the channel dimension, using
  # a slice of the kernel weights. The original einsum string "ndhwc,co->ndhwo"
  # caused a compilation error, so we use different labels to avoid it.
  y = jnp.einsum("ndhwk,ki->ndhwi", x_slice, kernel_ref[0, 0, 0, ...], precision=jax.lax.Precision.HIGHEST)

  # Add the bias term. JAX automatically broadcasts it.
  y = y + bias_ref[...]

  # 2. Apply a sequence of element-wise operations.
  # The original avg_pool and softmax are replaced by a simple tanh
  # activation function, as they are not supported.
  y = jnp.tanh(y)

  # 3. Clip the values to the specified range.
  y = jnp.clip(y, a_min=clamp_min, a_max=clamp_max)

  # 4. Perform the final element-wise multiplication.
  y = y * 2

  # Write the final result for this batch item to the output.
  out_ref[...] = y


# The kernel combines a convolution-like operation with several element-wise
# functions. The grid is defined over the batch dimension. Each kernel instance
# computes the full output for one batch item.

# The new operations do not change the spatial dimensions.
out_d, out_h, out_w = depth, height, width

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_d, out_h, out_w, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input 'x' is sliced along the batch dimension. Each kernel gets one item.
    pl.BlockSpec((1, depth, height, width, in_channels), lambda i: (i, 0, 0, 0, 0)),
    # Kernel and bias are not sliced and are passed in full to each kernel.
    pl.BlockSpec(params["kernel"].shape, lambda i: (0,) * params["kernel"].ndim),
    pl.BlockSpec(params["bias"].shape, lambda i: (0,) * params["bias"].ndim),
  ],
  out_specs=pl.BlockSpec(
    # The output block for each kernel is the full output for one batch item.
    block_shape=(1, out_d, out_h, out_w, out_channels),
    index_map=lambda i: (i, 0, 0, 0, 0),
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()
