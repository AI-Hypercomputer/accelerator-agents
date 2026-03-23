# Imports
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = (4, 4)
stride = (2, 2)
# In PyTorch ConvTranspose2d, with kernel_size=4, stride=2, padding=1,
# the output spatial size is doubled (16 -> 32).
# In Flax, padding='SAME' achieves this (H_out = H_in * stride).
conv_transpose_padding = "SAME"
maxpool_kernel_size = (2, 2)
maxpool_stride = (2, 2)
# PyTorch's MaxPool2d default padding is 0, which corresponds to 'VALID' in Flax.
maxpool_padding = "VALID"
hardtanh_min = -1
hardtanh_max = 1

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=conv_transpose_padding
)
variables = conv_transpose.init(key_params, x)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  This kernel processes a single batch item and applies the following sequence
  of operations:
  1. 2D Transposed Convolution
  2. Max Pooling
  3. Clipping (HardTanh)
  4. Mean reduction over spatial dimensions
  5. Tanh activation

  Args:
    x_ref: A reference to the input tensor for a single batch item.
           Shape: (1, height, width, in_channels)
    kernel_ref: A reference to the convolution kernel weights.
                Shape: (kernel_h, kernel_w, out_channels, in_channels)
    bias_ref: A reference to the convolution bias.
              Shape: (out_channels,)
    out_ref: A reference to the output tensor where the result is stored.
             Shape: (1, 1, 1, out_channels)
  """
  # Constants from the source code context
  stride_val = (2, 2)
  maxpool_kernel_size_val = (2, 2)
  maxpool_stride_val = (2, 2)
  hardtanh_min_val = -1
  hardtanh_max_val = 1

  # 1. Transposed Convolution
  # The 'dimension_numbers' specify the layout of the tensors.
  # 'NHWC' for input/output and 'HWIO' for the kernel is standard in JAX.
  dn = ("NHWC", "HWIO", "NHWC")
  # Using explicit padding. For stride=2, kernel=4, 'SAME' padding
  # to double the output size corresponds to adding 1 pixel on each side.
  conv_padding = [(1, 1), (1, 1)]
  x = jax.lax.conv_transpose(
    lhs=x_ref[...], rhs=kernel_ref[...], strides=stride_val, padding=conv_padding, dimension_numbers=dn
  )
  # Add the bias
  x += bias_ref[...]

  # 2. Max Pooling
  # jax.lax.reduce_window is the primitive for pooling operations.
  # We use -jnp.inf as the initial value for the max operation.
  # Using explicit padding. 'VALID' corresponds to no padding.
  # The input to reduce_window has 4 dimensions (N, H, W, C).
  pool_padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
  x = jax.lax.reduce_window(
    x,
    -jnp.inf,
    jax.lax.max,
    window_dimensions=(1, *maxpool_kernel_size_val, 1),
    window_strides=(1, *maxpool_stride_val, 1),
    padding=pool_padding,
  )

  # 3. Clipping (HardTanh)
  x = jnp.clip(x, a_min=hardtanh_min_val, a_max=hardtanh_max_val)

  # 4. Mean over spatial dimensions
  # In JAX's NHWC format, spatial dimensions are at axes 1 and 2.
  x = jnp.mean(x, axis=(1, 2), keepdims=True)

  # 5. Tanh activation
  x = jnp.tanh(x)

  # Store the final result in the output buffer.
  out_ref[...] = x


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=variables["params"]["kernel"].shape, index_map=lambda i: ()),
    pl.BlockSpec(block_shape=variables["params"]["bias"].shape, index_map=lambda i: ()),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(x, *jtu.tree_leaves(variables["params"])).block_until_ready()
