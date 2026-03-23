# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import ConvTranspose
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2
bias = False

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# JAX/Flax expect channels-last format: (N, D, H, W, C)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))

conv_transpose3d = ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation, dilation),
  use_bias=bias,
)
params = conv_transpose3d.init(params_key, x)["params"]

# Calculate output shape based on ConvTranspose parameters
output_depth = (depth - 1) * stride + dilation * (kernel_size - 1) - 2 * padding + 1
output_height = (height - 1) * stride + dilation * (kernel_size - 1) - 2 * padding + 1
output_width = (width - 1) * stride + dilation * (kernel_size - 1) - 2 * padding + 1


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel parallelizes the computation over the batch, depth, and height
  dimensions of the input tensor. Each kernel instance processes one row
  (along the width dimension) of the input, iterating through its elements
  and "painting" the scaled kernel onto the output map at strided,
  dilated, and padded locations. Atomic additions are used to safely
  accumulate results from multiple instances into the output tensor.
  """
  b_idx, d_idx, h_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  strides = (2, 2, 2)
  dilations = (2, 2, 2)
  paddings = (1, 1, 1)

  in_width = x_ref.shape[3]
  kd, kh, kw = kernel_ref.shape[0:3]
  out_d_dim, out_h_dim, out_w_dim = out_ref.shape[1:4]

  # Initialize the output block to zeros before accumulating.
  out_ref[...] = jnp.zeros_like(out_ref)

  for w_idx in range(in_width):
    x_slice = x_ref[0, 0, 0, w_idx, :]

    for kd_i in range(kd):
      for kh_i in range(kh):
        for kw_i in range(kw):
          out_d = d_idx * strides[0] + kd_i * dilations[0] - paddings[0]
          out_h = h_idx * strides[1] + kh_i * dilations[1] - paddings[1]
          out_w = w_idx * strides[2] + kw_i * dilations[2] - paddings[2]

          is_in_bounds = (
            (out_d >= 0) & (out_d < out_d_dim) & (out_h >= 0) & (out_h < out_h_dim) & (out_w >= 0) & (out_w < out_w_dim)
          )

          @pl.when(is_in_bounds)
          def true_branch():
            kernel_slice = kernel_ref[kd_i, kh_i, kw_i, :, :]
            update_vec = jnp.dot(x_slice, kernel_slice)
            # Use atomic add for safe updates from multiple program instances
            pl.atomic_add(out_ref, (0, out_d, out_h, out_w, 0), update_vec)

          true_branch()


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_depth, output_height, output_width, out_channels), x.dtype),
  grid=(batch_size, depth, height),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, 1, width, in_channels), index_map=lambda b, d, h: (b, d, h, 0, 0)),
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, kernel_size, in_channels, out_channels),
      index_map=lambda b, d, h: (0, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, output_depth, output_height, output_width, out_channels),
    index_map=lambda b, d, h: (b, 0, 0, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()
