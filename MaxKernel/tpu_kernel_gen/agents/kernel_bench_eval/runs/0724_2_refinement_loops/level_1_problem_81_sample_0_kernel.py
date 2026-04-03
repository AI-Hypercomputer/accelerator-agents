# Imports
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2
bias = False

key = random.PRNGKey(0)
x_key, params_key = random.split(key)

# JAX uses channels-last convention by default: (N, H, W, C)
x = random.normal(x_key, (batch_size, height_in, width_in, in_channels))

# In Flax, layers are defined, then initialized with a PRNG key and dummy data
conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation),
  use_bias=bias,
)
params = conv_transpose2d.init(params_key, x)["params"]


def kernel(x_ref, kernel_ref, y_ref):
  """Pallas kernel for 2D transposed convolution.

  This kernel computes a tile of the output tensor `y`. Each program instance
  is responsible for a specific tile, determined by its program IDs. The logic
  iterates through each pixel of the output tile and "gathers" contributions
  from the input image (`x_ref`) based on the convolution kernel (`kernel_ref`).
  This version uses `lax.fori_loop` to avoid unrolling large loops, which
  prevents timeouts on the TPU.

  Args:
    x_ref: A reference to the input tensor tile. In this invocation, it's the
      full (1, H, W, C_in) image for a given batch item.
    kernel_ref: A reference to the convolution kernel weights. In this
      invocation, it's the full (KH, KW, C_in, C_out) kernel.
    y_ref: A reference to the output tensor tile to be computed, with shape
      (1, bH, bW, C_out).
  """
  # Hyperparameters from the original Flax layer definition.
  kernel_height, kernel_width, _, _ = kernel_ref.shape
  height_in, width_in, _ = x_ref.shape[1:]
  stride = 5
  padding = 1
  dilation = 2

  # Get the tile indices from the program IDs.
  j = pl.program_id(1)
  k = pl.program_id(2)

  # Get the output tile's dimensions from its shape.
  _, bH, bW, C_out = y_ref.shape

  # Calculate the start coordinates of the output tile this program works on.
  h_start = j * bH
  w_start = k * bW

  # Define the body of the loop over the height of the output tile.
  def h_loop_body(h_out_local, _):
    h_out = h_start + h_out_local

    # Define the body of the loop over the width of the output tile.
    def w_loop_body(w_out_local, __):
      w_out = w_start + w_out_local
      acc = jnp.zeros((C_out,), dtype=y_ref.dtype)

      # Iterate over the kernel (these loops are small and will be unrolled).
      for kh in range(kernel_height):
        for kw in range(kernel_width):
          numerator_h = h_out - kh * dilation + padding
          numerator_w = w_out - kw * dilation + padding
          h_in = numerator_h // stride
          w_in = numerator_w // stride

          # Check if the contribution is valid.
          is_valid = (
            (numerator_h % stride == 0)
            & (numerator_w % stride == 0)
            & (h_in >= 0)
            & (h_in < height_in)
            & (w_in >= 0)
            & (w_in < width_in)
          )

          # Conditionally add the contribution.
          def _add_contribution(current_acc):
            # Ensure matmul operands are 2D to avoid a TPU lowering bug.
            lhs = jnp.expand_dims(x_ref[0, h_in, w_in, :], 0)
            rhs = kernel_ref[kh, kw, :, :]
            update = jnp.squeeze(lhs @ rhs, 0)
            return current_acc + update

          acc = lax.cond(is_valid, _add_contribution, lambda x: x, acc)

      # Write the final accumulated value to the output tile.
      y_ref[0, h_out_local, w_out_local, :] = acc
      return None

    # Use lax.fori_loop for the inner loop to avoid unrolling.
    lax.fori_loop(0, bW, w_loop_body, None)
    return None

  # Use lax.fori_loop for the outer loop to avoid unrolling.
  lax.fori_loop(0, bH, h_loop_body, None)


# Computation
# Determine the output shape and dtype from the original operation for robustness.
y_shape_struct = jax.eval_shape(lambda p, i: conv_transpose2d.apply({"params": p}, i), params, x)

# Define block sizes for tiling the output's spatial dimensions.
# These are chosen to be compatible with TPU hardware constraints.
bH = 16
bW = 128

# Define the execution grid. We parallelize over the batch dimension and
# the tiled spatial dimensions of the output.
grid = (
  batch_size,
  math.ceil(y_shape_struct.shape[1] / bH),
  math.ceil(y_shape_struct.shape[2] / bW),
)

y = pl.pallas_call(
  kernel,
  out_shape=y_shape_struct,
  grid=grid,
  in_specs=[
    # For input x: each kernel instance for a given batch item (i) gets the
    # full (H, W, C) image. The block shape is valid as its last two
    # dimensions match the full array's last two dimensions.
    pl.BlockSpec(
      block_shape=(1, height_in, width_in, in_channels),
      index_map=lambda i, j, k: (i, 0, 0, 0),
    ),
    # For kernel weights: each kernel instance gets the full kernel.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda i, j, k: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # For output y: each kernel instance (i, j, k) writes to a unique
    # tile in the output tensor.
    block_shape=(1, bH, bW, out_channels),
    index_map=lambda i, j, k: (i, j, k, 0),
  ),
)(x, params["kernel"]).block_until_ready()
