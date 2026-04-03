# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
# output_padding is not a standard argument in flax.linen.ConvTranspose.
# The output shape is determined by other parameters. We remove it.
# The user's environment does not support grouped convolutions via
# `feature_group_count`, so we remove it and the associated `groups` variable.
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX expects channels-last format: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv_transpose3d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias
)
params = conv_transpose3d.init(key_params, x)["params"]
# This is needed to determine the output shape for the pallas_call
output_shape_struct = jax.eval_shape(lambda: conv_transpose3d.apply({"params": params}, x))

# Block sizes for output tensor dimensions
# We parallelize over batch, depth, height, and width dimensions of the output
bD, bH, bW = 16, 32, 32

# The grid is defined by how many blocks we have in each dimension
grid = (
  batch_size,
  (output_shape_struct.shape[1] + bD - 1) // bD,
  (output_shape_struct.shape[2] + bH - 1) // bH,
  (output_shape_struct.shape[3] + bW - 1) // bW,
)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel implements an output-centric 3D transposed convolution. Each
  program computes a block of the output tensor. For each output pixel in its
  block, it iterates through the kernel's spatial dimensions to find the
  corresponding input pixels that contribute to its value.

  The calculation uses `jnp.einsum` to perform a matrix-vector product between
  the kernel weights and the input channels for each contributing pixel.

  Args:
    x_ref: Reference to the input tensor of shape
      (N, D_in, H_in, W_in, C_in).
    kernel_ref: Reference to the kernel tensor of shape
      (KD, KH, KW, C_in, C_out).
    out_ref: Reference to the output tensor slice to be written to. The shape
      is (bD, bH, bW, C_out), representing one block of the output.
  """
  # Derive shapes and constants from kernel_ref and out_ref
  kernel_depth, kernel_height, kernel_width, _, _ = kernel_ref.shape
  out_channels = out_ref.shape[-1]
  in_depth, in_height, in_width = x_ref.shape[1:4]
  out_depth, out_height, out_width = output_shape_struct.shape[1:4]
  bD, bH, bW, _ = out_ref.shape

  # Get the program ID for each dimension of the grid
  n = pl.program_id(0)
  d_idx = pl.program_id(1)
  h_idx = pl.program_id(2)
  w_idx = pl.program_id(3)

  # Calculate the starting coordinates of the output block this program is responsible for
  od_start = d_idx * bD
  oh_start = h_idx * bH
  ow_start = w_idx * bW

  # Accumulator for the entire output block, initialized to zeros
  acc = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)

  # Iterate over each pixel within the assigned output block
  for blk_d in range(bD):
    for blk_h in range(bH):
      for blk_w in range(bW):
        # Calculate the global coordinates of the current output pixel
        od = od_start + blk_d
        oh = oh_start + blk_h
        ow = ow_start + blk_w

        # Skip computation for pixels that are outside the valid output dimensions
        if od >= out_depth or oh >= out_height or ow >= out_width:
          continue

        # Accumulator for the current output pixel across all output channels
        pixel_acc = jnp.zeros((out_channels,), dtype=out_ref.dtype)

        # Iterate over the kernel's spatial dimensions to find contributing inputs
        for kd in range(kernel_depth):
          for kh in range(kernel_height):
            for kw in range(kernel_width):
              # For a transposed convolution, an input at `id` contributes to an
              # output at `od = id * stride - pad + kd`. We invert this to find
              # the source `id` for a given `od`: `id * stride = od + pad - kd`.
              # A valid input exists only if the RHS is divisible by the stride.
              if (
                (od + padding[0] - kd) % stride[0] == 0
                and (oh + padding[1] - kh) % stride[1] == 0
                and (ow + padding[2] - kw) % stride[2] == 0
              ):
                id = (od + padding[0] - kd) // stride[0]
                ih = (oh + padding[1] - kh) // stride[1]
                iw = (ow + padding[2] - kw) // stride[2]

                # Check if the calculated input coordinates are within the bounds
                if 0 <= id < in_depth and 0 <= ih < in_height and 0 <= iw < in_width:
                  # Load the input channels for the valid pixel
                  x_channels = x_ref[n, id, ih, iw, :]
                  # Load the kernel weights for the current spatial tap
                  k_weights = kernel_ref[kd, kh, kw, :, :]

                  # 'i,io->o' contracts the input channels (i) with the kernel
                  # to produce all output channels (o).
                  contribution = jnp.einsum("i,io->o", x_channels, k_weights, preferred_element_type=out_ref.dtype)

                  pixel_acc += contribution

        # Store the final computed pixel value in the block accumulator
        acc = acc.at[blk_d, blk_h, blk_w, :].set(pixel_acc)

  # Write the completed block from the accumulator to the output reference
  out_ref[...] = acc


output = pl.pallas_call(
  kernel,
  out_shape=output_shape_struct,
  grid_spec=True,
  grid=grid,
  in_specs=[
    pl.BlockSpec(x.shape, lambda *_: (0,) * x.ndim),
    pl.BlockSpec(params["kernel"].shape, lambda *_: (0,) * params["kernel"].ndim),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(bD, bH, bW, out_channels),
    index_map=lambda n, d, h, w: (n, d * bD, h * bH, w * bW, 0),
  ),
)(x, params["kernel"]).block_until_ready()
