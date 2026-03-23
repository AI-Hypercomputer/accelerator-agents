# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 8
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 256
width = 256
stride = (1, 1, 1)
# PyTorch padding=(0,0,0) is 'VALID' in JAX/Flax
padding = "VALID"
dilation = (1, 1, 1)
groups = 1
bias = False

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)

key = random.PRNGKey(0)
params_key, x_key = random.split(key)

# Note: JAX expects channels-last format (NDHWC) for convolutions
# For TPU compatibility, C_in must be a multiple of 8.
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))
variables = conv3d.init(params_key, x)


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution.

  This kernel computes a tile of a 3D convolution operation. It iterates
  through the width dimension of the convolution kernel, performing a
  vectorized multiply-add for each slice.

  Args:
    x_ref: A reference to the input tensor tile. This tile includes a "halo"
      region along the width dimension, which is necessary for the convolution.
      Expected shape: (1, KD, KH, in_block_w, C_in).
    kernel_ref: A reference to the complete convolution kernel tensor.
      Expected shape: (KD, KH, KW, C_in, C_out).
    out_ref: A reference to the output tensor tile that this kernel instance
      is responsible for computing. This is an in-place operation, and the
      result is written directly to this reference.
      Expected shape: (1, 1, 1, block_w, C_out).
  """
  # Get kernel dimensions and output block width from the shapes.
  KD, KH, KW, C_in, C_out = kernel_ref.shape
  block_w = out_ref.shape[3]

  # Initialize an accumulator with the same shape as the output tile.
  acc = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)

  # The main logic iterates over the width of the convolution kernel (KW).
  for kw in range(KW):
    # For each position `kw` in the kernel's width, we take a corresponding
    # slice from the input `x_ref`. The slice starts at `kw` and has a width
    # equal to the output block's width (`block_w`).
    x_slice = x_ref[:, :, :, kw : kw + block_w, :]

    # We also select the corresponding 2D slice from the kernel weights.
    k_slice = kernel_ref[:, :, kw, :, :]

    # Squeeze the batch dimension from x_slice for computation.
    x_squeezed = jnp.squeeze(x_slice, axis=0)

    # Reshape inputs to have a single batch dimension (KD * KH) for matmul.
    x_reshaped = x_squeezed.reshape(KD * KH, block_w, C_in)
    k_reshaped = k_slice.reshape(KD * KH, C_in, C_out)

    # Perform a batch matrix multiplication.
    # 'b' is the batch dim, 'w' is feature dim, 'c' is contracting dim.
    out_contribution_batch = jnp.einsum("bwc,bco->bwo", x_reshaped, k_reshaped, preferred_element_type=out_ref.dtype)

    # Sum over the batch dimension to get the final contribution.
    out_contribution = out_contribution_batch.sum(axis=0)

    # The result is added to the accumulator. It's reshaped
    # to match the 5D shape of the output tile.
    acc += out_contribution.reshape(out_ref.shape)

  # After iterating through the entire kernel width, the accumulator holds
  # the final values for the output tile. We write this result to the
  # output reference, performing the computation in-place.
  out_ref[...] = acc


# Convolution parameters from the initialization code
N, D_in, H_in, W_in, C_in = (batch_size, depth, height, width, in_channels)
KD, KH, KW = kernel_size
C_out = out_channels
S_D, S_H, S_W = stride
DIL_D, DIL_H, DIL_W = dilation

# Calculate output dimensions for 'VALID' padding
D_out = (D_in - (KD - 1) * DIL_D - 1) // S_D + 1
H_out = (H_in - (KH - 1) * DIL_H - 1) // S_H + 1
W_out = (W_in - (KW - 1) * DIL_W - 1) // S_W + 1
output_shape = (N, D_out, H_out, W_out, C_out)

# Define a TPU-friendly block size for the width dimension.
block_w = 128

# Define the execution grid. We flatten the (N, D_out, H_out) dimensions
# into the first grid dimension and tile the W_out dimension in the second.
grid_n_d_h = N * D_out * H_out
grid_w = (W_out + block_w - 1) // block_w
grid = (grid_n_d_h, grid_w)

# The input block for 'x' must include a halo region to account for the
# convolution kernel's size.
in_block_w = block_w + (KW - 1) * DIL_W
# Pad the block size to be a multiple of 8 for TPU compatibility.
padded_in_block_w = (in_block_w + 7) // 8 * 8

# Computation
# The pallas_call replaces the conv3d.apply(variables, x) computation.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    # Spec for the input tensor 'x'.
    # Each kernel instance gets a slice of 'x' corresponding to the output
    # tile it computes, plus a halo.
    pl.BlockSpec(
      block_shape=(1, KD, KH, padded_in_block_w, C_in),
      index_map=lambda i, j: (
        i // (D_out * H_out),  # Batch index
        (i // H_out) % D_out,  # Output depth index
        i % H_out,  # Output height index
        j * block_w,  # Output width start index
        0,
      ),
    ),
    # Spec for the convolution kernel weights.
    # The entire kernel is passed to each Pallas program instance.
    pl.BlockSpec(
      block_shape=(KD, KH, KW, C_in, C_out),
      index_map=lambda i, j: (0, 0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # Each kernel instance computes a tile of the output.
    block_shape=(1, 1, 1, block_w, C_out),
    index_map=lambda i, j: (
      i // (D_out * H_out),  # Batch index
      (i // H_out) % D_out,  # Output depth index
      i % H_out,  # Output height index
      j * block_w,  # Output width start index
      0,
    ),
  ),
)(x, variables["params"]["kernel"]).block_until_ready()
