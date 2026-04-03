# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# JAX uses channels-last format (N, D, H, W, C)
x = random.normal(x_key, (batch_size, D, H, W, in_channels))

# In Flax, the layer and its parameters are separate.
# We define the layer structure first.
conv = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=1,
  padding="VALID",  # PyTorch padding=0 is 'VALID' in JAX
)
params = conv.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for fused 3D convolution, mish, and tanh.

  This kernel computes a single row of the output tensor.

  Args:
    x_ref: A reference to the input slice. The slice contains the receptive
      field needed to compute one output row.
    kernel_ref: A reference to the full 3D convolution kernel weights.
    bias_ref: A reference to the convolution bias vector.
    out_ref: A reference to the output buffer where the result for the
      computed row is written.
  """
  # The `conv_general_dilated` primitive is not supported in Pallas kernels.
  # Instead, we implement the convolution manually using a for-loop and tensordot.
  # This kernel is responsible for computing an output slice of shape
  # (1, 1, out_W, out_channels). We iterate over the output width dimension.
  for w_o in range(out_ref.shape[3]):
    # Extract the receptive field (patch) for the current output position.
    # This is a sliding window of size `kernel_size` across the width of `x_ref`.
    patch = x_ref[0, :, :, w_o : w_o + kernel_size, :]

    # Perform the convolution for one output "pixel" by contracting the patch
    # with the kernel. This avoids the reshape operations that are not well
    # supported in Pallas on TPU.
    # axes=4 contracts the 4 dimensions of the patch (D, H, W, C_in) with
    # the first 4 dimensions of the kernel (D, H, W, C_in, C_out).
    conv_pixel = jnp.tensordot(patch, kernel_ref[...], axes=4)

    # Add the bias term.
    with_bias = conv_pixel + bias_ref[...]

    # Apply the first activation function, Mish, element-wise.
    mish_out = jax.nn.mish(with_bias)

    # Apply the second activation function, Tanh, element-wise.
    tanh_out = jnp.tanh(mish_out)

    # Write the final result to the corresponding position in the output reference.
    out_ref[0, 0, 0, w_o, :] = tanh_out


# The original computation involves a 3D convolution followed by element-wise
# mish and tanh activation functions. The Pallas kernel will fuse these
# operations.

# 1. Define the output shape of the computation.
# With 'VALID' padding and strides=1, the output spatial dimensions are reduced.
out_D = x.shape[1] - kernel_size + 1
out_H = x.shape[2] - kernel_size + 1
out_W = x.shape[3] - kernel_size + 1
output_shape = jax.ShapeDtypeStruct((batch_size, out_D, out_H, out_W, out_channels), x.dtype)

# 2. Construct the pallas_call to replace the computation.
# The grid is designed to parallelize across the batch, depth, and height
# dimensions of the output. Each kernel instance computes a single row
# (spanning all width and channels) of the output feature map.
x = pl.pallas_call(
  kernel,
  out_shape=output_shape,
  grid=(batch_size, out_D, out_H),
  in_specs=[
    # Input 'x': Each kernel instance (i, j, k) gets a slice of the
    # input tensor needed to compute the output at batch i, depth j, height k.
    # The slice must cover the receptive field of the convolution kernel.
    pl.BlockSpec(
      block_shape=(1, kernel_size, kernel_size, W, in_channels),
      index_map=lambda i, j, k: (i, j, k, 0, 0),
    ),
    # Kernel weights: The entire kernel is passed to each instance.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda i, j, k: (0, 0, 0, 0, 0),
    ),
    # Bias: The entire bias vector is passed to each instance.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j, k: (0,)),
  ],
  # Output 'x': Each kernel instance writes to a unique row in the output tensor.
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, out_W, out_channels),
    index_map=lambda i, j, k: (i, j, k, 0, 0),
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()
