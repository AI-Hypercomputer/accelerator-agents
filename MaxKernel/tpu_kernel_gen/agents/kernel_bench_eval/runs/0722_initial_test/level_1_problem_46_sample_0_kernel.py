# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1
key = random.PRNGKey(0)
# JAX/Pallas expect channels-last format: (N, D, H, W, C)
x = random.normal(key, (batch_size, depth, height, width, channels))

# Calculate output shape
out_depth = (depth + 2 * padding - kernel_size) // stride + 1
out_height = (height + 2 * padding - kernel_size) // stride + 1
out_width = (width + 2 * padding - kernel_size) // stride + 1
output_shape = (batch_size, out_depth, out_height, out_width, channels)


# Computation
def kernel(x_ref, o_ref):
  """Pallas kernel for 3D average pooling.

  Args:
    x_ref: A reference to the input block. The shape is
      (1, kernel_size, kernel_size, kernel_size, channels).
    o_ref: A reference to the output block. The shape is
      (1, 1, 1, 1, channels).
  """
  # The kernel_size is a constant captured from the surrounding scope.
  kernel_size = 3
  # Sum the values in the input window across the spatial dimensions (D, H, W).
  # x_ref has shape (1, D, H, W, C), so we sum over axes 1, 2, and 3.
  # The result will have shape (1, C), which broadcasts to the output shape.
  window_sum = jnp.sum(x_ref[...], axis=(1, 2, 3), keepdims=True)
  # Divide by the number of elements in the window to get the average.
  o_ref[...] = window_sum / (kernel_size * kernel_size * kernel_size)


# Pallas call
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, out_depth, out_height, out_width),
  in_specs=[
    pl.BlockSpec(
      (1, kernel_size, kernel_size, kernel_size, channels),
      lambda n, d, h, w: (
        n,
        d * stride - padding,
        h * stride - padding,
        w * stride - padding,
        0,
      ),
    )
  ],
  out_specs=pl.BlockSpec((1, 1, 1, 1, channels), lambda n, d, h, w: (n, d, h, w, 0)),
)(x).block_until_ready()
