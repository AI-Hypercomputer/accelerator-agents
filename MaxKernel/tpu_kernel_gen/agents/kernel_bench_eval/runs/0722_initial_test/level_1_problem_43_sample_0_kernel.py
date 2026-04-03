# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding_val = 1
dilation = 3
key = random.PRNGKey(0)

# JAX/Flax uses channels-last convention: (N, D, H, W, C)
x_shape = (batch_size, dim1, dim2, dim3, channels)
x_dtype = jnp.float32
x = random.normal(key, x_shape, dtype=x_dtype)

# Pooling parameters
window_shape = (kernel_size, kernel_size, kernel_size)
strides = (stride, stride, stride)
padding = [(padding_val, padding_val), (padding_val, padding_val), (padding_val, padding_val)]
window_dilation = (dilation, dilation, dilation)


def get_pool_output_shape(input_shape, window_shape, strides, padding, dilation):
  """Calculates the output shape of a pooling operation."""
  input_dims = input_shape[1:-1]
  output_spatial_dims = []
  for i in range(len(input_dims)):
    # Formula for output dimension: floor((N + 2*P - D*(K-1) - 1)/S) + 1
    dilated_kernel_size = dilation[i] * (window_shape[i] - 1) + 1
    pad_before = padding[i][0]
    pad_after = padding[i][1]
    output_dim = (input_dims[i] + pad_before + pad_after - dilated_kernel_size) // strides[i] + 1
    output_spatial_dims.append(output_dim)
  return (input_shape[0], *output_spatial_dims, input_shape[-1])


# Calculate the expected output shape
output_shape = get_pool_output_shape(x_shape, window_shape, strides, padding, window_dilation)
out_d, out_h, out_w = output_shape[1:4]
dilated_kernel_size = dilation * (kernel_size - 1) + 1


# Computation
def kernel(x_ref, o_ref):
  """
  Pallas kernel for 3D max pooling.

  This kernel computes the maximum value over a sliding window of the input tensor.
  The grid is defined over the output dimensions (batch, depth, height, width).
  For each output element, the corresponding window is loaded from the input
  tensor `x_ref` into SRAM. The maximum value is then computed over the spatial
  dimensions of this window for each channel independently.

  Args:
    x_ref: A reference to the entire input tensor in HBM.
    o_ref: A reference to the output block, which is a single element
      (across all channels) of the output tensor. This kernel writes the
      computed maximum value into this reference.
  """
  n, od, oh, ow = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)

  start_d = od * stride - padding_val
  start_h = oh * stride - padding_val
  start_w = ow * stride - padding_val

  window_block_shape = (1, dilated_kernel_size, dilated_kernel_size, dilated_kernel_size, channels)
  neg_inf = jnp.array(-jnp.inf, dtype=x_ref.dtype)

  # Load window from HBM to SRAM, with padding for boundary handling.
  x_sram = pl.load(
    x_ref,
    (n, start_d, start_h, start_w, 0),
    block_shape=window_block_shape,
    padding_mode="constant",
    padding_config=pl.PaddingConfig(constant_values=(neg_inf,)),
  )

  # We select the elements corresponding to the dilated kernel.
  x_dilated = x_sram[:, ::dilation, ::dilation, ::dilation, :]
  # The result of the max operation will have shape (1, 1, 1, 1, channels),
  # which matches the shape of o_ref.
  o_ref[...] = jnp.max(x_dilated, axis=(1, 2, 3), keepdims=True)


# The pallas_call invocation replaces the original computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x_dtype),
  in_specs=[pl.BlockSpec(memory_space=pl.MemorySpace.HBM)],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, 1, channels),
    index_map=lambda n, od, oh, ow: (n, od, oh, ow, 0),
  ),
  grid=(batch_size, out_d, out_h, out_w),
)(x)
