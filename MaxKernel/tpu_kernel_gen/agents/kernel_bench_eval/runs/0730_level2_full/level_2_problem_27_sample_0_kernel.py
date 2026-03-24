# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), use_bias=True)
params = conv.init(key_params, x)["params"]

bB = 8  # Block size for the batch dimension


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel that performs a sequence of CNN operations.
  """
  # Load the input data, convolution kernel, and bias from memory.
  x = x_ref[...]
  conv_kernel = kernel_ref[...]
  conv_bias = bias_ref[...]

  # Optimized manual 3D convolution.
  # This implementation avoids unsupported primitives and slow, deeply nested
  # loops by iterating only over the kernel dimensions and using vectorized
  # operations for the spatial dimensions.
  b, d, h, w, c_in = x.shape
  kd, kh, kw, _, c_out = conv_kernel.shape
  pad_d, pad_h, pad_w = kd // 2, kh // 2, kw // 2

  # Pad the input for 'SAME' convolution.
  x_padded = jnp.pad(x, [(0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0)])

  # Initialize the output tensor for accumulation.
  conv_out = jnp.zeros((b, d, h, w, c_out), dtype=x.dtype)

  # Iterate over the kernel dimensions.
  for i in range(kd):
    for j in range(kh):
      for k in range(kw):
        # Extract the slice of the kernel for this position.
        kernel_slice = conv_kernel[i, j, k, :, :]  # Shape: (c_in, c_out)

        # Extract the corresponding slice from the padded input.
        # The slice is shifted according to the kernel position.
        # lax.slice_in_dim is used because the slice parameters are static.
        input_slice = lax.slice_in_dim(x_padded, i, i + d, axis=1)
        input_slice = lax.slice_in_dim(input_slice, j, j + h, axis=2)
        input_slice = lax.slice_in_dim(input_slice, k, k + w, axis=3)
        # Shape of input_slice: (b, d, h, w, c_in)

        # Perform matrix multiplication between the input slice and kernel slice
        # and accumulate the result.
        # (b, d, h, w, c_in) @ (c_in, c_out) -> (b, d, h, w, c_out)
        conv_out += jnp.dot(input_slice, kernel_slice)

  x = conv_out

  # Add the bias. The bias is broadcast across the batch and spatial dimensions.
  x = x + conv_bias

  # Apply the sequence of activation functions.
  x = nn.hard_swish(x)
  x = nn.relu(x)
  x = nn.softmax(x, axis=-1)

  # Perform global average pooling by taking the mean over the spatial axes.
  # This reduces the tensor from (bB, D, H, W, C) to (bB, C).
  final_output = jnp.mean(x, axis=(1, 2, 3))

  # Write the final result to the output buffer.
  out_ref[...] = final_output


x_out = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_channels), x.dtype),
  grid=(batch_size // bB,),
  in_specs=[
    pl.BlockSpec(block_shape=(bB, *x.shape[1:]), index_map=lambda i: (i * bB, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bB, out_channels), index_map=lambda i: (i * bB, 0)),
)(x, params["kernel"], params["bias"]).block_until_ready()
