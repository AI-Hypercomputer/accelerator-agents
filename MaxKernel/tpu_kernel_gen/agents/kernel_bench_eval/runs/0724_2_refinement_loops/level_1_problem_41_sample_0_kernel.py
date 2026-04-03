# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
key = random.PRNGKey(0)
# Flax pooling expects input shape (N, L, C) vs PyTorch's (N, C, L)
x = random.normal(key, (batch_size, sequence_length, features))

# Calculate output shape based on pooling parameters
dilated_kernel_size = (kernel_size - 1) * dilation + 1
output_seq_len = (sequence_length + 2 * padding - dilated_kernel_size) // stride + 1
output_shape = (batch_size, output_seq_len, features)


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 1D max pooling with dilation and padding.

  This kernel translates a `lax.reduce_window` operation with `lax.max`
  for a single item in a batch. The `pallas_call` invocation pattern
  maps each program in the grid to a single batch item.

  Args:
    x_ref: A reference to the input data for a single batch item, with shape
      (1, sequence_length, features).
    out_ref: A reference to the output buffer, with shape
      (1, output_seq_len, features). The result is written here in-place.
  """
  # The parameters `kernel_size`, `stride`, `padding`, `dilation`,
  # `sequence_length`, `features`, and `output_seq_len` are captured from
  # the outer scope where this kernel is defined.

  # Iterate over each position in the output's sequence dimension.
  for j in range(output_seq_len):
    # Initialize the max value for the current window to negative infinity,
    # which is the identity element for the max operation.
    window_max = jnp.full((features,), -jnp.inf, dtype=x_ref.dtype)

    # Iterate over each element within the pooling window.
    for m in range(kernel_size):
      # Calculate the corresponding index in the original (unpadded) input.
      # The formula accounts for stride, dilation, and padding:
      # input_idx = (output_pos * stride) + (window_pos * dilation) - padding
      input_idx = j * stride - padding + m * dilation

      # Check if the calculated index is within the valid bounds of the input.
      # This check effectively handles the padding; indices that fall into the
      # padded region are ignored, so their value remains the initial -inf.
      if (input_idx >= 0) and (input_idx < sequence_length):
        # If the index is valid, load the input data for all features.
        current_val = x_ref[0, input_idx, :]
        # Update the running maximum for the current window.
        window_max = jnp.maximum(window_max, current_val)

    # After scanning the whole window, write the final max value to the output.
    out_ref[0, j, :] = window_max


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, sequence_length, features), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, output_seq_len, features), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
