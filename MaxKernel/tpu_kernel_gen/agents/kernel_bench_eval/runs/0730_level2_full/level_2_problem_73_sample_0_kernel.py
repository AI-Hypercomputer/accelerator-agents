# Imports
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses (N, H, W, C) channel-last convention
x = random.normal(key_x, (batch_size, height, width, in_channels))


# In Flax, layers are typically defined within a Module
class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    # The original code implies training mode for BatchNorm
    # so we set use_running_average=False to update batch statistics.
    x = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))(x)
    x = nn.BatchNorm(use_running_average=False)(x)
    return x


model = Model()
# Initialize parameters and batch statistics
variables = model.init(key_params, x)


# Computation
def kernel(
  x_ref,
  conv_kernel_ref,
  bn_scale_ref,
  bn_bias_ref,
  # The following two refs (mean_ref, var_ref) are the running averages from the
  # previous state, but are unused since use_running_average=False.
  # They are included to match the pallas_call signature.
  mean_ref,
  var_ref,
  # Output references
  out_ref,
  batch_mean_out_ref,
  batch_var_out_ref,
  # Static arguments
  *,
  scaling_factor: float,
):
  """
  Pallas kernel for a fused Conv -> BatchNorm -> Scale operation.

  This kernel processes a slice of the input batch, applies a convolution,
  then normalizes the result using batch statistics computed across all
  concurrent kernel instances.

  Args:
    x_ref: Reference to the input image data for this instance.
    conv_kernel_ref: Reference to the convolution kernel weights.
    bn_scale_ref: Reference to the scale parameters of the BatchNorm layer.
    bn_bias_ref: Reference to the bias parameters of the BatchNorm layer.
    mean_ref: Reference to the input running average of the mean (unused).
    var_ref: Reference to the input running average of the variance (unused).
    out_ref: Reference to the output buffer for the final feature map.
    batch_mean_out_ref: Reference to the output buffer for the computed batch mean.
    batch_var_out_ref: Reference to the output buffer for the computed batch variance.
    scaling_factor: A static float value for the final scaling operation.
  """
  # --- 1. Convolution Layer ---
  # Perform 2D convolution. The dimension numbers correspond to the 'NHWC'
  # data format for the input/output and 'HWIO' for the kernel.
  conv_out = lax.conv_general_dilated(
    lhs=x_ref[...],
    rhs=conv_kernel_ref[...],
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # --- 2. Batch Normalization (Training Mode) ---
  # In training mode, we calculate the mean and variance of the current batch.
  # This requires a cross-program reduction.

  # First, calculate the sum and sum of squares for the local data slice.
  # The reduction is over spatial dimensions (0, 1, 2) of the (1, H, W, C) slice.
  num_elements_per_channel = conv_out.shape[0] * conv_out.shape[1] * conv_out.shape[2]
  local_sum = jnp.sum(conv_out, axis=(0, 1, 2))
  local_sum_of_squares = jnp.sum(conv_out * conv_out, axis=(0, 1, 2))

  # Create scratchpads for aggregating statistics across programs.
  num_channels = bn_scale_ref.shape[0]
  sum_scratch = pl.make_scratch([num_channels], dtype=jnp.float32)
  sum_of_squares_scratch = pl.make_scratch([num_channels], dtype=jnp.float32)

  # Initialize scratchpads to zero from the first program.
  @pl.when(pl.program_id(0) == 0)
  def _init_scratch():
    sum_scratch[...] = jnp.zeros_like(sum_scratch)
    sum_of_squares_scratch[...] = jnp.zeros_like(sum_of_squares_scratch)

  # Synchronize to ensure scratchpads are initialized before use.
  pl.barrier()

  # Atomically add local statistics to the shared scratchpads.
  pl.atomic_add(sum_scratch, 0, local_sum)
  pl.atomic_add(sum_of_squares_scratch, 0, local_sum_of_squares)

  # Synchronize to ensure all atomic operations are complete.
  pl.barrier()

  # Load the aggregated sums and calculate the final batch statistics.
  total_elements_per_channel = pl.grid(0) * num_elements_per_channel
  batch_mean = sum_scratch[...] / total_elements_per_channel
  # E[x^2] - (E[x])^2
  batch_var = (sum_of_squares_scratch[...] / total_elements_per_channel) - (batch_mean * batch_mean)

  # Normalize the convolution output using the computed batch statistics.
  epsilon = 1e-5
  normalized_out = (conv_out - batch_mean) / jnp.sqrt(batch_var + epsilon)
  bn_out = normalized_out * bn_scale_ref[...] + bn_bias_ref[...]

  # --- 3. Final Scaling and Output ---
  # Apply the final scaling factor.
  final_out = bn_out * scaling_factor
  out_ref[...] = final_out.astype(out_ref.dtype)

  # The first program writes the final computed batch statistics to the output.
  @pl.when(pl.program_id(0) == 0)
  def _write_stats():
    batch_mean_out_ref[...] = batch_mean.astype(batch_mean_out_ref.dtype)
    batch_var_out_ref[...] = batch_var.astype(batch_var_out_ref.dtype)


# The original computation returns a tuple: (output_tensor, updated_state_dict).
# We define the shape of this nested structure for the Pallas kernel's output.
final_x_shape = (batch_size, height, width, out_channels)
stats_shape = (out_channels,)
out_shapes = (
  jax.ShapeDtypeStruct(final_x_shape, x.dtype),
  {
    "batch_stats": {
      "BatchNorm_0": {
        "mean": jax.ShapeDtypeStruct(stats_shape, x.dtype),
        "var": jax.ShapeDtypeStruct(stats_shape, x.dtype),
      }
    }
  },
)

# We replace the Flax model's apply method with a call to a custom Pallas kernel.
# The kernel is assumed to perform the entire fused operation: Conv -> BatchNorm -> Scaling.
x, updated_state = pl.pallas_call(
  kernel,
  out_shape=out_shapes,
  grid=(batch_size,),
  in_specs=[
    # Input 'x' is chunked along the batch dimension. Each kernel instance processes one image.
    pl.BlockSpec((1, height, width, in_channels), lambda i: (i, 0, 0, 0)),
    # All model parameters and initial batch stats are passed in full to each kernel instance.
    pl.BlockSpec(variables["params"]["Conv_0"]["kernel"].shape, lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(variables["params"]["BatchNorm_0"]["scale"].shape, lambda i: (0,)),
    pl.BlockSpec(variables["params"]["BatchNorm_0"]["bias"].shape, lambda i: (0,)),
    pl.BlockSpec(variables["batch_stats"]["BatchNorm_0"]["mean"].shape, lambda i: (0,)),
    pl.BlockSpec(variables["batch_stats"]["BatchNorm_0"]["var"].shape, lambda i: (0,)),
  ],
  out_specs=(
    # The output tensor is chunked along the batch dimension, matching the input chunking.
    pl.BlockSpec((1, height, width, out_channels), lambda i: (i, 0, 0, 0)),
    {
      "batch_stats": {
        "BatchNorm_0": {
          # The updated batch statistics are single, non-chunked outputs.
          # The kernel is assumed to handle the necessary cross-instance reduction.
          "mean": pl.BlockSpec(stats_shape, lambda i: (0,)),
          "var": pl.BlockSpec(stats_shape, lambda i: (0,)),
        }
      }
    },
  ),
  # The scaling_factor is passed as a static keyword argument to the kernel.
  scaling_factor=scaling_factor,
)(
  x,
  variables["params"]["Conv_0"]["kernel"],
  variables["params"]["BatchNorm_0"]["scale"],
  variables["params"]["BatchNorm_0"]["bias"],
  variables["batch_stats"]["BatchNorm_0"]["mean"],
  variables["batch_stats"]["BatchNorm_0"]["var"],
).block_until_ready()
