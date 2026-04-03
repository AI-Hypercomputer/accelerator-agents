# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

key = random.PRNGKey(0)
key, x_key, gemm_key, bias_key, groupnorm_key = random.split(key, 5)

x = random.normal(x_key, (batch_size, in_features))

gemm = nn.Dense(features=out_features)
gemm_params = gemm.init(gemm_key, x)["params"]

bias = random.normal(bias_key, bias_shape)

groupnorm = nn.GroupNorm(num_groups=num_groups)
groupnorm_params = groupnorm.init(groupnorm_key, jnp.ones((batch_size, out_features)))["params"]

# Define block sizes for tiling the computation
bM = 32
bN = 128


# Computation
def kernel(x_ref, gemm_kernel_ref, gemm_bias_ref, bias_ref, groupnorm_scale_ref, groupnorm_bias_ref, out_ref):
  """
  Pallas kernel for a sequence of GEMM, activations, and GroupNorm.

  Args:
    x_ref: Input data block.
    gemm_kernel_ref: Weight matrix for the dense layer.
    gemm_bias_ref: Bias vector for the dense layer.
    bias_ref: External bias vector.
    groupnorm_scale_ref: Scale parameter for GroupNorm.
    groupnorm_bias_ref: Bias parameter for GroupNorm.
    out_ref: Output buffer.
  """
  # Get the grid index for the column dimension.
  j = pl.program_id(axis=1)

  # Constants for GroupNorm
  epsilon = 1e-6

  # 1. Apply GEMM (Dense layer)
  y = x_ref[...] @ gemm_kernel_ref[...]

  # 2. Add biases
  col_offset = j * bN

  # Use lax.dynamic_slice to handle dynamic indices.
  gemm_bias_block = jax.lax.dynamic_slice(gemm_bias_ref[...], (col_offset,), (bN,))
  bias_block = jax.lax.dynamic_slice(bias_ref[...], (col_offset,), (bN,))

  y += gemm_bias_block.astype(y.dtype)
  y += bias_block.astype(y.dtype)

  # 3. Apply activation functions
  y = nn.hard_tanh(y)
  y = jax.nn.mish(y)

  # 4. Apply Group Normalization
  channels_per_group = out_features // num_groups
  num_groups_in_block = y.shape[1] // channels_per_group

  # Dynamically slice the scale and bias vectors for GroupNorm.
  groupnorm_scale_block = jax.lax.dynamic_slice(groupnorm_scale_ref[...], (col_offset,), (bN,))
  groupnorm_bias_block = jax.lax.dynamic_slice(groupnorm_bias_ref[...], (col_offset,), (bN,))

  y_out = jnp.empty_like(y)

  # This loop will be unrolled by JAX, so standard slicing is fine here.
  for g in range(num_groups_in_block):
    start_col = g * channels_per_group
    end_col = (g + 1) * channels_per_group
    group_slice = slice(start_col, end_col)

    y_group = y[:, group_slice]

    mean = jnp.mean(y_group, axis=1, keepdims=True)
    var = jnp.var(y_group, axis=1, keepdims=True)

    y_norm = (y_group - mean) / jnp.sqrt(var + epsilon)

    scale_group = groupnorm_scale_block[group_slice]
    bias_group = groupnorm_bias_block[group_slice]
    y_final_group = y_norm * scale_group + bias_group

    y_out = y_out.at[:, group_slice].set(y_final_group)

  out_ref[...] = y_out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // bM, out_features // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, in_features), index_map=lambda i, j: (i * bM, 0)),
    pl.BlockSpec(block_shape=(in_features, bN), index_map=lambda i, j: (0, j * bN)),
    # Load the full 1D vectors.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i * bM, j * bN)),
)(
  x,
  gemm_params["kernel"],
  gemm_params["bias"],
  bias,
  groupnorm_params["scale"],
  groupnorm_params["bias"],
).block_until_ready()
