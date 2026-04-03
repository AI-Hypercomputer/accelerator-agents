# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
num_groups = 8
hardtanh_min = -2.0
hardtanh_max = 2.0

key = random.PRNGKey(0)
key, x_key, gemm_key, gn_key = random.split(key, 4)

x = random.normal(x_key, (batch_size, in_features))

gemm = nn.Dense(features=out_features)
gemm_params = gemm.init(gemm_key, x)["params"]

group_norm = nn.GroupNorm(num_groups=num_groups)
group_norm_params = group_norm.init(gn_key, jnp.ones((batch_size, out_features)))["params"]


# Computation
def kernel(x_ref, gemm_kernel_ref, gemm_bias_ref, gn_scale_ref, gn_bias_ref, out_ref):
  """
  Pallas kernel that performs a GEMM, followed by Group Normalization, and a clip operation.

  Args:
    x_ref: Input data block.
    gemm_kernel_ref: Weight matrix for the GEMM.
    gemm_bias_ref: Bias vector for the GEMM.
    gn_scale_ref: Scale parameter for Group Normalization.
    gn_bias_ref: Bias parameter for Group Normalization.
    out_ref: Output buffer to store the result.
  """
  # Get the group index from the program ID
  group_idx = pl.program_id(axis=1)

  # Constants for the operations
  epsilon = 1e-5
  hardtanh_min = -2.0
  hardtanh_max = 2.0
  features_per_group = out_features // num_groups
  bS = x_ref.shape[0]

  # 1. Apply the dense layer (GEMM)
  # x_ref has shape (bS, in_features)
  # gemm_kernel_ref has shape (in_features, out_features)
  # gemm_bias_ref has shape (out_features,)
  x_gemm = x_ref[...] @ gemm_kernel_ref[...] + gemm_bias_ref[...]
  # x_gemm now has shape (bS, out_features)

  # 2. Select the slice for the current group using dynamic_slice
  group_x = jax.lax.dynamic_slice(x_gemm, (0, group_idx * features_per_group), (bS, features_per_group))

  # 3. Apply Layer Normalization to the group slice
  # The reduction is over the features within the group (axis=1)
  mean = jnp.mean(group_x, axis=1, keepdims=True)
  var = jnp.var(group_x, axis=1, keepdims=True)
  x_norm = (group_x - mean) / jnp.sqrt(var + epsilon)

  # 4. Apply the learnable scale and bias parameters for the group
  # We also need to slice the scale and bias parameters.
  group_scale = jax.lax.dynamic_slice(gn_scale_ref[...], (group_idx * features_per_group,), (features_per_group,))
  group_bias = jax.lax.dynamic_slice(gn_bias_ref[...], (group_idx * features_per_group,), (features_per_group,))
  x_scaled = x_norm * group_scale + group_bias

  # 5. Apply the clipping operation (HardTanh)
  result = jnp.clip(x_scaled, a_min=hardtanh_min, a_max=hardtanh_max)

  # 6. Write the final result to the output reference
  # out_ref has shape (bS, features_per_group)
  out_ref[...] = result


# Define a block size for the batch dimension that adheres to TPU constraints (divisible by 8)
bS = 8
features_per_group = out_features // num_groups

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // bS, num_groups),
  in_specs=[
    # Input 'x' is sliced along the batch dimension.
    pl.BlockSpec(block_shape=(bS, in_features), index_map=lambda i, j: (i, 0)),
    # The full GEMM weights are passed to each kernel.
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i, j: (0, 0)),
    # The full GEMM bias is passed to each kernel.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    # The full GroupNorm scale is passed to each kernel.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    # The full GroupNorm bias is passed to each kernel.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
  ],
  # The output is sliced by batch and group.
  out_specs=pl.BlockSpec(block_shape=(bS, features_per_group), index_map=lambda i, j: (i, j)),
)(
  x,
  gemm_params["kernel"],
  gemm_params["bias"],
  group_norm_params["scale"],
  group_norm_params["bias"],
).block_until_ready()
