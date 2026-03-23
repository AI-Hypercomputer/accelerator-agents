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
num_groups = 16
multiply_weight_shape = (out_features,)

key = random.PRNGKey(0)
key, x_key, gemm_key, gn_key, mw_key = random.split(key, 5)

x = random.normal(x_key, (batch_size, in_features))

gemm = nn.Dense(features=out_features)
gemm_params = gemm.init(gemm_key, x)["params"]

group_norm = nn.GroupNorm(num_groups=num_groups)
# The input to GroupNorm has shape (batch_size, out_features)
group_norm_dummy_input = jnp.ones((batch_size, out_features))
group_norm_params = group_norm.init(gn_key, group_norm_dummy_input)["params"]

multiply_weight = random.normal(mw_key, multiply_weight_shape)

batch_block_size = 8


# Computation
def kernel(x_ref, gemm_kernel_ref, gemm_bias_ref, gn_scale_ref, gn_bias_ref, multiply_weight_ref, out_ref):
  # Constants from the model architecture
  num_groups = 16
  out_features = 1024
  epsilon = 1e-5
  batch_block_size = x_ref.shape[0]
  channels_per_group = out_features // num_groups

  # 1. Dense layer (GEMM)
  # x_ref has shape (batch_block_size, in_features)
  # gemm_kernel_ref has shape (in_features, out_features)
  # The result 'y' has shape (batch_block_size, out_features)
  y = x_ref[...] @ gemm_kernel_ref[...]
  y = y + gemm_bias_ref[...]

  # Store intermediate GEMM result in out_ref to be able to load slices from it
  out_ref[...] = y

  # 2. Group Normalization (in-place on out_ref)
  for g in range(num_groups):
    start_channel = g * channels_per_group
    # Load a group from out_ref (which currently holds the GEMM result)
    y_group = out_ref[pl.dslice(0, batch_block_size), pl.dslice(start_channel, channels_per_group)]

    # Calculate statistics over the group channels (axis=1)
    mean = jnp.mean(y_group, axis=1, keepdims=True)
    var = jnp.var(y_group, axis=1, keepdims=True)

    # Normalize the group
    y_group_normalized = (y_group - mean) / jnp.sqrt(var + epsilon)

    # Load the scale and bias for this group
    gn_scale_group = gn_scale_ref[pl.dslice(start_channel, channels_per_group)]
    gn_bias_group = gn_bias_ref[pl.dslice(start_channel, channels_per_group)]

    # Apply scale and bias
    y_group_scaled = y_group_normalized * gn_scale_group + gn_bias_group

    # Write the processed group back to the output buffer
    out_ref[pl.dslice(0, batch_block_size), pl.dslice(start_channel, channels_per_group)] = y_group_scaled

  # Load the result of the group normalization
  y = out_ref[...]

  # 3. First SiLU activation (Sigmoid-weighted Linear Unit)
  y = y * jax.nn.sigmoid(y)

  # 4. Element-wise multiplication with a learned weight
  y = y * multiply_weight_ref[...]

  # 5. Second SiLU activation
  y = y * jax.nn.sigmoid(y)

  # Store the final result in the output buffer
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_block_size, in_features), index_map=lambda i: (i * batch_block_size, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, out_features), index_map=lambda i: (i * batch_block_size, 0)),
)(
  x,
  gemm_params["kernel"],
  gemm_params["bias"],
  group_norm_params["scale"],
  group_norm_params["bias"],
  multiply_weight,
).block_until_ready()
