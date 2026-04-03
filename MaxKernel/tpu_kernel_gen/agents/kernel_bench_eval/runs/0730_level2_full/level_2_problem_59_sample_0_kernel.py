# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0
key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))
b_batch = 64
b_out = 128


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref, scaling_factor: float):
  """Pallas kernel for a fused linear, SiLU, and scaling operation.

  Args:
    x_ref: A reference to the input block.
    weight_ref: A reference to the weight block.
    bias_ref: A reference to the bias block.
    out_ref: A reference to the output block.
    scaling_factor: A compile-time constant for scaling.
  """
  # Perform the matrix multiplication of a block of x with a block of weight.T
  # and add the bias.
  # x_ref.shape = (b_batch, in_features)
  # weight_ref.shape = (out_features, in_features)
  # bias_ref.shape = (out_features,)
  # The result `y` will have shape (b_batch, out_features).
  y = jnp.dot(x_ref[...], weight_ref[...].T) + bias_ref[...]

  # Apply the SiLU activation function (x * sigmoid(x)) element-wise.
  y = y * jax.nn.sigmoid(y)

  # Apply the scaling factor and write the final result to the output.
  out_ref[...] = y * scaling_factor


result = pl.pallas_call(
  lambda x_ref, w_ref, b_ref, o_ref: kernel(x_ref, w_ref, b_ref, o_ref, scaling_factor=scaling_factor),
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // b_batch,),
  in_specs=[
    pl.BlockSpec(block_shape=(b_batch, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_batch, out_features), index_map=lambda i: (i, 0)),
)(x, weight, bias).block_until_ready()
