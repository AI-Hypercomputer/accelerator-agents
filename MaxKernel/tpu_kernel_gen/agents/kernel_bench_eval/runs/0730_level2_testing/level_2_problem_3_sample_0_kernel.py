# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import tree_util
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
sum_weight_val = 1.0
pool_kernel_size = (2, 2, 2)

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax expect channel-last format: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))


class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    # PyTorch's padding and output_padding combination results in an
    # output shape that is equivalent to Flax's 'SAME' padding for this configuration.
    x = nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding="SAME")(x)
    sum_weight = self.param("sum_weight", nn.initializers.constant(sum_weight_val), ())
    x = x + sum_weight
    # PyTorch's LayerNorm was applied over (C, D, H, W). In Flax's channel-last
    # format (N, D, H, W, C), this corresponds to normalizing over all axes
    # except the batch axis (axis 0).
    x = nn.LayerNorm(reduction_axes=(1, 2, 3, 4))(x)
    x = nn.avg_pool(x, window_shape=pool_kernel_size, strides=pool_kernel_size)
    x = nn.gelu(x)
    return x


model = Model()
params = model.init(key_params, x)["params"].unfreeze()


# Computation
def kernel(x_ref, params_ref, out_ref):
  # Load the parameters from HBM into SRAM.
  params = tree_util.tree_map(lambda x: x[...], params_ref)
  # In a Pallas kernel, we operate on a single batch item at a time.
  # The input `x_ref` is a slice of the original batch, with shape (1, D, H, W, C_in).
  # We squeeze the batch dimension to work with shapes expected by Flax/JAX layers.
  x = jnp.squeeze(x_ref[...], axis=0)

  # The model's parameters are passed in as a PyTree.
  # We define a temporary model instance to apply the layers with the given parameters.
  # This is a convenient way to replicate the logic without re-implementing each layer from scratch.
  class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      # Replicate the nn.ConvTranspose layer
      x = nn.ConvTranspose(
        features=out_ref.shape[-1],  # out_channels
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="SAME",
        name="ConvTranspose_0",
      )(x)
      # Replicate the addition of the 'sum_weight' parameter
      # The parameter is a scalar, so we access it directly.
      sum_weight = self.param("sum_weight", nn.initializers.zeros, ())
      x = x + sum_weight
      # Replicate the nn.LayerNorm layer
      # The reduction is over all spatial and channel dimensions.
      x = nn.LayerNorm(reduction_axes=(0, 1, 2, 3), name="LayerNorm_0")(x)
      # Replicate the nn.avg_pool layer
      x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
      # Replicate the nn.gelu activation function
      x = nn.gelu(x)
      return x

  # Apply the model logic using the provided parameters.
  # The `bind` method creates a new model instance with the parameters applied.
  output = Model().apply({"params": params}, x)

  # The output has shape (D, H, W, C_out). We need to add the batch dimension
  # back to match the expected output shape of the kernel (1, D, H, W, C_out).
  out_ref[...] = jnp.expand_dims(output, axis=0)


# The grid will iterate over the batch dimension.
grid = (batch_size,)

# The output shape after the model is applied.
# ConvTranspose with stride=2 and padding='SAME' doubles the spatial dimensions.
# AvgPool with window_shape=2 and stride=2 halves them again.
# The number of channels changes from in_channels to out_channels.
# Initial shape: (128, 16, 32, 32, 32)
# After ConvT: (128, 32, 64, 64, 64)
# After AvgPool: (128, 16, 32, 32, 64)
output_shape = (batch_size, depth, height, width, out_channels)

# The spec for the params PyTree should have the same structure, but with None at the leaves.
params_spec = tree_util.tree_map(lambda _: None, params)

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    # For the input tensor 'x', we process one batch item per kernel instance.
    # The index_map `lambda i: (i, 0, 0, 0, 0)` selects the i-th slice along the batch axis.
    pl.BlockSpec(block_shape=(1, *x.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
    # The parameters are a PyTree. We treat them as closed-over values,
    # so they are not chunked and are passed to every kernel instance.
    # This is achieved by providing a spec with a matching PyTree structure of Nones.
    params_spec,
  ],
  # For the output, each kernel instance writes to its corresponding slice in the batch.
  out_specs=pl.BlockSpec(block_shape=(1, *output_shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
)(x, params).block_until_ready()
