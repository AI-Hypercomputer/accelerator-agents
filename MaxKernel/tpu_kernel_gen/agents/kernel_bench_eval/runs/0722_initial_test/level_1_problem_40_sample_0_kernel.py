def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """Pallas kernel for Layer Normalization.

  This kernel applies layer normalization to a single slice of the input tensor.
  The normalization is performed across the feature axes using a tiled,
  three-pass approach to manage memory usage.

  Args:
    x_ref: A reference to the input tensor slice.
    scale_ref: A reference to the scale parameter tensor.
    bias_ref: A reference to the bias parameter tensor.
    out_ref: A reference to the output tensor slice for storing the result.
  """
  # Default epsilon from flax.linen.LayerNorm
  epsilon = 1e-6
  N = features * dim1 * dim2

  # Pass 1: Compute mean
  mean_sum = jnp.zeros((), dtype=jnp.float32)
  for f in range(features):
    for d1 in range(dim1):
      block = pl.load(x_ref, (0, f, d1, slice(None)))
      mean_sum += jnp.sum(block)
  mean = mean_sum / N

  # Pass 2: Compute variance
  var_sum = jnp.zeros((), dtype=jnp.float32)
  for f in range(features):
    for d1 in range(dim1):
      block = pl.load(x_ref, (0, f, d1, slice(None)))
      var_sum += jnp.sum((block - mean) ** 2)
  var = var_sum / N

  inv_std = jax.lax.rsqrt(var + epsilon)

  # Pass 3: Normalize, scale, bias, and store
  for f in range(features):
    for d1 in range(dim1):
      x_block = pl.load(x_ref, (0, f, d1, slice(None)))
      scale_block = pl.load(scale_ref, (f, d1, slice(None)))
      bias_block = pl.load(bias_ref, (f, d1, slice(None)))
      result = (x_block - mean) * inv_std * scale_block + bias_block
      pl.store(out_ref, (0, f, d1, slice(None)), result)
