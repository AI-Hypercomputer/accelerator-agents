def kernel(predictions_ref, targets_ref, out_ref):
  """
  Computes the KL divergence mean in a single Pallas kernel.

  The overall computation is:
  jnp.sum(targets * (jnp.log(targets) - jnp.log(predictions)), axis=-1).mean()
  which is mathematically equivalent to:
  jnp.sum(targets * (jnp.log(targets) - jnp.log(predictions))) / batch_size

  This kernel computes the sum for a local block of data and atomically adds
  its contribution to the final mean.
  """
  # Each kernel instance calculates the sum for its corresponding block.
  local_kl_sum = jnp.sum(targets_ref[...] * (jnp.log(targets_ref[...]) - jnp.log(predictions_ref[...])))

  # To compute the mean, we need the total batch size. We can derive this
  # from the grid and block dimensions.
  # pl.num_programs(axis=0) gives the number of blocks in the batch dimension.
  # predictions_ref.shape[0] gives the batch size of a single block.
  total_batch_size = pl.num_programs(axis=0) * predictions_ref.shape[0]

  # Each block's contribution to the final mean is its local sum divided
  # by the total batch size. We perform an atomic add to accumulate these
  # contributions from all blocks into the scalar output.
  # The output buffer `out_ref` is automatically initialized to zero by JAX.
  pl.atomic_add(out_ref, 0, local_kl_sum / total_batch_size)
