def kernel(predictions_ref, targets_ref, out_ref):
  """Pallas kernel to compute Sum of Squared Errors.

  This kernel computes the sum of squared errors for a slice of the input
  and adds its contribution to the total sum, which is stored in out_ref.
  The reduction over all slices is handled by Pallas's atomic operations.
  The final division to compute the mean must be handled by the caller.
  """
  # Each program instance computes the sum of squared errors for its assigned row.
  sum_sq_err = jnp.sum((predictions_ref[...] - targets_ref[...]) ** 2)

  # Atomically add this program's contribution to the total sum.
  pl.atomic_add(out_ref, 0, sum_sq_err)
