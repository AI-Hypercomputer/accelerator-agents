"""
TARGETED JAX PATTERN: Batch-wise Cosine Similarity

When the PyTorch source uses F.cosine_similarity on 2D tensors, it computes
per-sample (row-wise) similarity. The JAX conversion MUST preserve this
batch-wise semantics. Do NOT use a library function that computes a single
global similarity scalar over the entire tensor.

## WRONG: Using optax.cosine_similarity (global, not per-sample)

    # PyTorch source:
    #   corr = F.cosine_similarity(
    #       expert_outputs[i].flatten(1),
    #       expert_outputs[j].flatten(1)
    #   ).mean()
    #
    # F.cosine_similarity with 2D input [B, D] returns a per-sample
    # similarity vector of shape [B], then .mean() averages over samples.

    # WRONG! optax.cosine_similarity computes a single scalar over the
    # entire tensor, not per-sample similarity.
    sim = optax.cosine_similarity(
        outputs[i].reshape(outputs[i].shape[0], -1),
        outputs[j].reshape(outputs[j].shape[0], -1)
    )
    return jnp.mean(sim)

## CORRECT: Per-sample cosine similarity with manual computation

    # CORRECT: Compute cosine similarity per sample (row), then average.
    def _cosine_similarity(a, b):
        '''Per-sample cosine similarity for 2D arrays [B, D] -> [B].'''
        a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return jnp.sum(a_norm * b_norm, axis=-1)

    sim = _cosine_similarity(
        outputs[i].reshape(outputs[i].shape[0], -1),
        outputs[j].reshape(outputs[j].shape[0], -1)
    )
    return jnp.mean(sim)

## CORRECT (alternative): Using jax.vmap over single-vector cosine similarity

    def _single_cosine_sim(a, b):
        '''Cosine similarity for 1D vectors.'''
        return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

    batch_cosine_sim = jax.vmap(_single_cosine_sim)
    sim = batch_cosine_sim(
        outputs[i].reshape(outputs[i].shape[0], -1),
        outputs[j].reshape(outputs[j].shape[0], -1)
    )
    return jnp.mean(sim)

## WRONG: Using einsum that sums over both batch AND feature dimensions

    # If you stack expert outputs into shape [num_experts, batch_size, features]
    # and normalize, you might be tempted to use a single einsum:

    outputs_stacked = jnp.stack([out.reshape(out.shape[0], -1) for out in expert_outputs])
    norms = jnp.linalg.norm(outputs_stacked, axis=2, keepdims=True)
    outputs_norm = outputs_stacked / (norms + 1e-8)

    # WRONG! This sums over BOTH batch (k) and feature (d) dimensions,
    # producing sum_k(sum_d(a[i,k,d] * b[j,k,d])) -- a single scalar per
    # expert pair that conflates batch and feature reductions.
    correlations = jnp.einsum('ikd,jkd->ij', outputs_norm, outputs_norm)

    # The result is NOT the mean of per-sample cosine similarities.
    # It equals batch_size * mean(per_sample_cos_sim) only when all samples
    # have equal norms, and even then the scaling is wrong.

## CORRECT: Using einsum with separate batch and feature reductions

    outputs_stacked = jnp.stack([out.reshape(out.shape[0], -1) for out in expert_outputs])
    norms = jnp.linalg.norm(outputs_stacked, axis=2, keepdims=True)
    outputs_norm = outputs_stacked / (norms + 1e-8)

    # CORRECT: First compute per-sample dot products with einsum over
    # features only (d), keeping the batch dimension (b):
    #   per_sample_sim[i, j, b] = sum_d(a[i,b,d] * b[j,b,d])
    per_sample_sim = jnp.einsum('ibd,jbd->ijb', outputs_norm, outputs_norm)

    # Then average over the batch dimension to get mean cosine similarity:
    correlations = per_sample_sim.mean(axis=2)

    # This matches F.cosine_similarity(...).mean() exactly:
    # for each expert pair (i,j), compute per-sample cosine sim, then average.

## WHY this matters:

1. **Semantic difference**: F.cosine_similarity(a, b) with a=[B,D], b=[B,D]
   returns shape [B] -- one similarity per sample. A global cosine similarity
   returns a single scalar, which conflates all samples into one value.
2. **Numerical difference**: mean(per_sample_cosine_sim) != global_cosine_sim.
   The global version effectively computes similarity between the "average
   direction" of all samples, losing per-sample variation.
3. **Metric correctness**: expert_correlation is a diagnostic metric. Wrong
   computation means misleading expert diversity analysis.
4. **General rule**: When the PyTorch source applies a pairwise operation
   along dim=0 (batch dimension) and then reduces, preserve the per-sample
   computation in JAX. Do not replace it with a global reduction.
"""
