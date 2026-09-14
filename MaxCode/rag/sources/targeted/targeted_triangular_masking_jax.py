"""
TARGETED JAX PATTERN: Triangular Masking for Causal Attention

For standard attention scores before softmax, use ADDITIVE masking with large negative
values, NOT multiplicative boolean masks. Multiplicative masks cause issues with
softmax (masked positions become 0 instead of being suppressed to near-zero probability).

## WRONG: Multiplicative boolean mask (DO NOT DO THIS):

    # WRONG! After softmax, masked positions get non-zero probability
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    attn_weights = attn_scores * causal_mask  # Zeros out future positions
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    # Problem: softmax(0) != 0, so masked positions still get some probability!

## CORRECT: Additive float mask with large negative value:

    import jax
    import jax.numpy as jnp

    def make_causal_mask(seq_len, dtype=jnp.float32):
        '''
        Create additive causal mask.

        Returns:
            mask: [seq_len, seq_len] where allowed=0.0, blocked=-1e9
        '''
        # Lower-triangular inclusive (k=0): position i can attend to j where j <= i
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=0)
        mask = jnp.where(causal, 0.0, -1e9)
        return mask.astype(dtype)

    # Usage:
    attn_scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(head_dim)
    mask = make_causal_mask(seq_len, dtype=attn_scores.dtype)
    attn_scores = attn_scores + mask  # Add mask BEFORE softmax
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

## Key functions:

    # Lower triangular inclusive (causal: attend to self and past)
    jnp.tril(jnp.ones((n, n)), k=0)
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]

    # Strict lower triangular (attend to past only, NOT self)
    jnp.tril(jnp.ones((n, n)), k=-1)
    # [[0, 0, 0],
    #  [1, 0, 0],
    #  [1, 1, 0]]

    # Strict upper triangular (what to BLOCK in causal attention)
    jnp.triu(jnp.ones((n, n)), k=1)
    # [[0, 1, 1],
    #  [0, 0, 1],
    #  [0, 0, 0]]

## For chunk-parallel attention (within-chunk causal mask):

    def make_chunk_causal_mask(chunk_size, dtype=jnp.float32):
        '''Causal mask for within-chunk attention.'''
        causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=0)
        return jnp.where(causal, 0.0, -1e9).astype(dtype)

    # For decay-based masking (gated delta rule):
    # The decay mask is multiplicative but applied to attention weights
    # BEFORE adding to the accumulator, not to raw scores before softmax.
    # This is different from standard attention masking.

    def make_decay_mask(log_decay, chunk_size):
        '''
        Create exponential decay mask for linear attention within a chunk.

        Args:
            log_decay: [batch, heads, chunk_size] log-decay values per timestep

        Returns:
            decay_mask: [batch, heads, chunk_size, chunk_size] where
                        mask[i,j] = exp(sum(log_decay[j+1:i+1])) for j <= i, 0 otherwise
        '''
        # Cumulative sum of log-decay gives log of product of decays
        cumsum = jnp.cumsum(log_decay, axis=-1)

        # decay_mask[i,j] = exp(cumsum[i] - cumsum[j])
        mask = jnp.exp(cumsum[..., :, None] - cumsum[..., None, :])

        # Zero out upper triangle (future positions)
        causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=0)
        return jnp.where(causal, mask, 0.0)

## Combining causal mask with padding mask:

    def make_combined_mask(seq_len, padding_lengths, dtype=jnp.float32):
        '''
        Combine causal mask with padding mask.

        Args:
            seq_len: sequence length
            padding_lengths: [batch] number of padding tokens at start

        Returns:
            mask: [batch, 1, seq_len, seq_len] broadcastable over heads
        '''
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=0)

        # Padding mask: True where position is valid (not padding)
        positions = jnp.arange(seq_len)
        valid = positions[None, :] >= padding_lengths[:, None]  # [batch, seq_len]

        # Combine: attend only to valid, causal positions
        combined = causal[None, :, :] & valid[:, None, :]  # [batch, seq_len, seq_len]
        mask = jnp.where(combined, 0.0, -1e9).astype(dtype)
        return mask[:, None, :, :]  # [batch, 1, seq_len, seq_len] for head broadcast

## Why additive masking:

1. **Correct softmax behavior**: Adding -1e9 before softmax makes masked positions
   have exp(-1e9) ~ 0 probability. Multiplying by 0 after scores but before
   softmax doesn't suppress probability correctly.
2. **Gradient flow**: Additive mask has clean gradients. Multiplicative mask
   creates 0 * gradient = 0 issues.
3. **JAX convention**: JAX/Flax examples universally use additive masking.
"""
