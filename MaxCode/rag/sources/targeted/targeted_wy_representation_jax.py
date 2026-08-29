"""
TARGETED JAX PATTERN: WY Representation for Chunk-Parallel Delta Rule

When converting a PyTorch for-loop that computes a Neumann series row-by-row
on a lower-triangular matrix, DO NOT translate it as a jax.lax.scan with
dynamic slicing like attn[..., i, :i]. Dynamic slice sizes are NOT compatible
with jax.jit because JAX requires static shapes at trace time.

INSTEAD, use jax.scipy.linalg.solve_triangular to compute (I - W)^{-1}
directly. This is mathematically equivalent to the Neumann series
I + W + W^2 + ... but is JIT-safe, GPU-parallelizable, and numerically stable.

## The PyTorch Pattern (for-loop, do NOT copy directly):

    # PyTorch: row-by-row Neumann series (CANNOT run under jax.jit)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + \\
            (attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2)
    attn = attn + torch.eye(chunk_size)

## The Correct JAX Pattern (solve_triangular):

    import jax
    import jax.numpy as jnp

    # raw_attn is strictly lower triangular: -(k_beta @ key^T) * decay_mask
    # with upper triangle and diagonal zeroed out
    upper_mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
    raw_attn = -(k_beta @ jnp.transpose(key, (0, 1, 2, 4, 3))) * decay_mask
    raw_attn = jnp.where(upper_mask, 0.0, raw_attn)

    # Compute (I - W)^{-1} using solve_triangular
    # This solves (I - W) @ X = I, giving X = (I - W)^{-1}
    eye = jnp.eye(chunk_size)
    attn = jax.scipy.linalg.solve_triangular(
        eye - raw_attn,   # unit lower triangular matrix
        eye,              # solve for identity -> gives the inverse
        lower=True,       # it's lower triangular
    )

    # Then apply the WY transform:
    value_corrected = attn @ v_beta
    k_cumdecay = attn @ (k_beta * jnp.exp(g_cumsum)[..., None])

## Why solve_triangular works:

The for-loop computes the Neumann series I + W + W^2 + ... which equals
(I - W)^{-1} for strictly lower triangular W. solve_triangular computes
this directly via back-substitution, which is:
- O(n^2) per row, same complexity as the for-loop
- JIT-compatible (no dynamic shapes)
- GPU-parallelizable (LAPACK/cuSOLVER backend)
- Numerically stable

## Inter-chunk scan pattern:

After computing the WY correction within each chunk, use jax.lax.scan
across chunks to accumulate the recurrent state:

    def chunk_scan_fn(S_prev, chunk_inputs):
        q_c, k_c, v_c, k_cumdec_c, g_c, decay_c = chunk_inputs

        # Intra-chunk attention
        intra_attn = (q_c @ jnp.transpose(k_c, (0, 1, 3, 2))) * decay_c
        intra_attn = jnp.where(upper_mask_strict, 0.0, intra_attn)

        # Inter-chunk: project through accumulated state
        v_prime = k_cumdec_c @ S_prev
        v_new = v_c - v_prime
        attn_inter = (q_c * jnp.exp(g_c)[..., None]) @ S_prev

        # Combine
        out_c = attn_inter + intra_attn @ v_new

        # Update state
        g_last = g_c[..., -1, None, None]
        k_weighted = k_c * jnp.exp(g_c[..., -1:] - g_c)[..., None]
        S_next = S_prev * jnp.exp(g_last) + jnp.transpose(k_weighted, (0, 1, 3, 2)) @ v_new

        return S_next, out_c

    final_state, core_attn_out = jax.lax.scan(chunk_scan_fn, init_S, scan_inputs)
"""
