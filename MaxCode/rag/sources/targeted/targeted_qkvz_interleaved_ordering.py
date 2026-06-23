"""
TARGETED JAX PATTERN: Interleaved QKVZ Weight Ordering (fix_query_key_value_ordering)

CRITICAL: When converting models where num_key_heads != num_value_heads,
the projection weights are stored in an INTERLEAVED order grouped by key heads.
You MUST NOT use a flat split on the concatenated projection output.

## The Problem:

If num_k_heads = 4 and num_v_heads = 8 (i.e., v_per_k = 2), the QKVZ
projection output is NOT laid out as [all_Q, all_K, all_V, all_Z].

Instead, it is grouped by key heads:
    [key_head_0_Q, key_head_0_K, key_head_0_V0, key_head_0_V1, key_head_0_Z0, key_head_0_Z1,
     key_head_1_Q, key_head_1_K, key_head_1_V0, key_head_1_V1, key_head_1_Z0, key_head_1_Z1,
     ...]

## WRONG approach (flat split -- DO NOT DO THIS):

    # WRONG! This assumes Q, K, V, Z are contiguous blocks
    q, k, v, z = jnp.split(proj_qkvz, [key_dim, key_dim*2, key_dim*2+value_dim], axis=-1)

## CORRECT approach (group by key heads, then split within each group):

    def fix_query_key_value_ordering(mixed_qkvz, mixed_ba, batch_size, seq_len,
                                      num_k_heads, num_v_heads, head_k_dim, head_v_dim):
        v_per_k = num_v_heads // num_k_heads

        # Step 1: Reshape to [B, T, num_k_heads, per_head_size]
        per_head_size = 2 * head_k_dim + 2 * v_per_k * head_v_dim
        qkvz = mixed_qkvz.reshape(batch_size, seq_len, num_k_heads, per_head_size)

        # Step 2: Split within each key-head group
        split_points = [head_k_dim, 2 * head_k_dim, 2 * head_k_dim + v_per_k * head_v_dim]
        q, k, v, z = jnp.split(qkvz, split_points, axis=-1)
        # q: [B, T, num_k_heads, head_k_dim]
        # k: [B, T, num_k_heads, head_k_dim]
        # v: [B, T, num_k_heads, v_per_k * head_v_dim]
        # z: [B, T, num_k_heads, v_per_k * head_v_dim]

        # Step 3: Reshape v, z to per-value-head
        v = v.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
        z = z.reshape(batch_size, seq_len, num_v_heads, head_v_dim)

        # Same for BA projection:
        ba_per_head = 2 * v_per_k
        ba = mixed_ba.reshape(batch_size, seq_len, num_k_heads, ba_per_head)
        b, a = jnp.split(ba, 2, axis=-1)
        b = b.reshape(batch_size, seq_len, num_v_heads)
        a = a.reshape(batch_size, seq_len, num_v_heads)

        return q, k, v, z, b, a

## Why this matters:

With num_k_heads=4 and num_v_heads=8, a flat split would assign the wrong
dimensions to Q, K, V, Z because the weights are interleaved per key-head group.
The model will produce completely wrong outputs if this ordering is not preserved.

This pattern appears in Qwen3-Next's GatedDeltaNet and similar models with
grouped key-value heads in linear attention layers.
"""
