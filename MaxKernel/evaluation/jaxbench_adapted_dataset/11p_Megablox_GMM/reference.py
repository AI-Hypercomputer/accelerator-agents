# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.bfloat16):
    CONFIG = {
        'name': 'megablox_gmm_qwen3_235b',
        'model': 'Qwen3-235B-A22B',
        'operator': 'grouped_matmul',
        'num_experts': 128,
        'num_experts_per_tok': 8,
        'emb_dim': 4096,
        'moe_mlp_dim': 1536,
        'seq_len': 4096,
    }
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    limit = 1 / (M * K)
    lhs = jax.random.uniform(k1, (M, K), dtype=dtype, minval=-limit, maxval=limit)
    lhs = lhs.astype(jnp.bfloat16).astype(dtype)
    rhs = jax.random.uniform(k2, (G, K, N), dtype=dtype, minval=-limit, maxval=limit)
    rhs = rhs.astype(jnp.bfloat16).astype(dtype)
    max_expert_size = M // G
    group_sizes = jnp.full((G,), max_expert_size, dtype=jnp.int32)
    
    dynamic_args = [lhs, rhs, group_sizes]
    static_args = [max_expert_size]
    
    return dynamic_args, static_args

# Computation
def computation(lhs, rhs, group_sizes, max_expert_size):
    G = rhs.shape[0]
    M, K = lhs.shape
    N = rhs.shape[2]

    group_ends = jnp.cumsum(group_sizes)
    group_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
    )

    res_flat = jnp.zeros((M + max_expert_size, N), dtype=lhs.dtype)

    def body_fun(carry_res_flat, i):
        start = group_starts[i]
        count = group_sizes[i]

        expert_lhs = jax.lax.dynamic_slice(
            lhs, (start, 0), (max_expert_size, K)
        )
        expert_rhs = rhs[i, :, :]

        res = jax.lax.dot(
            expert_lhs, expert_rhs, preferred_element_type=jnp.float32
        )

        mask = (
            jax.lax.broadcasted_iota(jnp.int32, (max_expert_size, N), 0) < count
        )
        res_masked = jnp.where(mask, res, 0.0)

        current_slice = jax.lax.dynamic_slice(
            carry_res_flat, (start, 0), (max_expert_size, N)
        )
        updated_slice = current_slice + res_masked.astype(carry_res_flat.dtype)
        carry_res_flat = jax.lax.dynamic_update_slice(
            carry_res_flat, updated_slice, (start, 0)
        )

        return carry_res_flat, None

    res_flat, _ = jax.lax.scan(body_fun, res_flat, jnp.arange(G))

    return res_flat[:M, :]