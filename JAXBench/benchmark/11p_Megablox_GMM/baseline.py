"""Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Reference grouped matmul: for each expert group, slice the input tokens
and multiply with that expert's weight matrix. Core primitive for MoE layers.
From JAX experimental pallas ops (reference_gmm).

"""

import jax
import jax.numpy as jnp

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


def create_inputs(dtype=jnp.bfloat16):
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
    return lhs, rhs, group_sizes, max_expert_size


def workload(lhs, rhs, group_sizes, max_expert_size):
    """Jittable grouped matmul using static shapes and masking.

    Computes dot product for each group with static slice sizes to allow JIT.
    """
    G = rhs.shape[0]
    M, K = lhs.shape
    N = rhs.shape[2]

    # Compute expert offsets
    group_ends = jnp.cumsum(group_sizes)
    group_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
    )

    # Initialize flat result array with padding
    res_flat = jnp.zeros((M + max_expert_size, N), dtype=lhs.dtype)

    def body_fun(carry_res_flat, i):
        start = group_starts[i]
        count = group_sizes[i]

        # Slice with a STATIC size
        expert_lhs = jax.lax.dynamic_slice(
            lhs, (start, 0), (max_expert_size, K)
        )
        expert_rhs = rhs[i, :, :]

        # Compute GEMM
        res = jax.lax.dot(
            expert_lhs, expert_rhs, preferred_element_type=jnp.float32
        )

        # Mask out invalid rows
        mask = (
            jax.lax.broadcasted_iota(jnp.int32, (max_expert_size, N), 0) < count
        )
        res_masked = jnp.where(mask, res, 0.0)

        # Read-Modify-Write to accumulate results
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


def get_flops():
    """Total FLOPs: each expert does (M/G) x K x N matmul."""
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    return 2 * M * K * N


def benchmark(num_warmup=2, num_iters=10):
    """Benchmark with JIT."""
    import time
    inputs = create_inputs()
    
    fn = jax.jit(workload, static_argnums=(3,))
    
    # Warmup
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    flops = 2 * M * K * N
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
