"""Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Reference grouped matmul: for each expert group, slice the input tokens
and multiply with that expert's weight matrix. Core primitive for MoE layers.
From JAX experimental pallas ops (reference_gmm).

Jit-compatible: uses static-shape slicing with masking on group_sizes.
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
    # Realistic top-k router with mild expert popularity bias.
    # bias_scale=0.15 produces max/mean ~1.9, CV ~0.30, no empty groups.
    'router_bias_scale': 0.15,
    # max_expert_size is a static jit arg (compile-time upper bound on the
    # largest group). 1024 = 4 * M/G is generous: covers our simulated router
    # (TPU-observed max ~520 at bias_scale=0.15) with ~2x margin, and would
    # still hold up to bias_scale ~0.4 if we ever wanted a more imbalanced
    # workload. Matches how production capacity-factor fast-paths over-size
    # the per-expert tile.
    'max_expert_size': 1024,
    'static_argnums': (3,),  # max_expert_size must be static for jit
}


def _simulate_router(key, S, G, top_k, bias_scale):
    """Simulate top-k token-choice routing with expert popularity bias.

    Returns (assignments, group_sizes) where assignments has shape (S*top_k,)
    and is sorted by expert id, suitable as an Megablox GMM input contract
    (lhs rows pre-grouped by expert).
    """
    k_bias, k_noise = jax.random.split(key, 2)
    expert_bias = jax.random.normal(k_bias, (G,)) * bias_scale
    per_token_noise = jax.random.normal(k_noise, (S, G))
    router_logits = expert_bias[None, :] + per_token_noise
    _, topk_idx = jax.lax.top_k(router_logits, top_k)  # (S, top_k)
    assignments = topk_idx.reshape(-1)  # (S*top_k,)
    sort_perm = jnp.argsort(assignments, stable=True)
    sorted_assignments = assignments[sort_perm]
    group_sizes = jnp.bincount(sorted_assignments, length=G).astype(jnp.int32)
    return sort_perm, group_sizes


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(42)
    k_router, k_lhs, k_rhs = jax.random.split(key, 3)
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    max_expert_size = CONFIG['max_expert_size']
    # Small-normal weights/activations (~0.02 scale): large enough that
    # matmul outputs are bf16-representable but small enough to avoid
    # overflow when accumulated across K=4096. Previous version used
    # `1/(M*K)` as a uniform limit, which underflowed to zero in bf16 and
    # let no-op kernels trivially pass np.allclose against an all-zero
    # reference.
    lhs_unsorted = jax.random.normal(k_lhs, (M, K), dtype=dtype) * 0.02
    rhs = jax.random.normal(k_rhs, (G, K, N), dtype=dtype) * 0.02
    sort_perm, group_sizes = _simulate_router(
        k_router, S, G, top_k, CONFIG['router_bias_scale']
    )
    # Sort lhs rows by expert id so contiguous rows belong to the same expert
    # (Megablox's input contract).
    lhs = lhs_unsorted[sort_perm]
    # Runtime check: assert max group size fits the static upper bound.
    assert int(group_sizes.max()) <= max_expert_size, (
        f"Simulated max group size {int(group_sizes.max())} exceeds "
        f"static max_expert_size={max_expert_size}; raise CONFIG['max_expert_size'] "
        f"or reduce router_bias_scale."
    )
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
