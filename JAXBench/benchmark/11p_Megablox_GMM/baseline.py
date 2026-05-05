"""Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Reference grouped matmul: for each expert group, slice the input tokens
and multiply with that expert's weight matrix. Core primitive for MoE layers.
From JAX experimental pallas ops (reference_gmm).

Not jit-compatible: uses data-dependent slicing on group_sizes.
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

_skip_jit = True


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
    tokens_per_expert = M // G
    group_sizes = jnp.full((G,), tokens_per_expert, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def workload(lhs, rhs, group_sizes):
    """Reference grouped matmul from upstream JAX tests.

    For each group i, slices lhs[start:start+size] and computes dot with rhs[i].
    Uses data-dependent slicing so must be run eagerly (not under jax.jit).
    """
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
        result = jax.lax.dot(
            lhs[start:start + size, :],
            rhs[i, :, :],
            preferred_element_type=jnp.float32,
        )
        out.append(result)
        start += group_sizes[i]
    return jnp.concatenate(out, axis=0)


def get_flops():
    """Total FLOPs: each expert does (M/G) x K x N matmul."""
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    return 2 * M * K * N


def benchmark(num_warmup=2, num_iters=10):
    """Benchmark eagerly (no JIT — data-dependent control flow)."""
    import time
    inputs = create_inputs()
    # Warmup
    for _ in range(num_warmup):
        out = workload(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = workload(*inputs)
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
