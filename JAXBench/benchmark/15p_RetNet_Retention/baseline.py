"""Multi-Scale Retention — Microsoft RetNet.

Replaces softmax attention with retention: a causal linear attention mechanism
with fixed exponential decay per head. Different heads use different decay rates
(multi-scale), giving each head a different "memory horizon".

Paper: "Retentive Network: A Successor to Transformer" (Sun et al., 2023)
Used in RetNet models and influenced Mamba-2, GLA, and other recent architectures.

Config based on RetNet-6.7B from the paper.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'retnet_6_7b_retention',
    'model': 'RetNet-6.7B',
    'operator': 'multi_scale_retention',
    'batch': 4,
    'seq_len': 4096,
    'num_heads': 16,
    'head_dim': 256,
    'd_model': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """Multi-scale retention with per-head exponential decay.

    Retention(X) = (Q K^T ⊙ D) V
    where D[i,j] = γ^(i-j) if i >= j, else 0

    Each head has a different decay rate γ_h, creating a multi-scale
    representation: some heads attend locally, others globally.
    """
    B, H, S, D = query.shape

    # Multi-scale decay rates (from RetNet paper)
    # γ_h = 1 - 2^(-5 - arange(H))
    gammas = 1.0 - jnp.exp2(-5.0 - jnp.arange(H, dtype=jnp.float32))  # (H,)
    # Gammas range from ~0.97 (long range) to ~1.0 (very long range)

    # Build causal decay matrix D[i,j] = γ^(i-j) for i >= j
    positions = jnp.arange(S, dtype=jnp.float32)
    # distance[i,j] = i - j
    distance = positions[:, None] - positions[None, :]  # (S, S)
    # D[h,i,j] = γ_h^(i-j) * (i >= j)
    causal_mask = (distance >= 0).astype(jnp.float32)
    # γ^distance: (H, S, S)
    log_gamma = jnp.log(gammas)  # (H,)
    decay = jnp.exp(log_gamma[:, None, None] * distance[None, :, :])  # (H, S, S)
    decay = decay * causal_mask[None, :, :]  # apply causal mask

    # Retention: (Q K^T ⊙ D) V
    # QK^T: (B, H, S, S)
    qk = jnp.einsum('bhsd,bhtd->bhst', query.astype(jnp.float32), key.astype(jnp.float32))

    # Apply decay mask
    qk = qk * decay[None, :, :, :]  # (B, H, S, S)

    # Normalize by retention sum (per-query normalization)
    retention_sum = jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
    retention_sum = jnp.maximum(retention_sum, 1.0)
    qk = qk / retention_sum

    # Output
    output = jnp.einsum('bhst,bhtd->bhsd', qk.astype(query.dtype), value)
    return output


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    # QK^T: 2*B*H*S*S*D, AV: 2*B*H*S*S*D
    flops = 2 * B * H * S * S * D * 2
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
