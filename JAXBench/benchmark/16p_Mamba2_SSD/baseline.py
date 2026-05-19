"""Mamba-2 State Space Duality (SSD) — Dao & Gu.

The SSD layer shows that structured state space models are equivalent to a form
of linear attention with input-dependent (selective) decay. This is the matrix
(parallel) form of Mamba-2's core computation.

Paper: "Transformers are SSMs" (Dao & Gu, 2024)
Mamba-2 is the dominant alternative to standard transformers in 2024-2025.

Config based on Mamba-2-2.7B from the paper.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'mamba2_2_7b_ssd',
    'model': 'Mamba-2-2.7B',
    'operator': 'state_space_duality',
    'batch': 4,
    'seq_len': 4096,
    'num_heads': 64,
    'head_dim': 64,
    'd_state': 128,
    'd_model': 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.key(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    # In Mamba-2 SSD: C maps to Q, B maps to K, x maps to V
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)  # C (output projection)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)  # B (input projection)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)  # x (hidden state)
    # A: input-dependent decay (after log-space parameterization)
    # Initialized negative (stable decay), per-head scalar
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def workload(query, key, value, A_log):
    """Mamba-2 SSD: structured linear attention with selective decay.

    y = (L ⊙ (C B^T)) x
    where L[i,j] = Π_{k=j+1}^{i} a_k for i > j, 1 for i=j, 0 for i<j
    and a_k = exp(A_log_k) is the selective (input-dependent) decay.
    """
    B, H, S, D = query.shape

    # Compute per-position decay: a = sigmoid(A_log) to keep in (0, 1)
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))  # (B, H, S)

    # Build causal mask L with cumulative decay
    # log(a) cumsum then exponentiate: L[i,j] = exp(Σ_{k=j+1}^{i} log(a_k))
    log_a = jnp.log(a + 1e-8)  # (B, H, S)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)  # (B, H, S)

    # L[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j, 0 for i < j
    diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

    # SSD attention: (L ⊙ CB^T) x
    # CB^T: (B, H, S, S) — "attention scores"
    scores = jnp.einsum('bhsd,bhtd->bhst',
                        query.astype(jnp.float32),
                        key.astype(jnp.float32))

    # Apply selective decay mask
    scores = scores * L

    # Normalize
    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

    # Output
    output = jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)
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
    # Scores CB^T: 2*B*H*S*S*D, output: 2*B*H*S*S*D
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
