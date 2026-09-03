"""Flex Attention — Llama-3.1-70B with custom score modification.

Flexible attention with arbitrary score_mod function support, the JAX
equivalent of PyTorch's flex_attention. Baseline uses causal mask with
relative position bias as the score modifier.
From MaxText layers/attention_op.py (dot_product_attention with masks).
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_70b_flex_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'flex_attention',
    'batch': 4,
    'seq_len': 4096,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k, v, rel_pos_bias) tensors."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype) * 0.02
    # Relative position bias per head (learned, as in ALiBi-style)
    rel_pos_bias = jax.random.normal(k4, (H, S, S), dtype=dtype) * 0.01
    return q, k, v, rel_pos_bias


def workload(q, k, v, rel_pos_bias):
    """Flex attention: dot-product attention with score modification."""
    D = CONFIG['head_dim']
    S = CONFIG['seq_len']
    sm_scale = D ** -0.5

    # Attention scores
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale

    # Score modification: add relative position bias
    attn = attn + rel_pos_bias[None, :, :, :]

    # Causal mask
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(causal[None, None, :, :], attn, -1e30)

    attn = jax.nn.softmax(attn, axis=-1)

    # Output
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out


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
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    # QK dot: 2*B*H*S*S*D, score_mod add: B*H*S*S, AV dot: 2*B*H*S*S*D
    flops = 4 * B * H * S * S * D + B * H * S * S
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
