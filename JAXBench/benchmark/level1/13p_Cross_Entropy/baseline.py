"""Fused Linear + Cross-Entropy Loss — Llama 3.1 8B. From openxla/tokamax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_8b_cross_entropy',
    'model': 'Llama-3.1-8B',
    'operator': 'fused_cross_entropy',
    'batch_tokens': 8192,
    'hidden_dim': 4096,
    'vocab_size': 128256,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (hidden_states, lm_head_weight, labels)."""
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG['vocab_size']
    hidden = jax.random.normal(k1, (B, H), dtype=dtype)
    weight = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
    labels = jax.random.randint(k3, (B,), 0, V)
    return hidden, weight, labels


def workload(hidden, weight, labels):
    """Fused linear projection + softmax cross-entropy loss."""
    logits = jnp.dot(hidden, weight)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


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
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG['vocab_size']
    flops = B * H * V * 2 + B * V * 3  # matmul + softmax + loss
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
