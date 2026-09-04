"""52_Conv2d_Activation_BatchNorm — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp
import jax.lax as lax

CONFIG = {
    'name': '52_Conv2d_Activation_BatchNorm',
    'batch_size': 64,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size, in_channels, out_channels, kernel_size = 64, 64, 128, 3
    height, width = 128, 128
    x = jax.random.uniform(k1, (batch_size, in_channels, height, width), dtype=dtype)
    conv_weight = jax.random.normal(k2, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
    conv_bias = jax.random.normal(k3, (out_channels,), dtype=dtype)
    bn_weight = jnp.ones(out_channels, dtype=dtype)
    bn_bias = jnp.zeros(out_channels, dtype=dtype)
    return x, conv_weight, conv_bias, bn_weight, bn_bias


def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    """Conv2d + Mish activation + BatchNorm."""
    eps = 1e-5
    # Conv2d NCHW -> NHWC
    x = jnp.transpose(x, (0, 2, 3, 1))
    weight = jnp.transpose(conv_weight, (2, 3, 1, 0))
    x = lax.conv_general_dilated(
        x, weight,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + conv_bias.reshape(1, 1, 1, -1)
    x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

    # Mish activation: multiply(tanh(softplus(x)), x)
    softplus_x = jax.nn.softplus(x)
    x = jnp.multiply(jnp.tanh(softplus_x), x)

    # BatchNorm2d (training mode)
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    w = bn_weight.reshape(1, -1, 1, 1)
    b = bn_bias.reshape(1, -1, 1, 1)
    x = (x - mean) / jnp.sqrt(var + eps) * w + b
    return x

def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times_ms = np.array(times) * 1000
    avg = float(np.mean(times_ms))
    return {
        'name': CONFIG['name'],
        'config': {k: v for k, v in CONFIG.items() if k != 'name'},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times_ms)), 4),
        'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
