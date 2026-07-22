# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    num_groups = 256

    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb, kc = jax.random.split(rand_key, 3)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    gemm_weight = jax.random.normal(ka, (out_features, in_features), dtype=dtype) * 0.02
    gemm_bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02
    gn_weight = jnp.ones(out_features, dtype=dtype)
    gn_bias = jnp.zeros(out_features, dtype=dtype)
    multiply_weight = jax.random.normal(kc, out_features, dtype=dtype) * 0.02

    dynamic_args = [x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight]
    static_args = [num_groups, out_features]

    return dynamic_args, static_args

# Computation
def computation(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight, num_groups, out_features):
    x = jnp.matmul(x, gemm_weight.T) + gemm_bias
    batch_size = x.shape[0]
    group_size = out_features // num_groups
    x_grouped = x.reshape(batch_size, num_groups, group_size)
    mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
    var = jnp.var(x_grouped, axis=-1, keepdims=True)
    x_normalized = (x_grouped - mean) / jnp.sqrt(var + 1e-5)
    x = x_normalized.reshape(batch_size, out_features)
    x = x * gn_weight + gn_bias
    x = x * jax.nn.sigmoid(x)
    x = x * multiply_weight
    x = x * jax.nn.sigmoid(x)
    return x