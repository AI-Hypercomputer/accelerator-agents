# Imports
import jax
import jax.numpy as jnp
import time
import numpy as np
import json

# Initialization
def get_inputs(dtype=jnp.bfloat16):
    M = 8192
    K = 8192
    N = 28672
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    dynamic_args = [A, B]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(A, B):
    return jnp.dot(A, B)