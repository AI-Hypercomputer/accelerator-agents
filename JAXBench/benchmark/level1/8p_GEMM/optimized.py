# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas matmul TPU kernel — Llama-3.1-70B FFN dimensions.

Upstream kernel from jax.experimental.pallas.ops.tpu.matmul, wrapped as a
JAXBench workload with CONFIG / create_inputs / workload.

See discussion in https://docs.jax.dev/en/latest/pallas/tpu/matmul.html.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

CONFIG = {
    'name': 'pallas_matmul_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
    'atol': 1e-3,
    'rtol': 1e-2,
}

# Tuned by autotune_block_sizes.py. Re-run to update.
TUNED_PARAMS = {'block_shape': [1024, 2048], 'block_k': 1024}


def matmul_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref):
  @pl.when(pl.program_id(2) == 0)
  def init():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] = acc_ref[...] + jnp.dot(
      x_tile_ref[...],
      y_tile_ref[...],
      preferred_element_type=acc_ref.dtype,
  )
  # It is possible to make this conditional but in general this bundle packs
  # quite well for a simple matmul kernel
  o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=["block_shape", "block_k", "debug", "out_dtype"]
)
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape,
    block_k: int = 256,
    out_dtype: jnp.dtype | None = None,
    debug: bool = False,
) -> jax.Array:
  if out_dtype is None:
    if x.dtype != y.dtype:
      # TODO(tlongeri): Maybe we could use a deduction similar to jnp.dot
      raise TypeError(
          f"Cannot deduce output dtype for different input dtypes: {x.dtype},"
          f" {y.dtype}"
      )
    out_dtype = x.dtype
  acc_dtype = jnp.float32
  if x.dtype in [jnp.int8, jnp.int4, jnp.uint8, jnp.uint4]:
    acc_dtype = jnp.int32

  l, r = block_shape
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((l, block_k), lambda i, _, k: (i, k)),
              pl.BlockSpec((block_k, r), lambda _, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((l, r), lambda i, j, k: (i, j)),
          grid=(x.shape[0] // l, y.shape[1] // r, x.shape[1] // block_k),
          scratch_shapes=[pltpu.VMEM((l, r), acc_dtype)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
      debug=debug,
  )(x, y)


def get_flops():
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    return 2 * M * K * N


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    y = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return x, y


def workload(x, y):
    return matmul(x, y, block_shape=tuple(TUNED_PARAMS['block_shape']), block_k=TUNED_PARAMS['block_k'])


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
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator', 'atol', 'rtol')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
