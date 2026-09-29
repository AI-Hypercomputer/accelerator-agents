import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))
print('Devices:', jax.devices())


def f(x, labels, w):
  import tokamax

  return tokamax.linear_softmax_cross_entropy_loss(
      x, labels, w, reduction='sum', implementation='chunked_xla'
  )


@jax.jit
def g(x, labels, w):
  return jax.grad(f)(x, labels, w)


with mesh:
  # Full scale with chunk size 2048, checking if that helps compile time
  import tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla as chunked_xla

  chunked_xla.DEFAULT_B_BLOCK_SZ = 2048
  chunked_xla.DEFAULT_V_BLOCK_SZ = 2048

  x = jax.device_put(
      jnp.zeros((131072, 4096), dtype=jnp.bfloat16),
      jax.sharding.NamedSharding(mesh, P('fsdp')),
  )
  labels = jax.device_put(
      jnp.zeros((131072,), dtype=jnp.int32),
      jax.sharding.NamedSharding(mesh, P('fsdp')),
  )
  # Real vocab size is 151936
  w = jax.device_put(
      jnp.zeros((4096, 151936), dtype=jnp.bfloat16),
      jax.sharding.NamedSharding(mesh, P(None, 'fsdp')),
  )

  import time

  t0 = time.time()
  try:
    grad = g(x, labels, w)
    jax.block_until_ready(grad)
    print('Compiled and run in', time.time() - t0, 'seconds.')
  except Exception as e:
    print('Exception:', e)
