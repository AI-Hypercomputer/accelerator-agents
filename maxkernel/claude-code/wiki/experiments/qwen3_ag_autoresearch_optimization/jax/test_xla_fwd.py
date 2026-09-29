import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))


def f(x, labels, w):
  import tokamax

  return tokamax.linear_softmax_cross_entropy_loss(
      x, labels, w, reduction='sum', implementation='xla'
  )


with mesh:
  # Full scale
  x = jax.device_put(
      jnp.zeros((131072, 4096), dtype=jnp.bfloat16),
      jax.sharding.NamedSharding(mesh, P('fsdp')),
  )
  labels = jax.device_put(
      jnp.zeros((131072,), dtype=jnp.int32),
      jax.sharding.NamedSharding(mesh, P('fsdp')),
  )
  w = jax.device_put(
      jnp.zeros((4096, 151936), dtype=jnp.bfloat16),
      jax.sharding.NamedSharding(mesh, P(None, 'fsdp')),
  )

  print('Compiling xla CE fwd...')
  import time

  t0 = time.time()
  out = jax.jit(f)(x, labels, w)
  jax.block_until_ready(out)
  print('Compiled and run in', time.time() - t0, 'seconds.')
