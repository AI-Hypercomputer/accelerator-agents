import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))


def f(x, labels, w):
  from tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla import linear_softmax_cross_entropy_loss_fwd_chunked_xla

  loss, lse = linear_softmax_cross_entropy_loss_fwd_chunked_xla(
      x, labels, w, reduction='sum', b_block_sz=1024, v_block_sz=1024
  )
  return loss, lse


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

  print('Compiling chunked_xla CE fwd...')
  import time

  t0 = time.time()
  out = jax.jit(f)(x, labels, w)
  jax.block_until_ready(out)
  print('Compiled and run in', time.time() - t0, 'seconds.')
