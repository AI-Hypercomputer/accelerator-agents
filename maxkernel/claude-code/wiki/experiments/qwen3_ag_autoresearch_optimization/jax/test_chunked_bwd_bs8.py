import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))


def my_f_bwd(g, lse, x, labels, w):
  from tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla import linear_softmax_cross_entropy_loss_bwd_chunked_xla

  dx, dw = linear_softmax_cross_entropy_loss_bwd_chunked_xla(
      g, lse, x, labels, w, reduction='sum', b_block_sz=1024, v_block_sz=1024
  )
  return dx, None, dw


with mesh:
  # Scale down by 4, this should fit easily in VMEM since 1/4 the sizes
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
  g = jax.device_put(
      jnp.zeros((), dtype=jnp.bfloat16), jax.sharding.NamedSharding(mesh, P())
  )
  lse = jax.device_put(
      jnp.zeros((131072,), dtype=jnp.bfloat16),
      jax.sharding.NamedSharding(mesh, P('fsdp')),
  )

  print('Compiling test bwd ce...')
  t0 = time.time()
  try:
    out = jax.jit(my_f_bwd)(g, lse, x, labels, w)
    jax.block_until_ready(out)
    print('Compiled and run in', time.time() - t0, 'seconds.')
  except Exception as e:
    print('Exception:', e)
