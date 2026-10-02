import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))
print('Devices:', jax.devices())


def f(x, labels, w):
  logits = x @ w
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  return jnp.sum(loss)


@jax.jit
def g(x, labels, w):
  return jax.grad(f)(x, labels, w)


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

  print('Compiling pure optax CE full scale...')
  t0 = time.time()
  grad = g(x, labels, w)
  jax.block_until_ready(grad)
  print('Compiled and run in', time.time() - t0, 'seconds.')
