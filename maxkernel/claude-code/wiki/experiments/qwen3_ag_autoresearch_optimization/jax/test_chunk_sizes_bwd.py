import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))
print('Devices:', jax.devices())


def test_bs(b_block_sz, v_block_sz):
  print(f'Testing BWD b_block_sz={b_block_sz}, v_block_sz={v_block_sz}')

  def my_f_bwd(g, lse, x, labels, w):
    from tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla import linear_softmax_cross_entropy_loss_bwd_chunked_xla

    dx, dw = linear_softmax_cross_entropy_loss_bwd_chunked_xla(
        g,
        lse,
        x,
        labels,
        w,
        reduction='sum',
        b_block_sz=b_block_sz,
        v_block_sz=v_block_sz,
    )
    return dx, None, dw

  with mesh:
    x = jax.device_put(
        jnp.zeros((262144, 4096), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P('fsdp')),
    )
    labels = jax.device_put(
        jnp.zeros((262144,), dtype=jnp.int32),
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
        jnp.zeros((262144,), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P('fsdp')),
    )

    import time

    t0 = time.time()
    try:
      out = jax.jit(my_f_bwd)(g, lse, x, labels, w)
      jax.block_until_ready(out)
      print('Compiled BWD and run in', time.time() - t0, 'seconds.')
    except Exception as e:
      print('Exception:', e)


test_bs(8192, 16384)
