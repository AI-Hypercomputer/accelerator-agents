import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ("fsdp",))
print("Devices:", jax.devices())


def test_bs(b_block_sz, v_block_sz):
  print(f"Testing b_block_sz={b_block_sz}, v_block_sz={v_block_sz}")

  def f(x, labels, w):
    from tokamax._src.ops.linear_softmax_cross_entropy_loss.chunked_xla import linear_softmax_cross_entropy_loss_fwd_chunked_xla

    return linear_softmax_cross_entropy_loss_fwd_chunked_xla(
        x,
        labels,
        w,
        reduction="sum",
        b_block_sz=b_block_sz,
        v_block_sz=v_block_sz,
    )[0]

  def g(x, labels, w):
    return jax.grad(f)(x, labels, w)

  with mesh:
    x = jax.device_put(
        jnp.zeros((131072, 4096), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P("fsdp")),
    )
    labels = jax.device_put(
        jnp.zeros((131072,), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P("fsdp")),
    )
    w = jax.device_put(
        jnp.zeros((4096, 32768), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P(None, "fsdp")),
    )

    import time

    t0 = time.time()
    try:
      grad = jax.jit(g)(x, labels, w)
      jax.block_until_ready(grad)
      print("Compiled and run in", time.time() - t0, "seconds.")
    except Exception as e:
      print("Exception:", e)


test_bs(2048, 2048)
test_bs(4096, 4096)
test_bs(8192, 8192)
