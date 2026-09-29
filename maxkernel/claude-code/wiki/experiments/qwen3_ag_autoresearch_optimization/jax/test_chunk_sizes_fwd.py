import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ("fsdp",))
print("Devices:", jax.devices())


def test_bs(b_block_sz, v_block_sz):
  print(f"Testing FWD b_block_sz={b_block_sz}, v_block_sz={v_block_sz}")

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

  with mesh:
    x = jax.device_put(
        jnp.zeros((262144, 4096), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P("fsdp")),
    )
    labels = jax.device_put(
        jnp.zeros((262144,), dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, P("fsdp")),
    )
    # Real vocab size is 151936
    w = jax.device_put(
        jnp.zeros((4096, 151936), dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P(None, "fsdp")),
    )

    import time

    t0 = time.time()
    try:
      out = jax.jit(f)(x, labels, w)
      jax.block_until_ready(out)
      print("Compiled FWD and run in", time.time() - t0, "seconds.")
    except Exception as e:
      print("Exception:", e)


test_bs(8192, 16384)
