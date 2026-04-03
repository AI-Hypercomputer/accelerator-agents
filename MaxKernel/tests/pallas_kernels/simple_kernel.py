import jax
import numpy as np
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]


def matmul(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
      pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
      pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j)),
    ],
    out_specs=pl.BlockSpec(
      (x.shape[0] // 2, y.shape[1] // 2),
      lambda i, j: (i, j),
    ),
  )(x, y)


k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (1024, 1024))
y = jax.random.normal(k2, (1024, 1024))
z = matmul(x, y)
np.testing.assert_allclose(z, x @ y)
