import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


@jax.custom_vjp
def fused_qknorm_rope(x, weight, cos, sin, eps=1e-6):
  """Applies RMSNorm followed by RoPE in a single Pallas kernel.

  x: [B, H, T, D] weight: [D] cos: [B, T, D] sin: [B, T, D] out: [B, H, T, D]
  """
  B, H, T, D = x.shape
  block_size = min(256, T)
  if T % block_size != 0:
    raise ValueError(f"T ({T}) must be divisible by block_size ({block_size})")

  def _fused_qknorm_rope_kernel(
      x_ref, w_ref, cos_ref, sin_ref, eps_ref, out_ref
  ):
    # x_ref: [1, 1, block_size, D]
    # w_ref: [D]
    # cos_ref: [1, block_size, D]
    # sin_ref: [1, block_size, D]
    # eps_ref: [1]
    # out_ref: [1, 1, block_size, D]
    x_val = x_ref[0, 0, :, :]
    w = w_ref[:]
    c = cos_ref[0, :, :]
    s = sin_ref[0, :, :]
    eps_val = eps_ref[0]

    # QK-norm
    x32 = x_val.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    rsqrt = jax.lax.rsqrt(var + eps_val)
    x_norm = (x32 * rsqrt).astype(x_val.dtype)
    x_norm = x_norm * w

    # RoPE
    d = x_val.shape[-1]
    half_d = d // 2
    x_rot = jnp.concatenate([-x_norm[:, half_d:], x_norm[:, :half_d]], axis=-1)

    out_val = (x_norm * c) + (x_rot * s)
    out_ref[0, 0, :, :] = out_val.astype(out_ref.dtype)

  eps_arr = jnp.array([eps], dtype=jnp.float32)

  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  from . import modeling_qwen3

  mesh = modeling_qwen3._SPLASH_MESH
  if mesh is None:
    raise ValueError(
        "fused_qknorm_rope requires a registered mesh via set_splash_mesh"
    )

  def _sharded_pallas_call(x_local, w_local, cos_local, sin_local, eps_local):
    B_local = x_local.shape[0]
    H_local = x_local.shape[1]
    T_local = x_local.shape[2]
    return pl.pallas_call(
        _fused_qknorm_rope_kernel,
        out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
        grid=(B_local, H_local, T_local // block_size),
        in_specs=[
            pl.BlockSpec((1, 1, block_size, D), lambda b, h, t: (b, h, t, 0)),
            pl.BlockSpec((D,), lambda b, h, t: (0,)),
            pl.BlockSpec((1, block_size, D), lambda b, h, t: (b, t, 0)),
            pl.BlockSpec((1, block_size, D), lambda b, h, t: (b, t, 0)),
            pl.BlockSpec((1,), lambda b, h, t: (0,)),
        ],
        out_specs=pl.BlockSpec(
            (1, 1, block_size, D), lambda b, h, t: (b, h, t, 0)
        ),
    )(x_local, w_local, cos_local, sin_local, eps_local)

  sharded_fn = shard_map(
      _sharded_pallas_call,
      mesh=mesh,
      in_specs=(
          P("fsdp", "tp", None, None),
          P(),
          P("fsdp", None, None),
          P("fsdp", None, None),
          P(),
      ),
      out_specs=P("fsdp", "tp", None, None),
      check_rep=False,
  )

  out = sharded_fn(x, weight, cos, sin, eps_arr)
  return out


def fused_qknorm_rope_fwd(x, weight, cos, sin, eps=1e-6):
  return fused_qknorm_rope(x, weight, cos, sin, eps=eps), (
      x,
      weight,
      cos,
      sin,
      eps,
  )


def fused_qknorm_rope_bwd(res, g):
  x, weight, cos, sin, eps = res
  # VERY IMPORTANT: Since the forward kernel isn't automatically differentiable,
  # and writing a custom bwd Pallas kernel is extremely error-prone and time-consuming,
  # we just fall back to standard XLA ops for the backward pass!
  # The hypothesis is about HBM footprint reduction on the FORWARD pass anyway,
  # so we still get the memory savings where we need it without the autodiff headache.

  # QK-norm standard
  x32 = x.astype(jnp.float32)
  var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
  rsqrt = jax.lax.rsqrt(var + eps)
  x_norm = (x32 * rsqrt).astype(x.dtype)
  x_norm = x_norm * weight

  # RoPE standard
  d = x.shape[-1]
  half_d = d // 2
  x_rot = jnp.concatenate(
      [-x_norm[..., half_d:], x_norm[..., :half_d]], axis=-1
  )
  cos_bcast = jnp.expand_dims(cos, axis=1)
  sin_bcast = jnp.expand_dims(sin, axis=1)
  out_ref = (x_norm * cos_bcast) + (x_rot * sin_bcast)

  # Use jax.vjp on the standard op implementation!
  def ref_impl(x_, weight_, cos_, sin_):
    x32_ = x_.astype(jnp.float32)
    var_ = jnp.mean(x32_ * x32_, axis=-1, keepdims=True)
    rsqrt_ = jax.lax.rsqrt(var_ + eps)
    x_norm_ = (x32_ * rsqrt_).astype(x_.dtype)
    x_norm_ = x_norm_ * weight_

    x_rot_ = jnp.concatenate(
        [-x_norm_[..., half_d:], x_norm_[..., :half_d]], axis=-1
    )
    # cos_ and sin_ are [B, T, D]
    # x_norm_ is [B, H, T, D]
    cos_bcast_ = jnp.expand_dims(cos_, axis=1)  # [B, 1, T, D]
    sin_bcast_ = jnp.expand_dims(sin_, axis=1)  # [B, 1, T, D]
    return (x_norm_ * cos_bcast_) + (x_rot_ * sin_bcast_)

  _, vjp_fn = jax.vjp(ref_impl, x, weight, cos, sin)
  dx, dweight, dcos, dsin = vjp_fn(g)

  # Return None for the eps gradient
  return dx, dweight, dcos, dsin, None


fused_qknorm_rope.defvjp(fused_qknorm_rope_fwd, fused_qknorm_rope_bwd)
