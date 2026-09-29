---
title: 'Module: jax/_src/pallas/fuser/fusible.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/fusible.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.fusible`/
symbols:
  fusible_p: fusible_p.
  fusible.decorator.wrapper: fusible().decorator().wrapper().
  _make_trivial_fusion: _make_trivial_fusion().
  _fusible_to_lojax: _fusible_to_lojax().
  fusible.decorator.wrapper.wrapped: fusible().decorator().wrapper().wrapped().
  fusible: fusible().
  _fusible_trivial_batching_rule: _fusible_trivial_batching_rule().
  fusible.decorator: fusible().decorator().
  _fusible_is_high: _fusible_is_high().
---
# Module: [`jax/_src/pallas/fuser/fusible.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py)

## Functions
- `_fusible_is_high(*_, jaxpr, **params)` — [`L34`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L34)
- `_fusible_to_lojax(*hi_args, jaxpr, num_consts, **_)` — [`L119`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L119)
- `_fusible_trivial_batching_rule(axis_data, args, dims, **kwargs)` — [`L103`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L103)
- `_make_trivial_fusion(x: jax.Array)` — [`L41`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L41)
- `decorator(f)` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L51)
- `fusible(f=None, *, output_fusion_prefix: Any = True)` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L50)
- `wrapped(*args)` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L53)
- `wrapper(*args)` — [`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L52)

## Module values
- `fusible_p` — [`L31`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusible.py#L31)

