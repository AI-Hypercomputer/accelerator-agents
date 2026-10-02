---
title: 'Module: torchax/decompositions.py'
type: catalog
provenance: extracted
module: torchax/decompositions.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.decompositions`/
symbols:
  aten: aten.
  _try_register: _try_register().
  _grid_sampler_3d.compute_source_index: _grid_sampler_3d().compute_source_index().
  _grid_sampler_3d: _grid_sampler_3d().
  _reflection_pad: _reflection_pad().
  _replication_pad: _replication_pad().
  _reflection_or_replication_pad: _reflection_or_replication_pad().
  _grid_sampler_3d.get_summand: _grid_sampler_3d().get_summand().
  _grid_sampler_3d.compute_coordinates: _grid_sampler_3d().compute_coordinates().
  _grid_sampler_3d.clip: _grid_sampler_3d().clip().
  DECOMPOSITIONS: DECOMPOSITIONS.
  bernoulli: bernoulli().
  _grid_sampler_3d.reflect_coordinates: _grid_sampler_3d().reflect_coordinates().
  MUTABLE_DECOMPOSITION: MUTABLE_DECOMPOSITION.
  _reflection_pad.idx: _reflection_pad().idx().
  _replication_pad.idx: _replication_pad().idx().
  rand_like: rand_like().
  channel_shuffle: channel_shuffle().
  bernoulli_float: bernoulli_float().
  _sum_tensors: _sum_tensors().
  _grid_sampler_3d.unnormalize: _grid_sampler_3d().unnormalize().
  _grid_sampler_3d.in_bounds_cond: _grid_sampler_3d().in_bounds_cond().
  DispatchKey: DispatchKey.
  __all__.__all__: __all__.__all__.
---
# Module: [`torchax/decompositions.py`](../../../../../raw/code/torchax/torchax/decompositions.py)

## Functions
- `_grid_sampler_3d(a: torch.Tensor, grid: torch.Tensor, interpolation_mode: int = 0, padding_mode: int = 0, align_corners: bool = False)` — [`L160`](../../../../../raw/code/torchax/torchax/decompositions.py#L160) — References: https://github.com/pytorch/pytorch/blob/06a7dc21c1005750598c37f3adbc031183c74de6/torch/_decomp/decompositions.py#L4075
- `_reflection_or_replication_pad(a: Tensor, padding: tuple[int, ...], idx_fn: Callable[[int, int, int], Tensor])` — [`L88`](../../../../../raw/code/torchax/torchax/decompositions.py#L88)
- `_reflection_pad(a: Tensor, padding: tuple[int, ...])` — [`L53`](../../../../../raw/code/torchax/torchax/decompositions.py#L53)
- `_replication_pad(a: Tensor, padding: tuple[int, ...])` — [`L71`](../../../../../raw/code/torchax/torchax/decompositions.py#L71)
- `_sum_tensors(ts)` — [`L155`](../../../../../raw/code/torchax/torchax/decompositions.py#L155)
- `_try_register(op, impl)` — [`L44`](../../../../../raw/code/torchax/torchax/decompositions.py#L44)
- `bernoulli(self, *, generator=None)` — [`L120`](../../../../../raw/code/torchax/torchax/decompositions.py#L120)
- `bernoulli_float(self, p=0.5)` — [`L147`](../../../../../raw/code/torchax/torchax/decompositions.py#L147)
- `channel_shuffle(self, groups)` — [`L132`](../../../../../raw/code/torchax/torchax/decompositions.py#L132)
- `clip(xs: torch.Tensor, ys: torch.Tensor, zs, ws: torch.Tensor)` — [`L234`](../../../../../raw/code/torchax/torchax/decompositions.py#L234)
- `compute_coordinates(coords: Tensor, size: int)` — [`L205`](../../../../../raw/code/torchax/torchax/decompositions.py#L205)
- `compute_source_index(coords: Tensor, size: int)` — [`L217`](../../../../../raw/code/torchax/torchax/decompositions.py#L217)
- `get_summand(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, w)` — [`L251`](../../../../../raw/code/torchax/torchax/decompositions.py#L251)
- `idx(left, middle, right)` — [`L54`](../../../../../raw/code/torchax/torchax/decompositions.py#L54)
- `idx(left, middle, right)` — [`L72`](../../../../../raw/code/torchax/torchax/decompositions.py#L72)
- `in_bounds_cond(xs: Tensor, ys: Tensor, zs)` — [`L225`](../../../../../raw/code/torchax/torchax/decompositions.py#L225)
- `rand_like(self, **kwargs)` — [`L127`](../../../../../raw/code/torchax/torchax/decompositions.py#L127)
- `reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int)` — [`L193`](../../../../../raw/code/torchax/torchax/decompositions.py#L193)
- `unnormalize(coords: Tensor, size: int)` — [`L182`](../../../../../raw/code/torchax/torchax/decompositions.py#L182)

## Module values
- `DECOMPOSITIONS` — [`L308`](../../../../../raw/code/torchax/torchax/decompositions.py#L308) — documented in [torchax-export](../../concepts/torchax-export.md)
- `DispatchKey` — [`L35`](../../../../../raw/code/torchax/torchax/decompositions.py#L35)
- `MUTABLE_DECOMPOSITION` — [`L793`](../../../../../raw/code/torchax/torchax/decompositions.py#L793)
- `__all__` — [`L39`](../../../../../raw/code/torchax/torchax/decompositions.py#L39)
- `aten` — [`L41`](../../../../../raw/code/torchax/torchax/decompositions.py#L41)

