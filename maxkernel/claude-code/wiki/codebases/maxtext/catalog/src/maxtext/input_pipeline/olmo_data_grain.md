---
title: 'Module: src/maxtext/input_pipeline/olmo_data_grain.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/olmo_data_grain.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.olmo_data_grain`/
symbols:
  OlmoNpyDataSource.__getitem__: OlmoNpyDataSource#__getitem__().
  make_olmo_grain_data_loader: make_olmo_grain_data_loader().
  OlmoIndexSampler.__getitem__: OlmoIndexSampler#__getitem__().
  OlmoIndexSampler._shard_indices_for_epoch: OlmoIndexSampler#_shard_indices_for_epoch().
  OlmoNpyDataSource.__repr__: OlmoNpyDataSource#__repr__().
  OlmoIndexSampler.__repr__: OlmoIndexSampler#__repr__().
  NgramFilterTransform.map: NgramFilterTransform#map().
  _MmapCache.get: _MmapCache#get().
  OlmoIndexSampler.shuffled_global_indices: OlmoIndexSampler#shuffled_global_indices().
  OlmoIndexSampler.shard_indices: OlmoIndexSampler#shard_indices().
  _MmapCache._mmaps: _MmapCache#_mmaps.
  OlmoIndexSampler.num_local_instances_per_epoch: OlmoIndexSampler#num_local_instances_per_epoch().
  OlmoIndexSampler._total: OlmoIndexSampler#_total.
  OlmoNpyDataSource.__len__: OlmoNpyDataSource#__len__().
  OlmoNpyDataSource.__setstate__: OlmoNpyDataSource#__setstate__().
  OlmoNpyDataSource._dtype: OlmoNpyDataSource#_dtype.
  OlmoNpyDataSource._sequence_length: OlmoNpyDataSource#_sequence_length.
  OlmoNpyDataSource._mmaps: OlmoNpyDataSource#_mmaps.
  OlmoNpyDataSource._index: OlmoNpyDataSource#_index.
  OlmoNpyDataSource._resolve_path: OlmoNpyDataSource#_resolve_path().
  OlmoIndexSampler._shard_index: OlmoIndexSampler#_shard_index.
  OlmoIndexSampler._shard_count: OlmoIndexSampler#_shard_count.
  OlmoIndexSampler._cached_shard_indices: OlmoIndexSampler#_cached_shard_indices.
  OlmoNpyDataSource.__init__: OlmoNpyDataSource#__init__().
  OlmoIndexSampler.num_instances: OlmoIndexSampler#num_instances().
  OlmoIndexSampler.__setstate__: OlmoIndexSampler#__setstate__().
  OlmoNpyDataSource._path_remap: OlmoNpyDataSource#_path_remap.
  OlmoIndexSampler._seed: OlmoIndexSampler#_seed.
  OlmoIndexSampler._shuffle: OlmoIndexSampler#_shuffle.
  OlmoIndexSampler._cached_epoch: OlmoIndexSampler#_cached_epoch.
  OlmoIndexSampler._cache_lock: OlmoIndexSampler#_cache_lock.
  _MmapCache: _MmapCache#
  OlmoNpyDataSource: OlmoNpyDataSource#
  _MmapCache._max: _MmapCache#_max.
  _MmapCache._lock: _MmapCache#_lock.
  OlmoIndexSampler: OlmoIndexSampler#
  OlmoIndexSampler._initial_step: OlmoIndexSampler#_initial_step.
  _combine_seed_epoch: _combine_seed_epoch().
  NgramFilterTransform: NgramFilterTransform#
  NgramFilterTransform._max_period: NgramFilterTransform#_max_period.
  NgramFilterTransform._min_period: NgramFilterTransform#_min_period.
  NgramFilterTransform._max_count: NgramFilterTransform#_max_count.
  NgramFilterTransform._mask_value: NgramFilterTransform#_mask_value.
  ShiftToInputsTargets: ShiftToInputsTargets#
  OlmoNpyDataSource.__getstate__: OlmoNpyDataSource#__getstate__().
  _MmapCache.__init__: _MmapCache#__init__().
  OlmoIndexSampler.__init__: OlmoIndexSampler#__init__().
  OlmoIndexSampler.__getstate__: OlmoIndexSampler#__getstate__().
  NgramFilterTransform.__init__: NgramFilterTransform#__init__().
  ShiftToInputsTargets.map: ShiftToInputsTargets#map().
---
# Module: [`src/maxtext/input_pipeline/olmo_data_grain.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py)

## Classes
### `NgramFilterTransform`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/olmo_data_grain.py:337`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L337)
- doc: Add an `instance_mask` field per OLMo-core's repetition filter.
- signature: `class NgramFilterTransform(grain.transforms.Map):`
- members:
  - `map(self, element: Dict[str, Any])` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L359) — Add ``instance_mask`` to ``element`` based on the n-gram filter.
- protocol/private: `__init__`[`L346`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L346), `_mask_value`[`L357`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L357), `_max_count`[`L356`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L356), `_max_period`[`L354`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L354), `_min_period`[`L355`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L355)
- uses (calls/refs, reference-scoped): [`is_clean_instance`](olmo_data.md#is_clean_instance)
- used by: [`make_olmo_grain_data_loader`](olmo_data_grain.md#make_olmo_grain_data_loader)

### `OlmoIndexSampler`
- def: [`src/maxtext/input_pipeline/olmo_data_grain.py:153`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L153)
- doc: Global-shuffle sampler over an OLMo numpy mix.
- signature: `class OlmoIndexSampler:`
- members:
  - `__getitem__(self, index: int)` — [`L281`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L281) — Map a per-host global step ``index`` to the next record to fetch.
  - `__repr__(self)` — [`L307`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L307) — Stable repr — Grain compares ``repr(sampler)`` between the checkpoint
  - `num_instances(self)` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L226)
  - `num_local_instances_per_epoch(self)` — [`L230`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L230) — Instances assigned to *this* host per epoch (drops trailing remainder).
  - `shard_indices(self, *, seed: int, epoch: int)` — [`L249`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L249) — Slice the global shuffled order down to this host's share.
  - `shuffled_global_indices(self, *, seed: int, epoch: int)` — [`L234`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L234) — Build the full shuffled list for ``(seed, epoch)``.
- protocol/private: `__getstate__`[`L266`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L266), `__init__`[`L194`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L194), `__setstate__`[`L275`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L275), `_cache_lock`[`L221`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L221), `_cached_epoch`[`L219`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L219), `_cached_shard_indices`[`L220`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L220), `_initial_step`[`L215`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L215), `_seed`[`L211`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L211), `_shard_count`[`L213`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L213), `_shard_index`[`L212`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L212), `_shard_indices_for_epoch`[`L257`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L257), `_shuffle`[`L214`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L214), `_total`[`L210`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L210)
- uses (calls/refs, reference-scoped): [`_combine_seed_epoch`](olmo_data_grain.md#_combine_seed_epoch)
- used by: [`make_olmo_grain_data_loader`](olmo_data_grain.md#make_olmo_grain_data_loader)

### `OlmoNpyDataSource`  ·  implements/extends RandomAccessDataSource
- def: [`src/maxtext/input_pipeline/olmo_data_grain.py:39`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L39)
- doc: Random-access view of an OLMo numpy mix as a stream of token windows.
- signature: `class OlmoNpyDataSource(grain.sources.RandomAccessDataSource):`
- members:
  - `__repr__(self)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L112) — Stable repr — Grain compares ``repr(data_source)`` between the
- protocol/private: `__getitem__`[`L79`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L79), `__getstate__`[`L102`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L102), `__init__`[`L61`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L61), `__len__`[`L76`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L76), `__setstate__`[`L108`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L108), `_dtype`[`L69`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L69), `_index`[`L68`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L68), `_mmaps`[`L72`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L72), `_path_remap`[`L71`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L71), `_resolve_path`[`L96`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L96), `_sequence_length`[`L70`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L70)
- uses (calls/refs, reference-scoped): [`global_to_local`](olmo_data.md#global_to_local), [`total_instances`](olmo_data.md#OlmoNpyIndex.total_instances), [`files`](olmo_data.md#OlmoNpyIndex.files), [`get`](olmo_data_grain.md#_MmapCache.get), [`OlmoNpyIndex`](olmo_data.md#OlmoNpyIndex), [`sequence_length`](olmo_data.md#OlmoNpyIndex.sequence_length), [`dtype`](olmo_data.md#OlmoNpyIndex.dtype), [`fingerprint`](olmo_data.md#OlmoNpyIndex.fingerprint), [`path`](olmo_data.md#OlmoNpyFileEntry.path), [`_MmapCache`](olmo_data_grain.md#_MmapCache)
- used by: [`make_olmo_grain_data_loader`](olmo_data_grain.md#make_olmo_grain_data_loader)

### `ShiftToInputsTargets`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/olmo_data_grain.py:374`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L374)
- doc: Convert a `tokens` array into the keys MaxText's pretrain trainer expects.
- signature: `class ShiftToInputsTargets(grain.transforms.Map):`
- members:
  - `map(self, element: Dict[str, Any])` — [`L397`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L397) — Convert ``tokens`` into ``inputs`` / ``targets`` / segmentation tensors.
- used by: [`make_olmo_grain_data_loader`](olmo_data_grain.md#make_olmo_grain_data_loader)

### `_MmapCache`
- def: [`src/maxtext/input_pipeline/olmo_data_grain.py:125`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L125)
- members:
  - `get(self, path: str, dtype: np.dtype)` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L133) — Return a cached ``np.memmap`` for ``path``, opening it lazily.
- protocol/private: `__init__`[`L128`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L128), `_lock`[`L131`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L131), `_max`[`L129`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L129), `_mmaps`[`L130`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L130)
- used by: [`__getitem__`](olmo_data_grain.md#OlmoNpyDataSource.__getitem__), [`__setstate__`](olmo_data_grain.md#OlmoNpyDataSource.__setstate__), [`_mmaps`](olmo_data_grain.md#OlmoNpyDataSource._mmaps)

## Functions
- `_combine_seed_epoch(seed: int, epoch: int)` — [`L326`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L326) — Stable 64-bit mix of (seed, epoch) for the per-epoch shuffle RNG.
- `make_olmo_grain_data_loader(index: OlmoNpyIndex, *, seed: int, batch_size: int, shard_index: int, shard_count: int, apply_ngram_filter: bool = True, shift_to_inputs_targets: bool = True, path_remap: Optional[Dict[str, str]] = None, grain_worker_count: int = 0, grain_worker_buffer_size: int = 1, initial_step: int = 0)` — [`L432`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data_grain.py#L432) — Build a Grain ``DataLoader`` for OLMo-style fixed-seq-length training.

