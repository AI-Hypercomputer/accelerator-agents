---
title: 'Module: src/maxtext/input_pipeline/olmo_data.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/olmo_data.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.olmo_data`/
symbols:
  build_index: build_index().
  load_index: load_index().
  OlmoNpyIndex.to_json_dict: OlmoNpyIndex#to_json_dict().
  find_periodic_sequences: find_periodic_sequences().
  global_to_local: global_to_local().
  _file_entry_from_header: _file_entry_from_header().
  compute_fingerprint: compute_fingerprint().
  OlmoNpyIndex.total_instances: OlmoNpyIndex#total_instances.
  OlmoNpyIndex.files: OlmoNpyIndex#files.
  OlmoNpyIndex: OlmoNpyIndex#
  OlmoNpyIndex.sequence_length: OlmoNpyIndex#sequence_length.
  OlmoNpyIndex.__post_init__: OlmoNpyIndex#__post_init__().
  is_clean_instance: is_clean_instance().
  OlmoNpyFileEntry: OlmoNpyFileEntry#
  INDEX_FORMAT_VERSION: INDEX_FORMAT_VERSION.
  OlmoNpyIndex.dtype: OlmoNpyIndex#dtype.
  OlmoNpyIndex.fingerprint: OlmoNpyIndex#fingerprint.
  parse_npy_header: parse_npy_header().
  read_npy_header_from_path: read_npy_header_from_path().
  _find_start_last_consecutive_true: _find_start_last_consecutive_true().
  OlmoNpyFileEntry.path: OlmoNpyFileEntry#path.
  OlmoNpyFileEntry.n_tokens: OlmoNpyFileEntry#n_tokens.
  OlmoNpyFileEntry.instance_offset: OlmoNpyFileEntry#instance_offset.
  OlmoNpyIndex.format_version: OlmoNpyIndex#format_version.
  OlmoNpyIndex.tokenizer: OlmoNpyIndex#tokenizer.
  OlmoNpyIndex.total_tokens: OlmoNpyIndex#total_tokens.
  RepetitionTuple.times: RepetitionTuple#times.
  OlmoNpyIndex.save: OlmoNpyIndex#save().
  has_npy_magic: has_npy_magic().
  OlmoNpyFileEntry.n_instances: OlmoNpyFileEntry#n_instances.
  _NPY_MAGIC: _NPY_MAGIC.
  RepetitionTuple: RepetitionTuple#
  _find_end_first_consecutive_true: _find_end_first_consecutive_true().
  OlmoNpyFileEntry.label: OlmoNpyFileEntry#label.
  OlmoNpyIndex._instance_offset_starts: OlmoNpyIndex#_instance_offset_starts.
  RepetitionTuple.start: RepetitionTuple#start.
  RepetitionTuple.end: RepetitionTuple#end.
  RepetitionTuple.period: RepetitionTuple#period.
  _group_consecutive_values: _group_consecutive_values().
  read_raw_metadata_from_path: read_raw_metadata_from_path().
---
# Module: [`src/maxtext/input_pipeline/olmo_data.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py)

## Classes
### `OlmoNpyFileEntry`
- def: [`src/maxtext/input_pipeline/olmo_data.py:44`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L44) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- doc: One file in the mix: `n_tokens // sequence_length` instances starting
- signature: `class OlmoNpyFileEntry:`
- members:
  - `instance_offset` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L53) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `label` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L50) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `n_instances` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L52) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `n_tokens` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L51) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `path` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L49) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- used by: [`build_index`](olmo_data.md#build_index), [`load_index`](olmo_data.md#load_index), [`__getitem__`](olmo_data_grain.md#OlmoNpyDataSource.__getitem__), [`global_to_local`](olmo_data.md#global_to_local), [`_file_entry_from_header`](olmo_data.md#_file_entry_from_header), [`compute_fingerprint`](olmo_data.md#compute_fingerprint), [`files`](olmo_data.md#OlmoNpyIndex.files), [`__post_init__`](olmo_data.md#OlmoNpyIndex.__post_init__)

### `OlmoNpyIndex`
- def: [`src/maxtext/input_pipeline/olmo_data.py:57`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L57) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- doc: Index over the files in an OLMo data mix. Build via
- signature: `class OlmoNpyIndex:`
- members:
  - `save(self, path: str)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L93) — Write the index as JSON to ``path`` (local filesystem). — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `to_json_dict(self)` — [`L80`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L80) — Return a JSON-serializable view (drops cached lookup helpers). — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `dtype` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L64) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `files` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L66) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `fingerprint` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L69) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `format_version` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L62) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `sequence_length` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L63) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `tokenizer` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L65) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `total_instances` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L67) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `total_tokens` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L68) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- protocol/private: `__post_init__`[`L75`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L75), `_instance_offset_starts`[`L73`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L73)
- uses (calls/refs, reference-scoped): [`OlmoNpyFileEntry`](olmo_data.md#OlmoNpyFileEntry), [`instance_offset`](olmo_data.md#OlmoNpyFileEntry.instance_offset)
- used by: [`build_index`](olmo_data.md#build_index), [`load_index`](olmo_data.md#load_index), [`__getitem__`](olmo_data_grain.md#OlmoNpyDataSource.__getitem__), [`_make_loader_for_host`](olmo_grain_data_processing.md#_make_loader_for_host), [`global_to_local`](olmo_data.md#global_to_local), [`make_olmo_grain_data_loader`](olmo_data_grain.md#make_olmo_grain_data_loader), [`__repr__`](olmo_data_grain.md#OlmoNpyDataSource.__repr__), [`__len__`](olmo_data_grain.md#OlmoNpyDataSource.__len__), [`_dtype`](olmo_data_grain.md#OlmoNpyDataSource._dtype), [`_sequence_length`](olmo_data_grain.md#OlmoNpyDataSource._sequence_length), [`__init__`](olmo_data_grain.md#OlmoNpyDataSource.__init__)

### `RepetitionTuple`  ·  implements/extends NamedTuple
- def: [`src/maxtext/input_pipeline/olmo_data.py:311`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L311) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- doc: `arr[start:end]` is a periodic span of length `period`,
- signature: `class RepetitionTuple(NamedTuple):`
- members:
  - `end` — [`L316`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L316) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `period` — [`L317`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L317) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `start` — [`L315`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L315) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
  - `times` — [`L318`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L318) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- used by: [`find_periodic_sequences`](olmo_data.md#find_periodic_sequences), [`is_clean_instance`](olmo_data.md#is_clean_instance)

## Functions
- `_file_entry_from_header(path: str, label: str, dtype: str, shape: Tuple[int, ...], sequence_length: int, instance_offset: int)` — [`L228`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L228) — Build a file entry from a parsed .npy header (validates shape is 1-D). — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `_find_end_first_consecutive_true(arr: np.ndarray)` — [`L321`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L321) — End offset (exclusive) of the leading run of True in ``arr``. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `_find_start_last_consecutive_true(arr: np.ndarray)` — [`L336`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L336) — Start offset of the trailing run of True in ``arr``, or -1 if none. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `_group_consecutive_values(arr: np.ndarray, stepsize: int = 1)` — [`L342`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L342) — Split a 1-D array of ints into runs of consecutive values. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `build_index(paths_and_labels: Sequence[Tuple[str, str]], sequence_length: int, *, tokenizer: str, header_reader=read_npy_header_from_path)` — [`L250`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L250) — Build an :class:`OlmoNpyIndex` from ``(path, label)`` entries. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `compute_fingerprint(sequence_length: int, dtype: str, tokenizer: str, files: Sequence[OlmoNpyFileEntry])` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L146) — Stable hash over the fields a restart must preserve. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `find_periodic_sequences(arr: np.ndarray, max_period: int, min_period: int = 1, mask_value: int = -1)` — [`L349`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L349) — Yield :class:`RepetitionTuple` for periodic spans of length ≥ 3 in ``arr``. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `global_to_local(index: OlmoNpyIndex, instance_id: int)` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L131) — Global instance index → ``(file_idx, token_offset)``.
- `has_npy_magic(first_bytes: bytes)` — [`L223`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L223) — Quick check: does this look like a real .npy file?
- `is_clean_instance(input_ids: np.ndarray, *, repetition_max_period: int = 13, repetition_min_period: int = 1, repetition_max_count: int = 32, mask_value: int = -1)` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L398) — ``False`` iff ``input_ids`` has any periodic span (period ∈ — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `load_index(path: str)` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L99) — Load an index from JSON written by :meth:`OlmoNpyIndex.save`. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `parse_npy_header(stream: BinaryIO)` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L177) — Parse a .npy v1/v2/v3 header. Returns ``(dtype_str, shape)``. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `read_npy_header_from_path(path: str)` — [`L201`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L201) — Convenience wrapper for :func:`parse_npy_header` on a local file. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `read_raw_metadata_from_path(path: str, dtype: str)` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L207) — Headerless raw binary: ``n_tokens = file_size // itemsize``.

## Module values
- `INDEX_FORMAT_VERSION` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L40) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `_NPY_MAGIC` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_data.py#L174) — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)

