---
title: 'Module: src/alphafold3/model/params.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/params.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.params`/
symbols:
  _MultiFileIO.readinto: _MultiFileIO#readinto().
  _MultiFileIO.seek: _MultiFileIO#seek().
  _MultiFileIO._offsets: _MultiFileIO#_offsets.
  _MultiFileIO._handles: _MultiFileIO#_handles.
  _MultiFileIO._sizes: _MultiFileIO#_sizes.
  get_model_haiku_params: get_model_haiku_params().
  _MultiFileIO._abspos: _MultiFileIO#_abspos.
  _MultiFileIO._length: _MultiFileIO#_length.
  _MultiFileIO._abs_to_rel: _MultiFileIO#_abs_to_rel().
  _MultiFileIO._relpos: _MultiFileIO#_relpos.
  _read_record: _read_record().
  read_records: read_records().
  open_for_reading: open_for_reading().
  select_model_files: select_model_files().
  _MultiFileIO.close: _MultiFileIO#close().
  _MultiFileIO.closed: _MultiFileIO#closed().
  _MultiFileIO.tell: _MultiFileIO#tell().
  RecordError: RecordError#
  _MultiFileIO._stack: _MultiFileIO#_stack.
  _MultiFileIO: _MultiFileIO#
  _MultiFileIO._files: _MultiFileIO#_files.
  _match_model: _match_model().
  encode_record: encode_record().
  _MultiFileIO.__init__: _MultiFileIO#__init__().
  _MultiFileIO.fileno: _MultiFileIO#fileno().
  _MultiFileIO.readable: _MultiFileIO#readable().
---
# Module: [`src/alphafold3/model/params.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py)

## Classes
### `RecordError`  ·  implements/extends Exception
- def: [`src/alphafold3/model/params.py:40`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L40)
- doc: Error reading a record.
- signature: `class RecordError(Exception):`
- used by: [`_read_record`](params.md#_read_record)

### `_MultiFileIO`  ·  implements/extends RawIOBase
- def: [`src/alphafold3/model/params.py:95`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L95)
- doc: A file-like object that presents a concatenated view of multiple files.
- signature: `class _MultiFileIO(io.RawIOBase):`
- members:
  - `close(self)` — [`L119`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L119)
  - `closed(self)` — [`L123`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L123)
  - `fileno(self)` — [`L126`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L126)
  - `readable(self)` — [`L129`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L129)
  - `readinto(self, b: bytearray | memoryview)` — [`L148`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L148)
  - `seek(self, pos: int, whence: int = os.SEEK_SET,)` — [`L135`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L135)
  - `tell(self)` — [`L132`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L132)
- protocol/private: `__init__`[`L98`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L98), `_abs_to_rel`[`L115`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L115), `_abspos`[`L112`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L112), `_files`[`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L99), `_handles`[`L101`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L101), `_length`[`L108`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L108), `_offsets`[`L109`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L109), `_relpos`[`L113`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L113), `_sizes`[`L104`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L104), `_stack`[`L100`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L100)
- used by: [`open_for_reading`](params.md#open_for_reading)

## Functions
- `_match_model(paths: list[epath.Path], pattern: re.Pattern[str])` — [`L181`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L181) — Match files in a directory with a pattern, and group by model name.
- `_read_record(stream: IO[bytes])` — [`L62`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L62) — Reads a record encoded by `_encode_record` from a byte stream.
- `encode_record(scope: str, name: str, arr: np.ndarray)` — [`L44`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L44) — Encodes a single haiku param as bytes, preserving non-numpy dtypes.
- `get_model_haiku_params(model_dir: epath.PathLike)` — [`L224`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L224) — Get the Haiku parameters from a model name.
- `open_for_reading(model_files: list[epath.PathLike], is_compressed: bool)` — [`L172`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L172)
- `read_records(stream: IO[bytes])` — [`L89`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L89) — Fully reads the contents of a byte stream.
- `select_model_files(model_dir: epath.PathLike, model_name: str | None = None)` — [`L193`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/params.py#L193) — Select the model files from a model directory.

