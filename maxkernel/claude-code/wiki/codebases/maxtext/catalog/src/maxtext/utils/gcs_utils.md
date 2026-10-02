---
title: 'Module: src/maxtext/utils/gcs_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/gcs_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.gcs_utils`/
symbols:
  _gcs_guard: _gcs_guard().
  storage: storage.
  gcs_delete_directory: gcs_delete_directory().
  upload_blob: upload_blob().
  read_bytes_from_gcs: read_bytes_from_gcs().
  parse_gcs_bucket_and_prefix: parse_gcs_bucket_and_prefix().
  upload_dump: upload_dump().
  gcs_path_exists: gcs_path_exists().
  gcs_list_directories: gcs_list_directories().
  write_dict_to_gcs_json: write_dict_to_gcs_json().
  write_config_raw_keys_for_gcs: write_config_raw_keys_for_gcs().
  gcs_glob_pattern: gcs_glob_pattern().
  read_json_from_gcs: read_json_from_gcs().
  mkdir_and_check_permissions: mkdir_and_check_permissions().
  gcs_delete_directory._delete_blob: gcs_delete_directory()._delete_blob().
  add_trailing_slash: add_trailing_slash().
---
# Module: [`src/maxtext/utils/gcs_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py)

## Functions
- `_delete_blob(blob)` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L191)
- `_gcs_guard(operation_name: str)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L35) — Check GCS availability for an operation.
- `add_trailing_slash(path)` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L72)
- `gcs_delete_directory(directory_path: str)` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L172) — Deletes a "directory" (all blobs with the prefix) from GCS.
- `gcs_glob_pattern(pattern)` — [`L201`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L201) — Globs GCS files and returns a list of full GCS paths.
- `gcs_list_directories(directory_path)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L135) — Lists "directories" (prefixes one level down) within a GCS "directory".
- `gcs_path_exists(file_path)` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L119) — Checks if a GCS file_path exists (no-op if not found and decoupled).
- `mkdir_and_check_permissions(path: str | epath.Path)` — [`L279`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L279) — Creates a directory if it doesn't exist and verifies write permissions.
- `parse_gcs_bucket_and_prefix(destination_gcs_name)` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L65)
- `read_bytes_from_gcs(file_path)` — [`L212`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L212) — Read raw bytes from a GCS file.
- `read_json_from_gcs(file_path)` — [`L234`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L234) — Read a json file from gcs bucket.
- `upload_blob(destination_gcs_name, source_file_name)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L78) — Uploads a file to a GCS location (no-op if not found and decoupled).
- `upload_dump(local_dir, target_dir, module_name=None, delete_local_after=True, all_host_upload=False)` — [`L89`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L89) — Uploads a directory to a GCS location, with an optional filter (no-op if not found and decoupled).
- `write_config_raw_keys_for_gcs(raw_keys)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L45) — Writes config raw keys to GCS (no-op if disabled or decoupled).
- `write_dict_to_gcs_json(data_dict, file_path)` — [`L254`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L254) — Writes a Python dictionary to a JSON file in GCS.

## Module values
- `storage` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/utils/gcs_utils.py#L32)

