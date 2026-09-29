---
title: 'Module: easydel/data/utils.py'
type: catalog
provenance: extracted
module: easydel/data/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.utils`/
symbols:
  with_retry: with_retry().
  is_streaming: is_streaming().
  wrap_format_callback: wrap_format_callback().
  glob_files: glob_files().
  with_retry.decorator: with_retry().decorator().
  align_columns_intersection: align_columns_intersection().
  with_retry.decorator.wrapper: with_retry().decorator().wrapper().
  wrap_format_callback.wrapped: wrap_format_callback().wrapped().
  get_cached_filesystem: get_cached_filesystem().
  infer_builder_from_ext: infer_builder_from_ext().
  warn_deprecated: warn_deprecated().
---
# Module: [`easydel/data/utils.py`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py)

## Functions
- `align_columns_intersection(datasets: list)` — [`L212`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L212) — Align datasets to have only common columns.
- `decorator(func: Callable)` — [`L57`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L57)
- `get_cached_filesystem(protocol: str, cache_dir: str | Path, cache_type: str = "filecache", expiry_time: int = 86400, storage_options: dict | None = None)` — [`L80`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L80) — Get a cached fsspec filesystem for remote protocols.
- `glob_files(pattern: str, recursive: bool = True)` — [`L155`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L155) — Expand glob patterns to actual file paths.
- `infer_builder_from_ext(path: str)` — [`L135`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L135) — Infer the dataset builder type from file extension.
- `is_streaming(ds)` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L123) — Check if a dataset is a streaming dataset.
- `warn_deprecated(msg: str)` — [`L240`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L240) — Issue a deprecation warning.
- `with_retry(max_retries: int = 3, initial_delay: float = 0.1, max_delay: float = 10, backoff_factor: float = 2, retryable_exceptions: tuple = (IOError, OSError, TimeoutError))` — [`L32`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L32) — Decorator for retry with exponential backoff on transient errors.
- `wrap_format_callback(fn, content_key: str = "content")` — [`L187`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L187) — Wrap a format callback to ensure it returns a dictionary.
- `wrapped(ex)` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L201)
- `wrapper(*args, **kwargs)` — [`L59`](../../../../../../raw/code/EasyDeL/easydel/data/utils.py#L59)

