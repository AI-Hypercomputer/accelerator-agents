---
title: 'Module: ejkernel/callib/_ejit.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_ejit.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._ejit`/
symbols:
  ejit: ejit().
  result3: result3.
  smart_compile: smart_compile().
  COMPILE_FUNC_DIR: COMPILE_FUNC_DIR.
  ejit.wrapper: ejit().wrapper().
  ejit.get_compiled_and_cache: ejit().get_compiled_and_cache().
  jnp: jnp.
  result1: result1.
  result2: result2.
  load_cached_functions: load_cached_functions().
  COMPILED_FILE_NAME: COMPILED_FILE_NAME.
  RECOMPILE_FORCE: RECOMPILE_FORCE.
  ECACHE_COMPILES: ECACHE_COMPILES.
  COMPILED_CACHE.COMPILED_CACHE: COMPILED_CACHE.COMPILED_CACHE.
  a: a.
  b: b.
  ALLOW_FULL_CACHE: ALLOW_FULL_CACHE.
  CACHE_DIR: CACHE_DIR.
  _get_args_signature: _get_args_signature().
  ejit._resolve_static_key_part: ejit()._resolve_static_key_part().
  c: c.
  d: d.
  P: P.
  my_function: my_function().
  save_compiled_fn: save_compiled_fn().
  load_compiled_fn: load_compiled_fn().
  hash_fn: hash_fn().
  R: R.
  SIGNATURE_FILE_NAME: SIGNATURE_FILE_NAME.
  _get_hardware_signature: _get_hardware_signature().
  _get_leaf_signature: _get_leaf_signature().
  get_safe_hash_int: get_safe_hash_int().
  get_hash_of_lowering: get_hash_of_lowering().
---
# Module: [`ejkernel/callib/_ejit.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py)

## Functions
- `_get_args_signature(args: tuple, kwargs: dict)` — [`L118`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L118) — Create signature for function arguments.
- `_get_hardware_signature()` — [`L93`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L93) — Create signature for current JAX hardware environment.
- `_get_leaf_signature(leaf: tp.Any)` — [`L102`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L102) — Generate hashable signature for PyTree leaf.
- `_resolve_static_key_part()` — [`L256`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L256) — Lazily resolve cache key components to avoid import-time backend init.
- `ejit(func: tp.Callable[P, R] | None = None, *, static_argnums: int | tp.Sequence[int] | None = None, static_argnames: str | tp.Iterable[str] | None = None, donate_argnums: int | tp.Sequence[int] | None = None, in_shardings: tp.Any = None, out_shardings: tp.Any = None, donate_argnames: str | tp.Iterable[str] | None = None, keep_unused: bool = False, backend: str | None = None, inline: bool = False, compiler_options: dict[str, tp.Any] | None = None)` — [`L136`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L136) — Enhanced JIT compilation with persistent caching.
- `get_compiled_and_cache(static_key: str, args_sig: str, args, kwargs)` — [`L292`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L292) — Retrieve or compile a function with the given argument signature.
- `get_hash_of_lowering(lowered_func: Lowered)` — [`L572`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L572) — Generate a SHA-256 hash of a lowered JAX function.
- `get_safe_hash_int(text, algorithm="md5")` — [`L538`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L538) — Generate a deterministic integer hash of text using the specified algorithm.
- `hash_fn(self)` — [`L517`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L517) — Generate a deterministic integer hash for an object based on its attribute values.
- `load_cached_functions(verbose: bool = True)` — [`L388`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L388) — Pre-load all cached compiled functions from disk into memory.
- `load_compiled_fn(path: str | os.PathLike, prefix: str | None = None)` — [`L478`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L478) — Load a previously saved compiled function from disk.
- `my_function(x, y)` — [`L687`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L687) — Example function demonstrating ejit usage.
- `save_compiled_fn(path: str | os.PathLike, fn: Compiled, prefix: str | None = None)` — [`L427`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L427) — Save a compiled JAX function to disk for later reuse.
- `smart_compile(lowered_func: Lowered, tag: str | None = None, verbose: bool = True, cache_key: tuple[str, tuple] | None = None)` — [`L590`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L590) — Compile a lowered JAX function with intelligent disk caching.
- `wrapper(*args: P.args, **kwargs: P.kwargs)` — [`L353`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L353) — Execute the cached compiled function or fall back to jitted version.

## Module values
- `ALLOW_FULL_CACHE` — [`L80`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L80)
- `CACHE_DIR` — [`L82`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L82)
- `COMPILED_CACHE` — [`L90`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L90)
- `COMPILED_FILE_NAME` — [`L88`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L88)
- `COMPILE_FUNC_DIR` — [`L83`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L83)
- `ECACHE_COMPILES` — [`L79`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L79)
- `P` — [`L75`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L75)
- `R` — [`L76`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L76)
- `RECOMPILE_FORCE` — [`L78`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L78)
- `SIGNATURE_FILE_NAME` — [`L89`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L89)
- `a` — [`L691`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L691)
- `b` — [`L692`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L692)
- `c` — [`L697`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L697)
- `d` — [`L698`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L698)
- `jnp` — [`L684`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L684)
- `result1` — [`L694`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L694)
- `result2` — [`L695`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L695)
- `result3` — [`L699`](../../../../../../raw/code/ejkernel/ejkernel/callib/_ejit.py#L699)

