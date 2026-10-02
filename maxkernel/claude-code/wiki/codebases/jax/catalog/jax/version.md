---
title: 'Module: jax/version.py'
type: catalog
provenance: extracted
module: jax/version.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.version`/_
symbols:
  _get_version_string: get_version_string().
  _get_version_for_build: get_version_for_build().
  __version__: _version__.
  _version: version.
  __version_info__: _version_info__.
  _release_version._release_version: release_version._release_version.
  _get_cmdclass: get_cmdclass().
  _get_cmdclass._build_py.run: get_cmdclass()._build_py#run().
  _get_cmdclass._sdist.make_release_tree: get_cmdclass()._sdist#make_release_tree().
  _minimum_jaxlib_version_info: minimum_jaxlib_version_info.
  _write_version: write_version().
  _minimum_jaxlib_version: minimum_jaxlib_version.
  _version_from_todays_date: version_from_todays_date().
  _version_from_git_tree: version_from_git_tree().
  _version_as_tuple: version_as_tuple().
  _get_cmdclass._build_py: get_cmdclass()._build_py#
  _get_cmdclass._sdist: get_cmdclass()._sdist#
  _is_prerelease: is_prerelease().
  _git_hash._git_hash: git_hash._git_hash.
---
# Module: [`jax/version.py`](../../../../../raw/code/jax/jax/version.py)

## Classes
### `_build_py`  ·  implements/extends build_py
- def: [`jax/version.py:129`](../../../../../raw/code/jax/jax/version.py#L129)
- signature: `class _build_py(build_py_orig):`
- members:
  - `run(self)` — [`L130`](../../../../../raw/code/jax/jax/version.py#L130)
- uses (calls/refs, reference-scoped): [`_release_version`](version.md#_release_version._release_version), [`_write_version`](version.md#_write_version)
- used by: [`_get_cmdclass`](version.md#_get_cmdclass)

### `_sdist`  ·  implements/extends sdist
- def: [`jax/version.py:149`](../../../../../raw/code/jax/jax/version.py#L149)
- signature: `class _sdist(sdist_orig):`
- members:
  - `make_release_tree(self, base_dir, files)` — [`L150`](../../../../../raw/code/jax/jax/version.py#L150)
- uses (calls/refs, reference-scoped): [`_release_version`](version.md#_release_version._release_version), [`_write_version`](version.md#_write_version)
- used by: [`_get_cmdclass`](version.md#_get_cmdclass)

## Functions
- `_get_cmdclass(pkg_source_path)` — [`L125`](../../../../../raw/code/jax/jax/version.py#L125)
- `_get_version_for_build()` — [`L73`](../../../../../raw/code/jax/jax/version.py#L73) — Determine the version at build time.
- `_get_version_string()` — [`L34`](../../../../../raw/code/jax/jax/version.py#L34)
- `_is_prerelease()` — [`L97`](../../../../../raw/code/jax/jax/version.py#L97) — Determine if this is a pre-release ("rc" wheels) build.
- `_version_as_tuple(version_str)` — [`L162`](../../../../../raw/code/jax/jax/version.py#L162)
- `_version_from_git_tree(base_version: str)` — [`L49`](../../../../../raw/code/jax/jax/version.py#L49)
- `_version_from_todays_date(base_version: str)` — [`L44`](../../../../../raw/code/jax/jax/version.py#L44)
- `_write_version(fname: str)` — [`L103`](../../../../../raw/code/jax/jax/version.py#L103) — Used by setup.py to write the specified version info into the source tree.

## Module values
- `__version__` — [`L159`](../../../../../raw/code/jax/jax/version.py#L159)
- `__version_info__` — [`L165`](../../../../../raw/code/jax/jax/version.py#L165)
- `_git_hash` — [`L32`](../../../../../raw/code/jax/jax/version.py#L32)
- `_minimum_jaxlib_version` — [`L160`](../../../../../raw/code/jax/jax/version.py#L160)
- `_minimum_jaxlib_version_info` — [`L166`](../../../../../raw/code/jax/jax/version.py#L166)
- `_release_version` — [`L28`](../../../../../raw/code/jax/jax/version.py#L28)
- `_version` — [`L24`](../../../../../raw/code/jax/jax/version.py#L24)

