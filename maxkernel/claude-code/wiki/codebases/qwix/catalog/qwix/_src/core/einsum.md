---
title: 'Module: qwix/_src/core/einsum.py'
type: catalog
provenance: extracted
module: qwix/_src/core/einsum.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.einsum`/
symbols:
  einsum: einsum().
  get_how_to_quantize: get_how_to_quantize().
  _perform_binary_einsum: _perform_binary_einsum().
---
# Module: [`qwix/_src/core/einsum.py`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum.py)

## Functions
- `_perform_binary_einsum(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, einsum_str: str, dot_general_func: Callable[..., jax.Array], preferred_element_type: jax.typing.DTypeLike | None, **kwargs: Any)` — [`L73`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum.py#L73) — Performs a binary einsum using the given dot_general function. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
- `einsum(*args, _qwix_dot_general=dot_general.dot_general, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L95`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum.py#L95) — Computes Einstein summation convention with support for ``QArray`` inputs. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)
- `get_how_to_quantize(*, einsum_str: str, ndims: tuple[int, int], for_lhs: bool, tile_size: int | float | None, **kwargs: Any)` — [`L27`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum.py#L27) — Get how to quantize from an einsum string. — documented in [qwix-_src-core-qarray](../../../../concepts/qwix-_src-core-qarray.md)

