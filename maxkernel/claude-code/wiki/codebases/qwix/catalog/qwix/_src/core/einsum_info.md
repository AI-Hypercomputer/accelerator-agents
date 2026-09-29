---
title: 'Module: qwix/_src/core/einsum_info.py'
type: catalog
provenance: extracted
module: qwix/_src/core/einsum_info.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.einsum_info`/
symbols:
  EinsumInfo.batch_chars: EinsumInfo#batch_chars().
  EinsumInfo.contract_chars: EinsumInfo#contract_chars().
  EinsumInfo.parse: EinsumInfo#parse().
  EinsumInfo.output_perm: EinsumInfo#output_perm().
  EinsumInfo.rhs: EinsumInfo#rhs.
  EinsumInfo.dimension_numbers: EinsumInfo#dimension_numbers().
  EinsumInfo.lhs: EinsumInfo#lhs.
  EinsumInfo: EinsumInfo#
  broadcast_operands: broadcast_operands().
  EinsumInfo.out: EinsumInfo#out.
  sanitize_shape: sanitize_shape().
  EinsumInfo._validate_binary_einsum: EinsumInfo#_validate_binary_einsum().
---
# Module: [`qwix/_src/core/einsum_info.py`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py)

## Classes
### `EinsumInfo`
- def: [`qwix/_src/core/einsum_info.py:27`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L27) — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
- doc: Info needed to perform a binary einsum using dot_general.
- signature: `class EinsumInfo:`
- members:
  - `_validate_binary_einsum(einsum_str: str)` — [`L82`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L82) — Validates that the einsum string is a supported binary operation. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `batch_chars(self)` — [`L102`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L102) — Returns the list of batch characters (present in lhs, rhs, and out). — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `contract_chars(self)` — [`L111`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L111) — Returns the list of contraction characters (in lhs, rhs but not out). — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `dimension_numbers(self)` — [`L120`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L120) — Returns dimension_numbers for dot_general compatibility. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `output_perm(self)` — [`L144`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L144) — Returns the output permutation if needed, or None. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `parse(cls, einsum_str: str, ndims: tuple[int, int] | None = None)` — [`L45`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L45) — Parses a binary einsum string into an EinsumInfo object. — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `lhs` — [`L40`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L40) — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `out` — [`L42`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L42) — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
  - `rhs` — [`L41`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L41) — documented in [qwix-_src-core-einsum_info](../../../../concepts/qwix-_src-core-einsum_info.md)
- used by: [`einsum`](../../contrib/padded_ptq.md#einsum), [`get_how_to_quantize`](einsum.md#get_how_to_quantize), [`_parse_einsum_str_for_lora`](../providers/lora.md#_parse_einsum_str_for_lora), [`_perform_binary_einsum`](einsum.md#_perform_binary_einsum)  (4 test-only)

## Functions
- `broadcast_operands(operands: Sequence[qarray.MaybeQArray], operand_subs_list: Sequence[str])` — [`L197`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L197) — Broadcasts operands to matching shapes for shared dimensions.
- `sanitize_shape(shape: Sequence[int | Any])` — [`L177`](../../../../../../../raw/code/qwix/qwix/_src/core/einsum_info.py#L177) — Replaces non-concrete integer dimensions with 1.

