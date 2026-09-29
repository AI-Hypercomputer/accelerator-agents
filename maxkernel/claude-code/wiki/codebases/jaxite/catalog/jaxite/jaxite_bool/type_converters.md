---
title: 'Module: jaxite/jaxite_bool/type_converters.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/type_converters.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.type_converters`/
symbols:
  bit_slice_to_u8: bit_slice_to_u8().
  u8_to_bit_slice: u8_to_bit_slice().
  u8_list_to_bit_slice: u8_list_to_bit_slice().
  bit_slice_to_bytes: bit_slice_to_bytes().
  str_to_cleartext: str_to_cleartext().
  cleartext_to_str: cleartext_to_str().
  bit_slice_to_uint: bit_slice_to_uint().
  uint_to_bit_slice: uint_to_bit_slice().
---
# Module: [`jaxite/jaxite_bool/type_converters.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py)

## Functions
- `bit_slice_to_bytes(bit_slice: List[bool])` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L47) — Given a bitwise representation, returns an ASCII bytes object.
- `bit_slice_to_u8(bit_slice: List[bool])` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L23) — Given a bit slice of length 8, returns a base-10 int representation.
- `bit_slice_to_uint(bit_slice: List[bool], num_bits: int)` — [`L6`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L6) — Given a bit slice of num_bits, returns a base-10 int representation.
- `cleartext_to_str(cleartext: List[bool])` — [`L73`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L73)
- `str_to_cleartext(text: str, static_len: Optional[int] = None, padding_byte: bytes = bytes(' ', 'ascii'))` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L62)
- `u8_list_to_bit_slice(input_list: List[int])` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L37) — Given a list of u8 values, returns a flattened bitwise representation.
- `u8_to_bit_slice(input_int: int)` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L30) — Given an integer [0, 255], returns a bitwise representation.
- `uint_to_bit_slice(input_int: int, num_bits: int)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/type_converters.py#L15) — Given an integer [0, 2**num_bits - 1], returns a bitwise representation.

