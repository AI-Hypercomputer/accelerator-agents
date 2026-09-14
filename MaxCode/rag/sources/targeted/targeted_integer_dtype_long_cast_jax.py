"""
TARGETED RAG: Preserve .long() / .int() Integer Dtype Casts in JAX
====================================================================

When PyTorch code explicitly calls .long() (int64) or .int() (int32) on a
tensor, you MUST preserve the equivalent dtype cast in JAX. These casts
exist for a reason -- often for indexing, embedding lookups, or API
compatibility.

WRONG -- Omitting the .long() cast:
-------------------------------------
    # PyTorch source:
    #   positions = make_positions(input, padding_idx, left_pad)
    #   return new_tensor.masked_scatter_(mask, positions[mask]).long()

    # WRONG! Missing .long() -- returns int32 instead of int64
    def make_positions(tensor, padding_idx, left_pad):
        ...
        return jnp.where(mask, positions, tensor)

WHY THIS IS WRONG:
- .long() converts to int64 (torch.int64)
- Without the cast, positions may be int32, causing:
  1. Dtype mismatches when used as indices into int64-indexed arrays
  2. Overflow for very large sequence lengths or vocabularies
  3. Subtle bugs when comparing with other int64 tensors
- The source author explicitly added .long() for a reason

CORRECT -- Preserve the int64 cast:
-------------------------------------
    # CORRECT: .long() -> .astype(jnp.int64) or jnp.int64
    def make_positions(tensor, padding_idx, left_pad):
        ...
        return jnp.where(mask, positions, tensor).astype(jnp.int64)

PATTERN MATCHING:
-----------------
  PyTorch: `tensor.long()`       -> JAX: `tensor.astype(jnp.int64)`
  PyTorch: `tensor.int()`        -> JAX: `tensor.astype(jnp.int32)`
  PyTorch: `tensor.short()`      -> JAX: `tensor.astype(jnp.int16)`
  PyTorch: `tensor.float()`      -> JAX: `tensor.astype(jnp.float32)`
  PyTorch: `tensor.double()`     -> JAX: `tensor.astype(jnp.float64)`
  PyTorch: `tensor.half()`       -> JAX: `tensor.astype(jnp.float16)`
  PyTorch: `tensor.bfloat16()`   -> JAX: `tensor.astype(jnp.bfloat16)`
  PyTorch: `tensor.bool()`       -> JAX: `tensor.astype(jnp.bool_)`
  PyTorch: `tensor.to(dtype)`    -> JAX: `tensor.astype(dtype)`
  PyTorch: `tensor.type_as(ref)` -> JAX: `tensor.astype(ref.dtype)`

RULE: Every explicit dtype cast in PyTorch (.long(), .float(), .type_as(), etc.)
must have an equivalent .astype() in JAX. Never drop dtype casts.
"""
