"""
TARGETED RAG: Preserve Buffer Dtypes When Converting register_buffer to JAX
=============================================================================

When converting PyTorch's register_buffer() to JAX, you MUST preserve the
exact dtype of the buffer tensor. torch.Tensor() creates float32 by default,
torch.LongTensor() creates int64, etc.

WRONG -- Changing buffer dtype during conversion:
---------------------------------------------------
    # PyTorch source:
    #   self.register_buffer('version', torch.Tensor([2]))
    #   # torch.Tensor([2]) creates a float32 tensor containing [2.0]

    # WRONG! Changed dtype from float32 to int32
    self.sow('buffers', 'version', jnp.array([2], dtype=jnp.int32))

WHY THIS IS WRONG:
- torch.Tensor([2]) creates float32, NOT int32
- Changing the dtype means the buffer has different bit representation
- Code that checks buffer dtype or uses it in float operations will break
- State dict comparison tools will flag the dtype mismatch

CORRECT -- Match the exact PyTorch dtype:
-------------------------------------------
    # PyTorch: torch.Tensor([2])  ->  float32
    # CORRECT: preserve float32 dtype
    self.sow('buffers', 'version', jnp.array([2.0], dtype=jnp.float32))

DTYPE REFERENCE for torch tensor constructors:
------------------------------------------------
  torch.Tensor([...])       -> float32  ->  jnp.array([...], dtype=jnp.float32)
  torch.FloatTensor([...])  -> float32  ->  jnp.array([...], dtype=jnp.float32)
  torch.DoubleTensor([...]) -> float64  ->  jnp.array([...], dtype=jnp.float64)
  torch.HalfTensor([...])   -> float16  ->  jnp.array([...], dtype=jnp.float16)
  torch.LongTensor([...])   -> int64    ->  jnp.array([...], dtype=jnp.int64)
  torch.IntTensor([...])    -> int32    ->  jnp.array([...], dtype=jnp.int32)
  torch.BoolTensor([...])   -> bool     ->  jnp.array([...], dtype=jnp.bool_)
  torch.tensor([...])       -> inferred ->  match the inferred dtype
  torch.zeros(N)            -> float32  ->  jnp.zeros(N, dtype=jnp.float32)
  torch.ones(N)             -> float32  ->  jnp.ones(N, dtype=jnp.float32)

REGISTER_BUFFER conversion patterns:
--------------------------------------
  # PyTorch:
  self.register_buffer('name', torch.Tensor([2]))
  # JAX (using sow for mutable state):
  self.sow('buffers', 'name', jnp.array([2.0], dtype=jnp.float32))

  # PyTorch:
  self.register_buffer('mask', torch.ones(seq_len, seq_len).triu(1).bool())
  # JAX (using variable for persistent state):
  mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.float32), k=1).astype(jnp.bool_)

RULE: Every buffer's dtype must match the PyTorch source exactly.
torch.Tensor() is float32, not int32. Always check the constructor.
"""
