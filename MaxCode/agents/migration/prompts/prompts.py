"""Prompts for code migration."""

JAX_BEST_PRACTICES = """
## JAX/Flax Best Practices (MUST follow)

1. **Use Flax Linen with @nn.compact**: Define all submodules inline inside
   `@nn.compact def __call__`. Do NOT use a separate `setup()` method or NNX.
2. **KV Cache**: Use pre-allocated fixed-size caches updated via
   `jax.lax.dynamic_update_slice`. NEVER grow the cache with `jnp.concatenate`
   or Python list appends -- that breaks XLA compilation.
3. **Causal Conv1d**: Use `padding="VALID"` with explicit left-padding
   (`jnp.pad` with `pad_width` on the time axis). Do NOT use `padding="SAME"`
   as it is non-causal and leaks future information. Implement as a standalone
   function with both prefill (full sequence) and decode (single-step with
   conv_state) paths. Use `jax.lax.conv_general_dilated` with
   `feature_group_count=channels` for depthwise convolution.
4. **Standalone Imports**: Only import from `jax`, `jax.numpy`, `flax.linen`,
   `numpy`, and `math`. Do NOT import from `torch`, `transformers`, or any
   PyTorch library in the output.
5. **Static Shapes**: All tensor shapes must be determinable at trace time for
   `jax.jit` compatibility. Avoid data-dependent shapes, Python loops over
   dynamic lengths, and boolean indexing with data-dependent masks.
   CRITICAL: Never use `array[..., :i]` where `i` is dynamic inside
   `jax.lax.scan` -- this creates dynamically-sized slices that break JIT.
6. **Variable Ordering**: Define every variable before its first use. No forward
   references -- JAX traces sequentially and undefined names cause errors.
7. **Attention Masking**: Use additive masking: add 0.0 where attention is
   allowed and a large negative value (e.g., -1e9 or `jnp.finfo(dtype).min`)
   where it should be blocked. Do NOT use multiplicative boolean masks.
8. **RMS Norm**: Implement as `x * jax.lax.rsqrt(mean(x^2) + eps) * weight`.
   Do NOT call `torch.nn.functional` or leave PyTorch API calls.
9. **Activation Functions**: Use `jax.nn.silu`, `jax.nn.gelu`, etc. Map
   `F.silu` -> `jax.nn.silu`, `F.gelu` -> `jax.nn.gelu`.
10. **Rotary Embeddings**: Precompute `cos` and `sin` tables. Apply as
    `(x * cos) + (rotate_half(x) * sin)`. Shapes must broadcast correctly.
11. **Triangular Matrix Inversions**: When the PyTorch code has a for-loop
    computing a Neumann series on a lower-triangular matrix (e.g., the WY
    representation in chunk-parallel delta rules), convert it to
    `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)`. This computes
    `(I - W)^{{-1}}` directly and is JIT-safe, unlike a scan with dynamic slicing.
12. **Interleaved Weight Ordering**: When the source code has a
    `fix_query_key_value_ordering` function or groups projections by key heads
    (e.g., when num_key_heads != num_value_heads), you MUST preserve this
    ordering exactly. Reshape to [B, T, num_k_heads, per_head_size] and split
    within each group. NEVER flatten to a single dimension and do a flat split
    -- this produces wrong tensors when num_k_heads != num_v_heads.

## CRITICAL: Faithfulness to Source Code

NEVER simplify complex tensor reshaping, reordering, or algorithmic patterns
from the source code. If the PyTorch code uses a specific interleaved weight
layout, chunk-parallel algorithm, or multi-step computation, convert it
faithfully to JAX. The RAG context shows EXAMPLES of similar patterns -- use
them as guidance for JAX idioms, but always follow the ACTUAL source code's
logic and structure.
"""

PYTORCH_TO_JAX_SINGLE_FILE_PROMPT = """You are an expert in JAX and PyTorch.
Your task is to convert the following PyTorch code to JAX.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
The PyTorch code to convert is as follows:
```python
{pytorch_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the PyTorch code above.
Ensure that the JAX code is idiomatic and follows best practices, such as using
pure functions and handling random number generation correctly with JAX's PRNG
keys. Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

PYTORCH_TO_JAX_REPO_PROMPT = """You are an expert in JAX and PyTorch. Your task
is to convert a repository from PyTorch to JAX. You will be given a file path
and the content of the file. You need to convert the given file from PyTorch to
JAX, considering its context within the repository.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
File path: {file_path}
File content:
```python
{pytorch_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the file content above.
Ensure that the JAX code is idiomatic and follows best practices, such as using
pure functions and handling random number generation correctly with JAX's PRNG
keys. The conversion should maintain compatibility with other files in the
repository, assuming they will also be converted to JAX.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

HF_TO_JAX_SINGLE_FILE_PROMPT = """You are an expert in JAX and PyTorch, with
special expertise in HuggingFace Transformers. Your task is to convert the
following HuggingFace Transformers code (which uses PyTorch) to JAX.
If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
The code is as follows:
```python
{code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the code above, using JAX libraries like
Flax if appropriate for transformer models. Ensure that the JAX code is
idiomatic and follows best practices, such as using pure functions and handling
random number generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

MODEL_CONVERSION_PROMPT = """You are an expert in JAX and PyTorch model
architectures. Your task is to convert the ENTIRE PyTorch file below to a
single JAX/Flax file. You MUST convert ALL classes, helper functions,
constants, and configuration dataclasses -- not just one class.

If it is helpful, you can use the following JAX code snippets as context for
functionality that might be similar to your conversion task:
---
{rag_context}
---
PyTorch model file:
```python
{pytorch_model_code}
```

IMPORTANT CONVERSION RULES:
1. Convert EVERY class and function in the file above. The output must include
   JAX equivalents for all nn.Module subclasses, all helper functions (rotary
   embeddings, attention masking, loss functions, etc.), and all supporting code.
2. If the source has a `fix_query_key_value_ordering` method or groups QKVZ
   projections by key heads, convert it FAITHFULLY. Reshape to
   [B, T, num_k_heads, ...] and split within each key-head group. Do NOT
   replace it with a flat split -- that produces wrong tensors when
   num_k_heads != num_v_heads.
3. If the source has a chunk-parallel delta rule with a for-loop computing a
   Neumann series (WY representation), convert it using
   `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)` instead of
   jax.lax.scan with dynamic slicing. See the RAG context for the pattern.
4. If the source has both a chunk (prefill) and recurrent (decode) mode for
   linear attention, implement BOTH modes and dispatch based on sequence length.
5. Implement causal_conv1d as a standalone function with both prefill and
   single-step decode paths.

Please think step by step about the conversion process before generating the code.
Then, provide the complete JAX equivalent of the entire file above.
Ensure that the JAX code is idiomatic and follows best practices for defining
models in JAX, such as using pure functions and handling random number
generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES
