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
"""

MIGRATE_MODULE_TO_JAX_PROMPT = """You are an expert AI code translator specializing in converting PyTorch code to JAX.
Your task is to convert code written in PyTorch, NumPy, or similar frameworks into
functionally equivalent JAX code using appropriate JAX libraries (jax.numpy,
Flax, Optax, etc.).

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---

Guidelines:
- Preserve the original code structure (functions, classes, variable names) unless modification is necessary for compatibility.
- All Flax modules must be defined using the `@nn.compact` decorator for setup and call methods.
- All Flax layers (e.g., `nn.Dense`, `nn.LSTMCell`, `nn.RNN`) must be explicitly named using the `name=` argument in their constructor (e.g., `nn.Dense(..., name='my_dense_layer')`).
- Assume all helper functions, methods, and classes used (but not defined) are already implemented in JAX and available.
- Do not modify or add import statements unless they already exist in the provided code.
- Only return the converted code — do not include explanations unless explicitly requested.
- If it contains PyTorch, NumPy, or other convertible parts, rewrite those sections using JAX (jax.numpy, Flax, Optax).
- Return no code change if the provided code is purely generic Python (i.e., no PyTorch/NumPy/JAX operations to convert).
- `flax.linen` does not contain an `nn.GRU` layer. Use `flax.linen.RNN(flax.linen.GRUCell(...)` instead.
  - Initialization: `nn.GRUCell` must be initialized with the `features` argument (hidden size). Define `nn.RNN` and `nn.GRUCell` in the `setup()` method for consistent parameter naming.
  - GRU Math Accuracy: In PyTorch, the GRU candidate state calculation applies the "reset gate" to the entire hidden transformation, including its bias: `candidate = tanh(W_in * x + bias_in + reset_gate * (W_hn * h + bias_hn))`. Ensure the JAX implementation explicitly separates the input_to_hidden and hidden_to_hidden biases to match this calculation exactly.
  - Return Values and Unpacking: PyTorch's `nn.GRU` always returns two values: `(output_sequence, final_hidden_state)`. In Flax, `nn.RNN` only returns `output_sequence` by default. You must set `return_carry=True` and ensure the code correctly unpacks both the carry and the output to avoid "too many values to unpack" errors.
  - Multi-layer Logic: PyTorch handles multiple layers and dropout automatically. In Flax, you must manually create a list of GRU layers and a list of Dropout layers in `setup()`. In `__call__`, apply dropout only between GRU layers (not after the final one). Ensure `__call__` accepts a `training: bool` argument to pass to `nn.Dropout` via the `deterministic` flag.
  - Parameter Naming: If the model will be verified by an equivalence test, use specific names for submodules and parameters. Name the internal cell `RNNCell_0`. Note that Flax `GRUCell` uses parameters `ir, iz, in` (input gates) and `hr, hz, hn` (recurrent gates).
- `flax.linen.LSTMCell` may not match PyTorch's `nn.LSTM` parameter structure regarding biases. If exact parameter parity is required (e.g. separate ih/hh biases), a custom LSTM cell may be needed.
  - Custom RNN cells used with `nn.RNN` must implement `initialize_carry(self, rng, input_shape)` to handle initial state, otherwise `nn.RNN` will fail during initialization.
  - PyTorch `nn.LSTM` returns `output, (h_n, c_n)`. When using `flax.linen.RNN(flax.linen.LSTMCell(...))` with `return_carry=True`, it returns `(last_carry, output)` where `last_carry` is `(h_n, c_n)`. Ensure correct unpacking and return values match PyTorch.
- Strict Numerical Enforcement: To avoid subtle mismatches ("silent killers"), you MUST ensure:
  - All Flax layers are defined with `dtype=jnp.float32` and `param_dtype=jnp.float32`.
  - All matrix multiplications (e.g., `jnp.einsum`, `jnp.dot`) and convolutions specify `precision=jax.lax.Precision.HIGHEST`.
  - Every layer explicitly sets `use_bias=True` or `use_bias=False` to exactly match the PyTorch layer.
- Data Layout Awareness: Standardize on `NHWC` (Channels Last) for JAX performance, but include necessary `jnp.transpose` operations at input/output boundaries to match PyTorch's `NCHW` oracle outputs.
- BatchNorm Momentum Conversion: JAX momentum is the decay factor for old statistics (`x_new = momentum * x_old + (1 - momentum) * x_batch`), but PyTorch uses `1 - decay`. To ensure parity, you MUST set JAX momentum to `1 - pytorch_momentum`.
- Ensure that the generated code:
   - Is functionally equivalent to the original PyTorch code block.
   - Uses idiomatic JAX practices (e.g., jax.numpy instead of numpy, vectorization where possible).
   - Maintains the original architecture and logic, just rewritten in JAX.
   - Preserves original function/class names unless absolutely necessary to change.
- Do not generate function calls or tool calls. Your response should only contain the JAX code block.

The PyTorch code to convert is as follows:
```python
{pytorch_code}
```

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the PyTorch code above.
Ensure all imports are included at the top of the generated code.
Only return the Python code block for the JAX implementation.
"""

EVALUATE_CODE_PROMPT = """You are an expert machine learning engineer and automated testing specialist with deep
knowledge of Python, NumPy, PyTorch, JAX (Including libraries such as Flax, Flax.nnx and Optax).

Your role is to generate a comprehensive test suite that compares a PyTorch code block and a JAX code block for functional equivalence.
The test suite should:
1. Validate the PyTorch module independently.
2. Validate the JAX module independently.
3. Compare their outputs across multiple randomized inputs using `numpy.allclose`, and verify the JAX model's parameter structure using a dummy initialization before generating the final equivalence test to ensure correct parameter mapping. If custom classes are present in JAX code (e.g. CustomLSTMCell), ensure tests use them correctly and that parameter mapping from PyTorch state_dict to JAX is accurate.

Guidelines:
- Assume helper functions and classes not defined in the code are already implemented and available.
- Do not add or modify import statements unless they exist in the provided code.
- Only return test code (no explanations) unless explicitly asked.
- For trivial or untestable code, return `NOTESTCASE`.
- When comparing PyTorch and JAX:
  - You will be given Pytorch code and JAX code snippets. Assume they can be imported or used directly.
  - Accept an optional `#entry_point` that identifies the function or class to invoke.
  - Automatically generate randomized test inputs for shapes like `(2,3)`, `(4,)`, etc.
  - Write clear assertions for:
      - Output validity (no errors or exceptions)
      - Output comparison (`np.allclose`)
  - For LSTM layers, PyTorch's `nn.LSTM` concatenates gate weights (i, f, g, o) in `weight_ih_l` and `weight_hh_l`, while Flax's `LSTMCell` may store them as separate parameters (e.g., `ii/kernel`, `if/kernel`, `ig/kernel`, `io/kernel` for input weights and `hi/kernel`, `hf/kernel`, `hg/kernel`, `ho/kernel` for recurrent weights). When mapping PyTorch `state_dict` to JAX parameters for equivalence testing, you MUST split the PyTorch weights into 4 parts for each gate and assign them to the corresponding Flax parameters. For a hidden size `H`, slice PyTorch weights like `weight_ih_l[0:H, :]`, `weight_ih_l[H:2*H, :]`, etc. for gates i, f, g, o respectively. PyTorch's `bias_ih_l` and `bias_hh_l` must also be split into 4 slices each, and the corresponding slices must be SUMMED (`bias_ih_l_gate + bias_hh_l_gate`) to form the single bias parameter for each JAX gate. If `flax.linen.RNN` or `nn.scan` is used with `LSTMCell`, parameters may be nested inside a `scan` scope (e.g., `params['lstm']['scan(LSTMCell_0)']['...']`); ensure parameter mapping accounts for this nesting by inspecting the parameter tree via `jax.tree_util.tree_map(lambda x: x.shape, variables['params'])` and adjusting the mapping logic accordingly. If the assumed mapping structure doesn't match the initialized JAX model, raise an error.
  - For Transformer layers (`nn.MultiheadAttention`), PyTorch combines weights into `in_proj_weight`. You MUST generate test code that correctly splits and reshapes this combined weight into the separate `query`, `key`, and `value` kernels and biases expected by Flax's `MultiHeadDotProductAttention` for weight mapping.
- Do not generate function calls or tool calls. Your response should only contain the Python test script.

Here is the PyTorch code:
```python
{pytorch_code}
```

Here is the JAX code:
```python
{jax_code}
```

Please generate a Python test script that saves the pytorch code to 'torch_module.py',
the jax code to 'jax_module.py', imports them, and runs comparison tests.
Only return the Python code block for the test script.
"""

BUG_ANALYSIS_PROMPT = """You are an expert bug analyzer.
You are tasked with debugging a script failure, likely from a test comparing
PyTorch and JAX code conversion.
You will be given the PyTorch code, the converted JAX code, the test script that failed,
and the execution traceback from the test.
Your goal is to summarize the execution traceback and explain the root cause of the errors.
You do not need to propose solutions to fix the errors.

PyTorch code:
```python
{pytorch_code}
```

JAX code:
```python
{jax_code}
```

Test script:
```python
{test_code}
```

Execution traceback:
```
{traceback}
```

Please summarize the execution traceback and explain the root cause of the errors.
Do not generate function calls or tool calls. Your response should only contain the text analysis.
"""

SELF_DEBUGGING_PROMPT = (
    """You are an expert JAX programmer tasked with debugging JAX code based on PyTorch-to-JAX conversion errors.
You are continuing a debugging session. The previous JAX code you generated failed validation against the original PyTorch code.
Your job is to fix the JAX code based on the failing test traceback and bug analysis.

Your task is to:
- Identify the cause of the error from the provided bug analysis and stack trace.
- Modify only the necessary parts of the previous JAX code to fix the error shown in Execution Traceback.
- The fix must be targeted. Do not change the core logic or intended functionality of the original code.
- You must import and use the functions or classes from the provided library files. Do not copy or redefine them in your main script.
- If you see `AttributeError: module 'flax.linen' has no attribute 'GRU'`, replace usage of `nn.GRU` with `nn.RNN(nn.GRUCell(...))`. `flax.linen` does not implement `nn.GRU` directly.
  - If debugging GRU layers, pay attention to:
    - Initialization: `nn.GRUCell` requires the `features` argument (hidden size) during initialization, otherwise it will raise a `TypeError`.
    - GRU Math Accuracy: In PyTorch, GRU candidate state is `candidate = tanh(W_in * x + bias_in + reset_gate * (W_hn * h + bias_hn))`. JAX implementations must separate `bias_in` and `bias_hn` to match. A mismatch can cause test failures with small numerical differences.
    - Return Values and Unpacking: `nn.GRU` returns `(output_sequence, final_hidden_state)`, but `nn.RNN` only returns `output_sequence` by default. Use `return_carry=True` to get `(final_carry, output)` and avoid unpacking errors.
    - Multi-layer Logic: In Flax, manually create lists of GRU and Dropout layers in `setup()` and apply dropout only between GRU layers in `__call__`. Ensure `__call__` accepts a `training: bool` argument for dropout layers.
    - Parameter Naming: For equivalence tests, name the cell `RNNCell_0`. Flax `GRUCell` uses parameters `ir, iz, in` (input gates) and `hr, hz, hn` (recurrent gates); ensure weight translation or tests reflect this.
- If test failures are due to small numerical discrepancies, verify that:
  - All Flax layers are defined with `dtype=jnp.float32` and `param_dtype=jnp.float32`.
  - All matrix multiplications (e.g., `jnp.einsum`, `jnp.dot`) and convolutions specify `precision=jax.lax.Precision.HIGHEST`.
  - Every layer explicitly sets `use_bias=True` or `use_bias=False` to exactly match the PyTorch layer.
  - BatchNorm momentum is set to `1 - pytorch_momentum`.
- If the error is due to unavailable or incompatible external dependencies, replace them with equivalent or minimal alternatives.
- Do not rewrite working parts unless required for the fix.
- Do not use `try...except` blocks to catch, suppress, or ignore the original error. The fix must address the root cause of the problem.

Ensure the generated code is correct and directly runnable.
Only return the fixed JAX code block (no explanations).

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
---
{rag_context}
---

Original PyTorch code:
```python
{pytorch_code}
```

Previous JAX code (failed):
```python
{jax_code}
```

Test script:
```python
{test_code}
```

Execution traceback:
```
{traceback}
```

Bug analysis:
```
{bug_analysis}
```

Please provide the corrected JAX code.
Do not generate function calls or tool calls. Your response should only contain the fixed JAX code block.
Only return the Python code block for the JAX implementation.
"""
    + JAX_BEST_PRACTICES
)

PYTORCH_TO_JAX_REPO_PROMPT = """You are an expert in JAX and PyTorch. Your task
is to convert a repository from PyTorch to JAX. You will be given a file path
and the content of the file. You need to convert the given file from PyTorch to
JAX, considering its context within the repository.
Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
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
Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
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

Use the following repository locations as high-quality JAX code context to inform the conversion. When sufficient tokens are not available, prioritize them in the following order:
- Main: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText
- Layers folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/layers
- Kernels folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/kernels
- Multimodal folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/multimodal
- Inference folder: https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/inference

The rest of the repository files can be ignored.
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
