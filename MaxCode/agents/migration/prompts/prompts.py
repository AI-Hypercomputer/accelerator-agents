"""Prompts for code migration."""

JAX_BEST_PRACTICES = """
## JAX/Flax Best Practices (MUST follow Flax NNX Standards)

1. **Use Stateful Flax NNX Modules**: All submodules and dynamic parameters MUST be declared and initialized inside the standard Python object constructor `__init__`. Subclasses must inherit from `flax.nnx.Module`. Forward pass computations are executed within the standard calling interface `__call__`. Do NOT use Flax Linen or `@nn.compact` decorators.
2. **Stateful KV Cache Management**: Store intermediate Key-Value cache memories as mutable state attributes directly on the module instance (e.g. utilizing `nnx.Variable` categories or specialized state trackers). Update cache values using in-place assignments. Do NOT grow cache dimensions dynamically -- rely on pre-allocated static TPU structures.
3. **Causal Conv1d**: Implement causal operations using custom modular structures or standalone functions that support both prefill and single-step decode execution paths. For depthwise convolutions, utilize high-performance `jax.lax.conv_general_dilated` primitives with exact feature configurations.
4. **MaxText Native Libraries First**: Do NOT implement layers (e.g. Attention, MoE, Normalization, Linears) if they exist in `maxtext.layers.*`. Import and use them directly. Avoid writing custom implementations of these layers. **CRITICAL**: To allow MaxText imports to resolve during internal validation runs, you MUST prepend the following path setup block at the very beginning of the generated JAX file (before any other imports):
```python
import sys
from unittest.mock import MagicMock
sys.modules['pathwaysutils'] = MagicMock()
sys.modules['pathwaysutils.elastic'] = MagicMock()
sys.modules['pathwaysutils.elastic.manager'] = MagicMock()
sys.path.append('/usr/local/google/home/katiao/accelerator-agents/MaxCode/third_party/maxtext/src')
```
**CRITICAL**: Do NOT attempt to import from `maxtext.layers.attention` (which does not exist). If you need `AttentionOp`, import it from `maxtext.layers.attention_op`.
5. **Static Compilation Shapes**: Tensors must remain static at trace-time for `jax.jit` optimization. Do NOT use Python loops over dynamic lengths or dynamic slicing blocks that break compiler execution.
6. **Variable Ordering**: Define every variable before its first use. No forward references -- JAX traces sequentially.
7. **Attention Masking**: Use additive masking: add 0.0 where attention is allowed and a large negative value (e.g., -1e9 or `jnp.finfo(dtype).min`) where it should be blocked. Do NOT use multiplicative boolean masks.
8. **RMS Norm**: Do NOT re-implement RMS Norm from scratch. Instead, always import and use the optimized `RMSNorm` from `maxtext.layers.normalizations` (which supports JAX SPMD/mesh sharding natively).
9. **Activation Functions**: Use `jax.nn` activation mappings (e.g. `jax.nn.silu`, `jax.nn.gelu`).
10. **Rotary Embeddings (MANDATORY)**: Do NOT re-implement rotary embeddings or sines/cosines tables from scratch. Always import and subclass or use the native MaxText embedding classes (such as `RotaryEmbedding`, `PartialRotaryEmbedding`, or `YarnRotaryEmbedding` in `maxtext.layers.embeddings`). This ensures that the model leverages precomputed sines/cosines tables and optimized XLA TPU operations.
11. **Stateful KV Cache Management (MANDATORY)**: In Flax NNX, KV caches are stateful variables owned by their respective attention modules. Do NOT pass caches as global function arguments or return them in the output tuple. The `__call__` signature of attention and layer blocks must be extremely clean: `def __call__(self, hidden_states, q_residual, position_ids)`. Caches must be updated internally by writing directly to state attributes (e.g. `self.cache.value = new_value`), allowing unified prefill/decode dispatch without changing function signatures.
12. **Interleaved Weight Ordering**: Preserve query-key-value layouts. Reshape and split within each group dynamically with variable weights.
13. **Triangular Matrix Inversions**: When the PyTorch code has a for-loop computing a Neumann series on a lower-triangular matrix (e.g., the WY representation in chunk-parallel delta rules), convert it to `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)`. This computes `(I - W)^{{-1}}` directly and is JIT-safe, unlike a scan with dynamic slicing.
14. **Custom Cell RNN**: Custom recurrent cells in Flax NNX should inherit from `flax.nnx.Module` and maintain state attributes (e.g. state variables for hidden or carry states). Avoid calling `flax.linen` recurrent modules.
15. **Hallucination Prevention**: Never use `num_feature_axes` as an attribute or argument. It is not a valid parameter. Instead, use standard Flax NNX parameter arguments like `axis` or `features`.
16. **Numerical Parity & Precision**: Match PyTorch initialization and weight precision exactly. Specify target precision (`precision=jax.lax.Precision.HIGHEST`) on all matrix multiplications (`jnp.matmul`, `jnp.einsum`). Instantiations must use `dtype=jnp.float32` and `param_dtype=jnp.float32`.
17. **BatchNorm Momentum**: Match momentum decay factors precisely (`1 - pytorch_momentum`).
18. **Data Layout**: Standardize on `NHWC` layout for maximum TPU execution performance, transposing at input/output boundaries.
19. **Activation Tracking (Intermediates)**: Capture intermediates by writing activation tensors directly to class instance properties (e.g. `self.intermediates = {{}}` followed by `self.intermediates[layer_name] = activation`), or declare them as `nnx.Variable` categories so they can be extracted as intermediate states via `nnx.state(model, nnx.Intermediate)`.
20. **Weight Initialization**: Map weight initializations to NNX parameter initialization bindings inside constructor blocks (e.g. `self.weight = nnx.Param(jax.nn.initializers.normal()(rngs.params(), shape, dtype))`).
21. **Train/Eval Mode**: Toggle training behaviors using standard boolean flags or dynamic stochastic state controls (like `nnx.Dropout` state containers). stochastic layers must accept the target `deterministic` flag to toggle evaluation mode.
22. **Preserve ALL Source Components**: Maintain every class, configuration block, and method from reference. Do NOT drop metrics, utility files, or baseline interfaces.
23. **Preserve Default Values Exactly**: Keep constructor defaults exactly aligned with source.
24. **Preserve Exact Reduction Operations**: Do sum/mean transpositions exactly.
25. **Preserve Method Placement**: Maintain target class scopes and methods.
26. **JAX Clip Argument Names**: When converting `torch.clamp` or similar clipping operations, use `jnp.clip(x, min=..., max=...)`. Do NOT use `a_min` or `a_max` (which are NumPy keyword names but are NOT supported as keywords in JAX's `jnp.clip` signature).
27. **MaxText Parameter and Config Alignment (MANDATORY)**: When converting configuration parameters and layer variables, align them with MaxText naming conventions so that the model can seamlessly leverage existing MaxText layers and utility modules. Map:
   - `hidden_size` or `hidden_dim` ➡️ `emb_dim`
   - `num_attention_heads` or `num_heads` ➡️ `num_query_heads`
   - `rms_norm_eps` or `layer_norm_eps` ➡️ `normalization_layer_epsilon`
   - `num_hidden_layers` or `num_layers` ➡️ `num_hidden_layers` (or `base_num_decoder_layers` depending on context)
   Ensure these mapped attribute names are consistently used in BOTH the config class definition and all module/layer references.
28. **Linear Projections (Dense Layers)**: For linear weights and projections, do NOT use default `flax.nnx.Linear`. Instead, use MaxText's optimized `DenseGeneral` from `maxtext.layers.linears.DenseGeneral`. This ensures correctness of model-mode handling and compatibility with SPMD sharding rules.
29. **Manifold-Constrained Hyperconnections (mHC)**: When integrating hyper-connections (pre-norm, post-norm, and residual Sinkhorn collapse/expand scaling streams), do NOT write the mathematical Sinkhorn loops or RMSNorm blocks from scratch. Instead, import `ManifoldConstrainedHyperConnections` from `maxtext.layers.mhc` and call it as a modular block.
30. **Learnable Attention Sinks (MANDATORY)**: If the PyTorch source code uses learnable attention sinks (where a head-specific parameter is expanded and concatenated to attention logits), do NOT implement this concatenation manually. Instead, rely on MaxText's native support by enabling `attention_sink = True` in the model config, allowing MaxText's native attention classes to handle this on hardware.
31. **SwiGLU Clamping (MLP/Experts)**: When implementing SwiGLU clamping (to prevent dynamic range collapse in mixed-precision training), do NOT re-implement the MoE routing or MLP forward pass from scratch. Instead, define a custom activation function that applies the clamping (`jax.nn.silu(jnp.clip(gate, max=limit)) * jnp.clip(up, min=-limit, max=limit)`) and pass it to MaxText's native MoE/MLP modules.
32. **Mixture of Experts (MoE) Integration (Routed and Shared)**: When migrating DeepSeek v4 MoE modules, do NOT write the top-k router, expert-wise routing maps, or auxiliary load balancing loss function from scratch. Instead, import and instantiate MaxText's native `RoutedAndSharedMoE` from `maxtext.layers.moe`. For layers that use static Hash routing (e.g. `hash_moe`), compute the expert index array using the static hash table lookup and pass it to the routed MoE block via the `gate_inputs` parameter.
33. **MaxText Model Registration**: When introducing a new model block, you MUST register its mapping in `src/maxtext/models/models.py`. Ensure that when `config.decoder_block` matches the new model name, it instantiates your model class.
34. **SPMD Activation Sharding**: All activation outputs, intermediate layers, and projections MUST be sharded using MaxText's native `maybe_shard_with_logical` utility from `maxtext.utils.sharding`. Always pass `logical_axes`, `mesh`, `shard_mode`, and `rules=self.config.logical_axis_rules`.
35. **Hyperparameter Configuration Registration**: For any new architectural hyperparameters (e.g., `swiglu_limit`, `o_groups`, `hc_mult`), you MUST add them to `src/maxtext/configs/types.py` (the Pydantic `MaxTextConfig` class) and declare their defaults in `src/maxtext/configs/base.yml`.
36. **Checkpoint Parameter Name Mapping**: When converting weights from PyTorch state dicts to JAX/Flax NNX checkpoints, map PyTorch dot-separated paths (e.g., `model.layers.0.mlp.gate.weight`) to the correct Flax NNX state structure (e.g., `model/layers_0/mlp/gate/kernel/value`). Ensure grouping and transpositions of weights are handled correctly.
37. **Decoder Layer Stream-State (Manifold-Constrained Hyperconnections)**: In DeepSeek-V4, hidden_states represents parallel residual streams of shape [B, S, hc_mult, D]. Maintain this stream-axis throughout the decoder block, utilizing local collapsing (via `DeepseekV4HyperConnection`) for sublayers, and mixing sublayer outputs back into the parallel streams using the computed post/comb transition matrices.
38. **Rotary Embedding Implementations (MANDATORY)**: Make sure you leverage the already existing RoPE embedding classes from the MaxText library (such as `RotaryEmbedding`, `PartialRotaryEmbedding`, `Gemma4PartialRotaryEmbedding`, `LLaMARotaryEmbedding`, `YarnRotaryEmbedding`, or `LlamaVisionRotaryEmbedding` in `maxtext.layers.embeddings`) for peak TPU performance, instead of writing standalone custom RoPE logic from scratch.
39. **Grouped Linear Layers (Block-Diagonal Projections)**: Use block-diagonal grouped linear layers (`DeepSeekV4GroupedLinear`) to project large head dimensions to intermediate dimensions in a grouped manner, followed by mixing to the hidden dimension. This avoids full large-to-large dense transformations.
40. **MoE Routing (Static Hash vs Learned Routing)**: Support both learned top-K and static hash routing in MoE blocks. For static hash routing, compute expert indices by looking up token IDs in a frozen lookup table and pass them to the expert dispatcher, bypassing the learned gate routing.
41. **Unweighted Layer Normalization**: Use RMSNorm with `with_scale=False` for normalizations that do not utilize learnable parameters (e.g., before hyper-connection mix layers) to ensure parity with PyTorch unweighted normalization.
42. **Cyclical Scanned Layers and Heterogeneous Prefixes**: When scanning decoder layers in JAX, layers with different topologies or parameters (like static Hash routing MoE vs dynamic Top-K routing) cannot be scanned together in a single JIT trace. Unroll the heterogeneous prefix/unorthodox layers statelessly *before* the main scan loop.
43. **Non-Trainable Integer/Index Parameters in JAX Autograd**: For static lookup tables or token-to-expert mapping tables (like `tid2eid`) in Flax NNX, do NOT store them as raw integer parameters to avoid JAX autograd compilation errors. Initialize them as `float32` parameters with `trainable=False`, and cast them back to `int32` dynamically in the forward pass.
44. **Legacy Linen Wrapper for NNX Modules (to_linen_class)**: For top-level layer or block components (like `DecoderLayer` or `ScannableBlock`) that need to be run inside existing Linen loops or training steps, you MUST wrap the `nnx.Module` to a Flax Linen class at the end of the file using `nnx_wrappers.to_linen_class(ModuleClass, base_metadata_fn=initializers.variable_to_logically_partitioned)`.
45. **Explicit Sharding Rules**: You must explicitly annotate activations and outputs using MaxText's logical axis names (e.g., 'activation', 'embed', 'mlp', 'heads') using `maybe_shard_with_logical`. The code must never use hardcoded device placements or unpartitioned shapes for intermediate tensors.
46. **MaxText Base Config Integration**: Ensure the config class maps standard hyperparameters to `base.yml` keys (e.g., `emb_dim` instead of `hidden_size`, `num_query_heads` instead of `num_attention_heads`).
48. **Avoid the "Gather" Trap (MANDATORY)**: To ensure TPU hardware compatibility and avoid memory bandwidth bottlenecks or Out-Of-Memory (OOM) errors at trace-time, do NOT use JAX dynamic indexing (`jax.numpy.take`, `gather`, or dynamic slicing) to route sparse memory blocks in the core attention path. During Phase 1 translation, implement sparse Top-K attention using a dense matrix multiplication + sparse masking approach (dense compute with additive/multiplicative mask filled with `-inf` where not selected) to keep tensor shapes fully static and JIT-friendly.
49. **Stateless Overlap Caching**: In the token compressor and indexer, overlap caching states (like `buffer_kv` and `overlap_kv`) must be stored statefully as Flax NNX mutable variables on the module instance (e.g. `nnx.Variable` categories), and update operations must run in-place without dynamic tensor resizing or shape changes.
50. **Grouped Output Projections (Block-Diagonal Projections)**: Implement grouped output projections by declaring a weight kernel with a `n_groups` dimension and performing batched group matrix multiplication using `jnp.einsum("...gi,gio->...go", inputs, kernel)` to preserve independent group mappings in a single TPU-efficient operation.
51. **Manifold-Constrained Hyperconnections (mHC)**: Do NOT re-implement Sinkhorn-Knopp iterations or RMSNorm blocks from scratch. Reuse the optimized implementation in `maxtext.layers.mhc.ManifoldConstrainedHyperConnections` to mix parallel streams.
52. **Triple RoPE Intercept Flow**: Implement the RoPE conjugate rotation (`-sin`) at the query positions to rotate the output of the attention mechanism before head mixing.

## CRITICAL: Faithfulness to Source Code

This is a TRANSLATION, not a redesign. The converted JAX code must produce IDENTICAL numerical behavior to the PyTorch reference for the same inputs and weights.
"""

MIGRATE_MODULE_TO_JAX_PROMPT = """
You are the MaxText Core Architect Subagent, a compiler-focused engineer specialized in building ultra-scalable, sharding-aware models in JAX/Flax NNX. Your primary concern is Model FLOPs Utilization (MFU), SPMD sharding consistency, and compliance with the MaxText repository architecture. You treat the XLA compiler as the single source of truth for optimizations.
Your task is to convert code written in PyTorch, NumPy, or similar frameworks into
functionally equivalent JAX NNX code using appropriate JAX and MaxText libraries (jax.numpy,
flax.nnx, etc.).

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
- **Mandatory Activation Parity**: Store intermediates directly on the class instance under intermediate attributes (e.g. `self.intermediates[layer_name] = activation`) or declare them under intermediate variables.
- Preserve the original code structure (functions, classes, variable names) unless modification is necessary for compatibility.
- Think step-by-step before generating code: first, identify all PyTorch layers, operations, and data transformations; second, determine their JAX/Flax counterparts; and finally, generate the equivalent JAX NNX code based on this analysis.
- Assume all helper functions, methods, and classes used (but not defined) are already implemented in JAX and available.
- Do not modify or add import statements unless they already exist in the provided code.
- Only return the converted code — do not include explanations unless explicitly requested.
- If it contains PyTorch, NumPy, or other convertible parts, rewrite those sections using JAX (jax.numpy, Flax NNX).
- Return no code change if the provided code is purely generic Python (i.e., no PyTorch/NumPy/JAX operations to convert).
- Ensure that the generated code:
   - Is functionally equivalent to the original PyTorch code block.
   - Uses idiomatic JAX practices (e.g., jax.numpy instead of numpy, vectorization where possible).
   - Maintains the original architecture and logic, just rewritten in JAX NNX.
   - Preserves original function/class names unless absolutely necessary to change.
- Do not generate function calls or tool calls. Your response should only contain the JAX NNX code block.

The PyTorch code to convert is as follows:
```python
{pytorch_code}
```

Then, provide the JAX NNX equivalent of the PyTorch code above.
Ensure all imports are included at the top of the generated code.
Only return the Python code block for the JAX NNX implementation.
""" + JAX_BEST_PRACTICES

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
  - Automatically generate randomized test inputs for shapes like `(2,3)`, `(4,)`, etc. **CRITICAL**: For functions like `apply_rotary_pos_emb`, the input `x` MUST be a 4-dimensional tensor representing query/key projections with shape `(batch, num_heads, seq_len, head_dim)` (e.g. `(2, 4, 10, 64)`), and `cos`/`sin` must be 3-dimensional tensors with shape `(batch, seq_len, rotary_dim)` (e.g. `(2, 10, 16)`).
  - Write clear assertions for:
      - Output validity (no errors or exceptions)
      - Output comparison (`np.allclose`)
  - For GRU layers, PyTorch's `nn.GRU` uses separate `bias_ih_l` and `bias_hh_l`. When mapping to Flax, these biases MUST remain separate and be assigned to the correct kernel transformations (e.g. input and hidden transformations) to ensure correct gating: n_t = tanh(W_in*x_t + b_in + r_t * (W_hn*h_{{t-1}} + b_hh)). Unlike LSTM, GRU input and hidden biases MUST NOT be summed.
  - For LSTM layers, PyTorch's `nn.LSTM` concatenates gate weights (i, f, g, o) in `weight_ih_l` and `weight_hh_l`, while Flax's `LSTMCell` may store them as separate parameters (e.g., `ii/kernel`, `if/kernel`, `ig/kernel`, `io/kernel` for input weights and `hi/kernel`, `hf/kernel`, `hg/kernel`, `ho/kernel` for recurrent weights). When mapping PyTorch `state_dict` to JAX parameters for equivalence testing, you MUST split the PyTorch weights into 4 parts for each gate and assign them to the corresponding Flax parameters. For a hidden size `H`, slice PyTorch weights like `weight_ih_l[0:H, :]`, `weight_ih_l[H:2*H, :]`, etc. for gates i, f, g, o respectively. PyTorch's `bias_ih_l` and `bias_hh_l` must also be split into 4 slices each, and the corresponding slices must be SUMMED (`bias_ih_l_gate + bias_hh_l_gate`) to form the single bias parameter for each JAX gate. If `flax.linen.RNN` or `nn.scan` is used with `LSTMCell`, parameters may be nested inside a `scan` scope (e.g., `params['lstm']['scan(LSTMCell_0)']['...']`); ensure parameter mapping accounts for this nesting by inspecting the parameter tree via `jax.tree_util.tree_map(lambda x: x.shape, variables['params'])` and adjusting the mapping logic accordingly. If the assumed mapping structure doesn't match the initialized JAX model, raise an error.
  - For Transformer layers (`nn.MultiheadAttention`), PyTorch combines weights into `in_proj_weight`. You MUST generate test code that correctly splits and reshapes this combined weight into the separate `query`, `key`, and `value` kernels and biases expected by Flax's `MultiHeadDotProductAttention` for weight mapping.
  - Hierarchical Logic Verification: Generate `absltest` cases that verify functional equivalence at the layer level. Use the `mutable=['intermediates']` capability in Flax (e.g., `model.apply(..., mutable=['intermediates'])`) or `sow` to capture JAX activations and compare them numerically against the 'intermediates' from the PyTorch oracle using `np.testing.assert_allclose`. Ensure that the names used to extract these activations match the naming convention used in the JAX code (e.g., checking for suffixes like `_act` or `_out` if used to avoid duplicate scope name errors). You MUST pass the `err_msg` parameter (e.g., `err_msg=f"Mismatch in layer: {{layer_name}}"`) to `assert_allclose` so the user can easily see which specific sublayer failed.
- Dynamic Parameter Inspection:
  - The generated test script MUST first initialize the JAX model and print its parameter structure using `jax.tree_util.tree_map(lambda x: x.shape, nnx.state(model))`.
  - Do NOT attempt to access `.value` inside `jax.tree_util.tree_map` on `nnx.State` (since the leaves are already unwrapped to raw Arrays). To assign values to parameters, do it by modifying the model properties directly (e.g., `model.layer.weight.value = jnp.array(pt_weight)`).
  - Use this structure to dynamically verify that the paths used in the weight mapping actually exist. For multi-layer models, check for both `model.layers[i]` patterns.
  - provide a helpful error message showing the expected vs. actual structure if they don't match.
- Do not generate function calls or tool calls. Your response should only contain the Python test script.

Here is the PyTorch code:
```python
{pytorch_code}
```

Here is the JAX code:
```python
{jax_code}
```

Please generate a Python test script that imports `torch_module` (which contains the PyTorch code) and `jax_module` (which contains the JAX code), and runs comparison tests.
Assume these modules are available in the Python path.
For your response, only return the Python test script that you generated.
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

SELF_DEBUGGING_PROMPT = """
You are an expert JAX programmer tasked with debugging JAX code based on PyTorch-to-JAX conversion errors.
You are continuing a debugging session. The previous JAX code you generated failed validation against the original PyTorch code.
Your job is to fix the JAX code based on the failing test traceback and bug analysis.

Your task is to:
- Identify the cause of the error from the provided bug analysis and stack trace.
- Modify only the necessary parts of the previous JAX code to fix the error shown in Execution Traceback, following the JAX/Flax best practices below.
- The fix must be targeted. Do not change the core logic or intended functionality of the original code.
- You must import and use the functions or classes from the provided library files. Do not copy or redefine them in your main script.
- If you see `AttributeError: module 'flax.linen' has no attribute 'GRU'`, replace usage of `nn.GRU` with `nn.RNN(nn.GRUCell(...))`.
- If test failures are due to small numerical discrepancies, check rules for **Numerical Parity** and **BatchNorm Momentum** below.
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
""" + JAX_BEST_PRACTICES

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

Guidelines:
- **Mandatory Activation Parity**: The JAX model must be structured to allow verification of intermediate results. Use Flax's `sow` mechanism to capture activations for every significant layer, using names that clearly correspond to the PyTorch module's attributes.

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

Guidelines:
- **Mandatory Activation Parity**: The JAX model must be structured to allow verification of intermediate results. Use Flax's `sow` mechanism to capture activations for every significant layer, using names that clearly correspond to the PyTorch module's attributes.

Please think step by step about the conversion process before generating the code.
Then, provide the JAX equivalent of the code above, using JAX libraries like
Flax if appropriate for transformer models. Ensure that the JAX code is
idiomatic and follows best practices, such as using pure functions and handling
random number generation correctly with JAX's PRNG keys.
Only return the Python code block for the JAX implementation.
""" + JAX_BEST_PRACTICES

MODEL_CONVERSION_PROMPT = (
    """You are the MaxText Core Architect Subagent, a compiler-focused engineer specialized in building ultra-scalable, sharding-aware models in JAX/Flax NNX. Your primary concern is Model FLOPs Utilization (MFU), SPMD sharding consistency, and compliance with the MaxText repository architecture. You treat the XLA compiler as the single source of truth for optimizations.
Your task is to convert the ENTIRE PyTorch file below to a single JAX/Flax NNX file. You MUST convert ALL classes, helper functions, constants, and configuration dataclasses -- not just one class.

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
   replace it with a flat split that produces wrong tensors when
   num_k_heads != num_v_heads.
3. If the source has a chunk-parallel delta rule with a for-loop computing a
   Neumann series (WY representation), convert it using
   `jax.scipy.linalg.solve_triangular(I - W, I, lower=True)` instead of
   jax.lax.scan with dynamic slicing. See the RAG context for the pattern.
4. If the source has both a chunk (prefill) and recurrent (decode) mode for
   linear attention, implement BOTH modes and dispatch based on sequence length.
5. Implement causal_conv1d as a standalone function with both prefill and
   single-step decode paths.
6. **Mandatory Activation Parity**: Store intermediates directly on the class
   instance under intermediate attributes (e.g. `self.intermediates[layer_name] = activation`)
   or declare them under an intermediate variable category.
7. For causal operations with decode-time state (causal conv1d, linear
   attention), implement SEPARATE prefill and decode functions. Do NOT use
   a single unified function with conditional branching.
8. ALWAYS include a `@dataclasses.dataclass` Config class at the top of the
   output file. Mirror ALL fields from the PyTorch configuration class with
   their types and default values. Use `dataclasses.field(default_factory=...)`
   for mutable defaults. Use the Config type (not `Any`) in module annotations.
9. The `load_balancing_loss` function MUST accept an optional `attention_mask`
   parameter. When the mask is provided, broadcast it to match the concatenated
   router logits shape and use it to exclude padding tokens from mean/sum
   statistics. See the RAG context for the full pattern.
10. **MoE Experts: Capacity-Based Dispatch (MANDATORY)**. The Experts class MUST
   use capacity-based dispatch with dispatch/combine tensors, NOT per-token
   gather of expert weights. The correct pattern is:
   a) Compute per-expert capacity: `capacity = ceil(T * K / E) * 1.5`
   b) Build dispatch tensor via `one_hot(selected_experts) -> cumsum -> positions
      -> one_hot(positions, capacity)` to get `dispatch: [T, E, C]`
   c) Build combine tensor: `combine = dispatch * routing_weights`
   d) Route tokens to expert buffers: `expert_in = einsum('tec,th->ech', dispatch, x)`
   e) Batched expert matmul: `expert_out = einsum('ech,ehi->eci', expert_in, W)`
   f) Scatter back: `output = einsum('tec,ech->th', combine, expert_out)`
   Do NOT use `weight[flat_indices]` gather or `jax.vmap` over individual experts.
   Do NOT use `jnp.einsum('td,edh->teh')` computing all experts for all tokens.
   The capacity-based approach is 10-50x more efficient for large E (e.g. E=64).
11. **KV Cache: Stateful Variables (MANDATORY)**. In Flax NNX, KV caches are
    stateful variables owned by their respective attention modules. Do NOT pass
    caches as global function arguments or return them in the output tuple. The
    `__call__` signature of attention and layer blocks must be extremely clean:
    `def __call__(self, hidden_states, q_residual, position_ids)`. Caches must
    be updated internally by writing directly to state attributes (e.g.
    `self.cache.value = new_value`).
    Provide an `init_kv_caches()` helper function if applicable.
12. **Tied Output Projection**: When the PyTorch source computes logits via
    `x @ self.token_embedding.weight.T`, convert it to
    `(x @ token_embedding.embedding.T).astype(jnp.float32)`.
    Do NOT use `token_embedding.attend(x)` -- that is for embedding lookup,
    not linear projection, and may produce different results.
13. **Fused QKV Projection**: Map to a single target NNX state matrix parameter.
    Slice it inside `__call__` for Q, K, and V. Provide query/key/value projection
    wrappers mapping the reference API.
14. **Float32 Softmax Upcast (MANDATORY)**: When the PyTorch source uses
    `.float()` or `dtype=torch.float32` before softmax, you MUST preserve this
    in JAX: `jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)` then
    cast back with `.astype(q.dtype)`. This is critical for numerical stability
    in bfloat16/float16. NEVER omit this upcast.
15. **Preserve ALL Source Components (MANDATORY)**: The output MUST contain a
    JAX NNX equivalent for EVERY class, function, method, and utility in the source.
    Do NOT merge base classes into subclasses. Do NOT drop get_config() or
    serialization methods. Do NOT omit utility classes (e.g., metrics classes)
    or standalone functions (e.g., metric computation functions). If the source
    has N classes and M functions, the output must have N classes and M functions.
16. **Preserve Default Values Exactly**: All constructor defaults, config
    defaults, and hyperparameter defaults MUST match the PyTorch source exactly.
    Do NOT change capacity_factor, dropout rates, noise epsilon, num_layers,
    or any other default value -- even if you think a different value is better.
17. **Train/Eval Mode in Flax NNX**: stochastic layers (such as `nnx.Dropout`)
    must accept the target `deterministic` flag to toggle evaluation mode. Use the
    stateful RNG keys container (`nnx.Rngs`) in model constructors for randomized
    parameters.
18. **MaxText Design Persona & Signature Conventions (MANDATORY)**:
    - Avoid literal, line-by-line translation of PyTorch helper functions or intermediate output patterns. Instead, design JAX classes to follow the MaxText design persona.
    - Specifically, for Rotary Embeddings (RoPE), the `__call__` method must follow the standard MaxText signature: it must take `(inputs, position_ids, ...)` where `inputs` is the rank-4 tensor of shape `[B, S, N, H]`, apply the rotation internally (by calling rotation logic/helpers), and return the rotated tensor of the same shape. Do NOT return raw `cos`/`sin` tuple unless the parent class contract strictly requires it.
    - If the PyTorch source has helper functions like `rotate_half` or `apply_rotary_pos_emb`, they can be redefined as private functions (prefixed with `_`) or replaced by native JAX operations.
    - Subclassing native MaxText layers remains mandatory (such as subclassing `RotaryEmbedding` or `nnx.Module` as appropriate for RoPE, `Attention` for attention layers, etc.).
    - Wrap the final block class using `nnx_wrappers.to_linen(...)` to ensure compatibility with Linen training loops.

Please think step by step about the conversion process before generating the JAX NNX code.
Then, provide the complete JAX NNX equivalent of the entire file above.
Ensure that the JAX NNX code is idiomatic and follows best practices.
Only return the Python code block for the JAX NNX implementation.
"""
    "" + JAX_BEST_PRACTICES
)
