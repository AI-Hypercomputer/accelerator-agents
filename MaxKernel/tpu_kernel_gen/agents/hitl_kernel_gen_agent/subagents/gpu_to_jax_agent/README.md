# GPU to JAX Conversion Agent

An intelligent agent that automatically converts GPU-optimized code (CUDA, Triton, PyTorch CUDA, CuDNN/cuBLAS) to clean, algorithmic JAX code. This agent strips hardware-specific optimizations and focuses on producing readable, correct JAX implementations.

## Features

- **Multi-Framework Support**: Converts from CUDA, Triton, PyTorch CUDA, and CuDNN/cuBLAS
- **Automatic Framework Detection**: Identifies the source framework automatically
- **Hardware Optimization Removal**: Strips GPU-specific optimizations (shared memory, thread management, etc.)
- **Comprehensive Testing**: Includes syntax validation, compilation checks, shape validation, and numerical correctness tests
- **Iterative Error Fixing**: Automatically fixes common conversion errors
- **Integrated with HITL Agent**: Available as a subagent in the human-in-the-loop kernel generation workflow

## Architecture

### Pipeline Overview

The conversion follows a 10-step sequential pipeline:

1. **Read GPU File** - Reads the source GPU code file
2. **Identify Framework** - Detects CUDA, Triton, PyTorch CUDA, etc.
3. **Organize Code** - Simplifies and linearizes the code structure
4. **Convert to JAX** - Translates to JAX with proper idioms
5. **Fix Errors (Loop)** - Iteratively fixes syntax and conversion errors
6. **Validate Compilation** - Checks that JAX code compiles
7. **Validate Shapes** - Verifies input/output shape consistency
8. **Generate Tests** - Creates numerical correctness tests
9. **Run Tests** - Executes correctness validation
10. **Generate Summary** - Produces comprehensive conversion report

### Directory Structure

```
gpu_to_jax_agent/
├── agent.py                    # Main agent orchestration
├── constants.py                # Configuration constants
├── prompts/                    # LLM prompts for each step
│   ├── identify_framework_prompt.py
│   ├── organize_gpu_code_prompt.py
│   ├── convert_to_jax_prompt.py
│   ├── fix_conversion_prompt.py
│   └── summary_prompt.py
├── evaluators/                 # Validation agents
│   ├── jax_syntax_checker.py
│   ├── shape_validator.py
│   ├── compilation_checker.py
│   └── correctness_checker.py
├── examples/                   # Example conversions
│   ├── example_cuda_vector_add.cu
│   ├── expected_cuda_vector_add_jax.py
│   ├── example_triton_matmul.py
│   ├── expected_triton_matmul_jax.py
│   ├── example_pytorch_mlp.py
│   └── expected_pytorch_mlp_jax.py
├── test_agent.py              # Unit tests
└── README.md                  # This file
```

## How to Run

### Prerequisites

1. **Start the Evaluation Server** (required for compilation and correctness checks):
   ```bash
   cd tpu_kernel_gen/agents/kernel_gen_agent/kernel_eval
   python eval_server.py
   ```
   The server should be running on `localhost:1245` (configurable in `constants.py`)

2. **Set Working Directory** (optional):
   ```bash
   export WORKDIR=/path/to/your/gpu/code
   ```
   If not set, defaults to the agent's directory.

### Running the Agent

#### Option 1: Through HITL Agent (Recommended)

The GPU-to-JAX conversion agent is **integrated as a subagent** in the HITL kernel generation orchestrator. It's automatically available when you use the HITL agent:

```python
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.agent import root_agent

# The root_agent now includes gpu_to_jax_conversion_agent as one of its subagents
# When you interact with root_agent, you can request GPU-to-JAX conversion naturally
```

Example interaction:
```
User: "Convert my CUDA kernel to JAX"
User: "I have a Triton kernel I want to translate to JAX"
User: "Please convert this PyTorch CUDA code to JAX: /path/to/my_kernel.py"
```

The orchestrator will automatically invoke the `GpuToJaxConversionAgent` when it detects conversion requests. The agent is now part of the sub_agents list in `hitl_kernel_gen_agent/agent.py:393`.

#### Option 2: Direct Python Import (Advanced)

For programmatic usage or custom workflows:

```python
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.agent import root_agent
from google.adk.sessions import Session

# Create a session
session = Session()

# Add your request to the session
session.add_user_message("Convert my CUDA kernel at /path/to/kernel.cu to JAX")

# Run the root agent (which includes gpu_to_jax_conversion_agent)
async for event in root_agent.run_async(session):
    print(event)
```

Or access the subagent directly:

```python
from tpu_kernel_gen.agents.hitl_kernel_gen_agent.gpu_to_jax_agent import (
    gpu_to_jax_conversion_agent,
)

# Run standalone (requires session setup with file path in state)
await gpu_to_jax_conversion_agent.run_async(session)
```

### Quick Test with Examples

Try one of the provided examples:

```bash
# From the HITL agent interface
User: "Convert the example CUDA kernel at tpu_kernel_gen/agents/hitl_kernel_gen_agent/gpu_to_jax_agent/examples/example_cuda_vector_add.cu to JAX"
```

The agent will:
1. Read the CUDA file
2. Identify it as CUDA
3. Organize and simplify the code
4. Convert to JAX
5. Validate syntax, compilation, shapes
6. Generate and run correctness tests
7. Provide a comprehensive summary

### Expected Output

The agent will write the converted JAX code to a new file and provide a summary like:

```
CONVERSION SUMMARY
==================

Successfully converted CUDA kernel to JAX.

Source Framework: CUDA
Target Framework: JAX

Conversion Process:
- Detected CUDA kernel with thread-level parallelism
- Removed hardware-specific optimizations (shared memory, thread indexing)
- Organized code into linear structure
- Translated CUDA operations to equivalent JAX operations

Test Results:
✓ Compilation: PASSED
✓ Syntax Validation: PASSED
✓ Shape Validation: PASSED
✓ Numerical Correctness: PASSED

Overall Assessment: The conversion is complete and ready to use.
```

## Conversion Philosophy

### What Gets Removed

The agent **strips all hardware-specific optimizations**:

- ✗ CUDA kernel launch configurations (`<<<blocks, threads>>>`)
- ✗ Shared memory declarations (`__shared__`)
- ✗ Thread block management (`threadIdx`, `blockIdx`)
- ✗ Memory coalescing patterns
- ✗ Warp-level primitives
- ✗ Triton tiling and blocking
- ✗ Device management (`.cuda()`, `.to(device)`)

### What Gets Preserved

The agent **focuses on algorithmic correctness**:

- ✓ Core mathematical operations
- ✓ Algorithm structure and logic
- ✓ Input/output shapes
- ✓ Numerical accuracy
- ✓ Function semantics

### Output Structure

All converted JAX code follows a consistent three-section structure:

```python
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
# ... variable initialization, data setup ...

# Computation
def computation(inputs...) -> output:
    # ... core algorithm ...
    return result

output = jax.block_until_ready(computation(inputs...))
```

## Examples

### Example 1: CUDA Vector Addition

**Input** (`example_cuda_vector_add.cu`):
```cuda
__global__ void vector_add_kernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

**Output** (`expected_cuda_vector_add_jax.py`):
```python
# Imports
import jax.numpy as jnp

# Initialization
N = 1024
A = jnp.arange(N, dtype=jnp.float32)
B = jnp.arange(N, dtype=jnp.float32) * 2

# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return A + B

C = computation(A, B)
```

### Example 2: Triton Matrix Multiplication

**Input**: Tiled Triton matmul kernel with blocking (see `example_triton_matmul.py`)

**Output**: Simple JAX matmul (see `expected_triton_matmul_jax.py`)
```python
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)
```

### Example 3: PyTorch MLP

**Input**: PyTorch `nn.Module` with `.cuda()` calls (see `example_pytorch_mlp.py`)

**Output**: Functional JAX implementation (see `expected_pytorch_mlp_jax.py`)

## Testing

Run the test suite:

```bash
pytest tpu_kernel_gen/agents/hitl_kernel_gen_agent/gpu_to_jax_agent/test_agent.py -v
```

## Configuration

Edit `constants.py` to customize:

```python
MODEL_NAME = "gemini-2.5-pro"           # LLM model
MAX_ITERATIONS = 5                       # Max error-fixing iterations
CONVERSION_TIMEOUT = 180                 # Timeout for conversions
NUMERICAL_TOLERANCE = 1e-5              # Tolerance for correctness checks
```

## Evaluators

### JaxSyntaxChecker

Validates JAX code syntax and checks for:
- Valid Python syntax
- Required JAX imports
- GPU-specific remnants (`.cuda()`, `<<<>>>`, etc.)
- Common API mismatches (`dim=` vs `axis=`)
- PRNG key usage

### ShapeValidator

Validates tensor shapes:
- Extracts shape information from code
- Checks dimension compatibility for operations
- Validates computation function signatures
- Detects reshape and reduction operations

### JaxCompilationChecker

Tests code compilation:
- Runs code on eval server
- Captures compilation errors
- Returns detailed error messages

### JaxCorrectnessChecker

Validates numerical accuracy:
- Compares outputs between GPU and JAX implementations
- Uses configurable tolerance (default: 1e-5)
- Reports maximum differences

## Common Conversion Patterns

### Random Number Generation

**Before** (PyTorch):
```python
x = torch.randn(N, M)
```

**After** (JAX):
```python
key = random.PRNGKey(0)
x = random.normal(key, (N, M))
```

### Tensor Operations

**Before** (PyTorch):
```python
y = x.view(N, M)
z = torch.sum(x, dim=1)
```

**After** (JAX):
```python
y = jnp.reshape(x, (N, M))
z = jnp.sum(x, axis=1)
```

### Activations

**Before** (PyTorch):
```python
y = F.relu(x)
z = F.softmax(x, dim=-1)
```

**After** (JAX):
```python
y = jax.nn.relu(x)
z = jax.nn.softmax(x, axis=-1)
```

## Troubleshooting

### Common Errors

1. **Missing PRNG Key**
   - Error: `module 'jax.numpy' has no attribute 'random'`
   - Fix: Add `import jax.random as random` and initialize PRNG key

2. **Wrong Axis Parameter**
   - Error: `got unexpected keyword argument 'dim'`
   - Fix: Change `dim=` to `axis=` in JAX functions

3. **Eval Server Not Running**
   - Error: `Cannot connect to evaluation server`
   - Fix: Start the eval server before running conversions

### Limitations

- Custom CUDA operations without JAX equivalents require manual intervention
- Complex memory access patterns may need additional review
- Performance characteristics will differ from optimized GPU code

## Future Enhancements

- [ ] Support for custom CUDA extensions
- [ ] Pallas kernel generation option (for performance)
- [ ] Automatic performance optimization suggestions
- [ ] Support for more GPU libraries (cuDNN, cuBLAS specifics)
- [ ] Interactive correction mode for failed conversions

## Contributing

Contributions are welcome! Areas for improvement:
- Additional framework support
- More comprehensive test cases
- Better error messages
- Performance optimization hints

## License

Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0.
