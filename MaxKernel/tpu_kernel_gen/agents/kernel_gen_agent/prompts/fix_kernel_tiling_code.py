PROMPT = """
You are an expert in Jax and writing Pallas kernels for TPUs. You are tasked with fixing a Pallas kernel script that either has compilation errors or does not produce the expected results.

Below are your previous attempts to generate a correct Pallas kernel script with tiling, along with the errors encountered in each attempt. Your job is to analyze these attempts and the errors, and then provide a corrected version of the Pallas kernel code with minimal changes.

### Starting code
{tiled_kernel_code}

### Previous Attempts
{tiled_kernel_prev_attempts}

### Tool Usage
You have access to tool `search_api_tool` that can search for and retrieve information about JAX APIs. You should use this tool to ensure you are using JAX and Pallas APIs correctly. For example, you can use it to verify function signatures, understand the purpose of different arguments, and check for official usage notes or constraints mentioned in the documentation.

To use the tool effectively, make sure to use tool name `search_api_tool` and provide the fully qualified name of the API as the tool's input (e.g., "jax.experimental.pallas.pallas_call" or "jax.experimental.pallas.BlockSpec").

### Full Pallas Documentation:
To provide you with the most comprehensive understanding of Pallas, here is the full documentation:
{pallas_docs}

### TPU Specs:
TPU generation to optimize for: tpu-v4
TPU high bandwidth memory (HBM) size: 32 GiB
TPU vector memory (VMEM) size: 32 MiB

### Some commonly observed errors:
- `ValueError: The Pallas TPU lowering currently supports only blocks of rank >= 1.`: This error indicates that either inputs or outputs of the kernel have tensor shape with rank < 1. Ensure that all inputs and outputs have a rank of at least 1.
- `NotImplementedError: Unimplemented primitive in Pallas TPU lowering for <primitive_name>`: This error indicates that a specific primitive operation is not supported in the Pallas TPU lowering and therefore cannot be used in the kernel. You should replace such operations with a computationally equivalent collection of simpler operations that are supported in Pallas.
- `jaxlib._jax.XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space vmem.`: This error indicates that the kernel is trying to use more memory than is available in the TPU's vector memory (VMEM). Since your focus is soley on implementing the simplest possible correct kernel, you should modify the tensor sizes in the initialization section to be smaller so that they fit in the VMEM requirements (specified in the TPU Specs section), ensuring that the logic remains unchanged.
- `ValueError: The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and 128 respectively, or be equal to the respective dimensions of the overall array.`: 

### Your task:
1. Analyze the previous attempts and the errors encountered.
2. Identify the specific issues in the kernel code that need to be fixed.
3. Provide a corrected version of the Pallas kernel code with tiling. Only change the code in the `computation` function (unless you need to change input size due to VMEM issues). The previously generated kernel code is designed to be as simple as possible, so you should not add any unnecessary complexity or optimizations. **DO NOT** add any additional input arguments to `pallas_call`. The kernel should only use the inputs provided in the initialization section and should not introduce any new variables or parameters.

Feel free to think out loud about your reasoning process, but ensure that the final output is a clean and corrected version of the Pallas kernel code without any comments addressing the fixes you made.
"""
