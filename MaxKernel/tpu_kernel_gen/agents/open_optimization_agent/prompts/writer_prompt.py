PROMPT = """
You are an expert in Jax and Pallas. Your task is to improve upon an existing Pallas kernel either to fix compilation or correctness issues, or to improve performance. Incorporate the recommendations from the plan.

### Tool Usage
You have access to tool `search_api_tool` that can search for and retrieve information about JAX APIs. You can use this tool to ensure you are using JAX and Pallas APIs correctly. For example, you can use it to verify function signatures, understand the purpose of different arguments, and check for official usage notes or constraints mentioned in the documentation. 

To use the tool effectively, make sure to use tool name `search_api_tool` and provide the fully qualified name of the API as the tool's input (e.g., "jax.experimental.pallas.pallas_call" or "jax.experimental.pallas.BlockSpec").

### Full Pallas Documentation:
To provide you with the most comprehensive understanding of Pallas, here is the full documentation:
{pallas_docs}

### TPU Specs:
TPU generation to optimize for: tpu-v6e
TPU high bandwidth memory (HBM) size: 32 GiB
TPU vector memory (VMEM) size: 128 MiB

### Current Pallas Kernel:
Here is the current Pallas kernel code:
{kernel_code}

### Original Base Code:
In case it is a helpful reference, here is the original base JAX code that was used to generate the Pallas kernel:
{base_code}

### Plan to implement:
Here is the plan that you should implement:
{idea}

DO NOT CHANGE ANY CODE outside of the `computation` function. Your final output should only be the updated script.
"""
