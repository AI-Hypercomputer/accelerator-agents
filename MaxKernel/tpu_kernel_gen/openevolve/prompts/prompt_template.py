PROMPT = """
You are an expert at optimizing code for TPUs using Jax and Pallas.

# TASK
Your task is to analyze the provided function and rewrite it into a highly optimized Pallas kernel for the Google TPU v4. Your goal should be to improve performance as much as possible.

# CONTEXT
The following sections aim to provide you with as much relevant information as possible to help you optimize the code. Please read them carefully.

## PALLAS DOCUMENTATION
{pallas_docs}

## KEY PALLAS APIS
{pallas_apis}

# IMPORTANT NOTES
When writing Pallas kernels, success is measured as a combination of correctness, followed by performance.
Ensure first that the kernel computes the same result as the original function, and then focus on optimizing performance.

# ** DO NOT CHANGE THE FOLLOWING **
# 1) Kernel function signature or input/output specifications
# 2) Overall algorithm correctness (must compute same result)
# 3) Output tensor shapes or semantics
# """
