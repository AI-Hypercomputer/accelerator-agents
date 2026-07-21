PROPOSE_PROMPT = """You are tasked with bootstrapping the reference kernel for the pipeline.

**Your Goal:**
1. Check the user's initial prompt and context. 
2. Identify the Pallas or JAX kernel code they want to optimize.
3. Save it to `{base_kernel_path?}` using the `restricted_write_file` tool.

**Instructions:**
- If the source code is pasted in the user's message/context, use the `restricted_write_file` tool to save it.
- If a source file name or path is provided in the user's message, use the `read_file` tool to read its content, and then use `restricted_write_file` to save it.
"""
