PROMPT = """You are tasked with bootstrapping the reference kernel for the pipeline.

**Your Goal:**
1. Check the user's initial prompt and context.
2. Identify the Pallas or JAX kernel code they want to optimize, and whether it came from a reference source file path on disk.
3. Save the main kernel file to `{base_kernel_path?}` using the `restricted_write_file` tool.
4. If the kernel came from a reference source file path on disk, call `discover_kernel_dependencies(source_file_path='<path_to_source_file>')` to automatically discover and register any required local workspace dependency files into session state.
5. If the kernel relies on additional local dependency files or helper modules that were not automatically found, use the `write_dependency_file` tool to save them.

**Instructions:**
- If the source code is pasted in the user's message/context (with no reference source file path on disk), use `restricted_write_file` to save it to `{base_kernel_path?}` and do NOT call `discover_kernel_dependencies`.
- If a source file name or path is provided in the user's message, use the `read_file` tool to read its content, use `restricted_write_file` to save it to `{base_kernel_path?}`, and ALWAYS call `discover_kernel_dependencies(source_file_path='<path_to_source_file>')` with the original source file path.
"""
