"""Validation file reading prompt for compilation validation workflows."""

PROMPT = """Your goal is to identify and read the kernel file that needs to be validated for compilation.

**Parse the user's message to identify the kernel file:**
- Look for explicit file paths: "auto-iterate on path/to/kernel.py", "validate kernel.py", "check compilation of my_kernel.py"
- Look for files mentioned in the conversation history
- Check if a kernel was recently implemented (look for `optimized_kernel_path` in state)
- All files are located in the working directory: {workdir}

**If you find the kernel file path:**
- Use `read_file` to read the specified file (this will automatically save the path to state)

**If the kernel path is unclear:**
- Ask the user: "Which kernel file would you like to validate? Please provide the file path."
"""
