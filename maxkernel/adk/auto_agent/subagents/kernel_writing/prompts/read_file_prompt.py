"""Validation file reading prompt for compilation validation workflows."""

PROMPT = """Your goal is to identify and read the kernel file that needs to be validated for compilation.

**Check Session State First:**
- Check if `optimized_kernel_path` is set in state: `{optimized_kernel_path?}`
- If it is set, use `read_file` to read that exact file.

**If not in state, parse the user's message to identify the kernel file:**
- Look for explicit file paths: "auto-iterate on path/to/kernel.py", "validate kernel.py", "check compilation of my_kernel.py"
- Look for files mentioned in the conversation history
- All files are located in the working directory: {workdir}

**If the kernel path is unclear:**
- Ask the user: "Which kernel file would you like to validate? Please provide the file path."
"""
