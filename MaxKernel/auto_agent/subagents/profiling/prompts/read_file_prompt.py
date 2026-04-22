"""Kernel file reading prompt for profiling workflows."""

PROMPT = """Your goal is to read the kernel file that needs to be profiled.

**Check Session State First:**
- Check if `optimized_kernel_path` is set in state: `{optimized_kernel_path?}`
- If it is set, use `read_file` to read that exact file.

**If not in state, parse the user's message to identify the file:**
- Look for explicit file names: "profile kernel.py", "analyze my_kernel.py", "check performance of test_kernel.py"
- Look in the conversation history for recently mentioned files
- All files are located in the working directory: {workdir}

**If unclear which file to read:**
- Ask: "Which kernel file would you like me to profile? Please provide the filename."

After reading, the file content will be available for profiling analysis.
"""
