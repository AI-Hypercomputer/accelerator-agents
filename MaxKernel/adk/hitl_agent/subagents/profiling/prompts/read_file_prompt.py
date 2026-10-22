"""Kernel file reading prompt for profiling workflows."""

PROMPT = """Your goal is to read the kernel file that needs to be profiled.

**Parse the user's message to identify the file:**
- Look for explicit file names: "profile kernel.py", "analyze my_kernel.py", "check performance of test_kernel.py"
- Look in the conversation history for recently mentioned files
- Check the working directory: {workdir}

**If the user's request is clear:**
- Use `read_file` to read the specified file

**If unclear which file to read:**
- Ask: "Which kernel file would you like me to profile? Please provide the filename."

After reading, the file content will be available for profiling analysis.
"""
