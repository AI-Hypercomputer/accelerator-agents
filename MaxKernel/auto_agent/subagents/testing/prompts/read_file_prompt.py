"""Test file reading prompt for test execution workflows."""

PROMPT = """Your goal is to identify and read the pytest test file that needs to be executed.

**Parse the user's message to identify the test file. Look for patterns like:**
- "test /path/to/test_file.py"
- "run tests in test_kernel.py"
- "execute my_test.py"
- "test this file: test_example.py"

**Check these sources in order:**
1. **Explicit path in user's message** - If user provides a file path, use it
2. **Recently generated test** - Check if a test file was generated: `{test_file_path?}`
3. **Look for test files** - Search for files starting with "test_" in {workdir}

**If the user specifies a test file clearly:**
- Use `read_file` to verify the file exists and is a valid test file

**If no test file is specified or found:**
- Ask: "Which test file would you like to run? Please either:
  - Provide the test file path (e.g., 'test /path/to/test_file.py')
  - Generate a new test file first using the test generation feature"

All files are located in: {workdir}
"""
