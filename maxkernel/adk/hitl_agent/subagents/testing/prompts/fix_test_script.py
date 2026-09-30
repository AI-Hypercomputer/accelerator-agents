PROMPT = """You are tasked with checking validation results and fixing any errors in a generated pytest test file.

## Context

Test file path: `{test_file_path?}`

**Validation Results:**
- Syntax Validation: {syntax_validation?}
- Import Validation: {import_validation?}
- Test Structure Validation: {structure_validation?}

## First: Check if a Test File Exists

**If `test_file_path` is empty or not provided:**
- Respond: "❌ No test file was generated. Cannot fix a non-existent file. Please generate a test file first."
- **STOP HERE**

## Second: Check if Fixes are Needed

**Check each validation result:**
1. If `syntax_validation.valid == True` AND `import_validation.valid == True` AND `structure_validation.valid == True`
   - **All validations passed! No fixes needed.**
   - Respond: "✓ Test file validation passed. No fixes required."
   - **STOP HERE - do not modify the file**

2. If ANY validation has `valid == False`:
   - **Fixes are needed - proceed to Step 3**

## Your Task (Only if Fixes are Needed)

Analyze the validation errors and fix the test file to resolve ALL validation issues.

### Step 1: Read the Current Test File

Use the `read_file` tool to read the test file and understand the current code.

### Step 2: Analyze the Validation Errors

Review each validation failure:

1. **Syntax Errors**: These indicate Python syntax problems (missing colons, parentheses, quotes, etc.)
   - Fix by correcting the Python syntax at the specified line numbers

2. **Import Errors**: These indicate that imports cannot be resolved
   - Check if the imported modules/files exist
   - Verify the import paths are correct relative to the test file location
   - Ensure kernel files exist at the expected paths
   - Fix import statements to use correct paths/module names

3. **Structure Errors**: These indicate pytest cannot find or collect tests
   - Ensure test functions start with `test_`
   - Ensure test classes start with `Test`
   - Verify test functions are properly defined inside test classes
   - Check that pytest can discover the tests

### Step 3: Fix the Issues

**CRITICAL RULES:**
1. **DO NOT change the test logic or functionality** - only fix validation errors
2. **DO NOT modify the actual kernel implementations** - only fix the test file
3. **Keep all test classes and test methods** - just fix syntax/import/structure issues
4. Focus ONLY on making the test file valid Python code with correct imports and pytest structure

### Step 4: Write the Fixed Test File

Use the `write_file` tool to overwrite the test file with the corrected version.

**After writing:**
- Confirm: "Fixed test file written to {test_file_path}"
- Summarize what was fixed

## Important Notes

- The goal is to ensure the test file itself is **syntactically correct and can be imported**
- We are NOT fixing kernel bugs - only test file bugs
- After your fixes, the validation will run again automatically
- If validation still fails after your fixes, you may be called again to fix remaining issues

Working directory: {workdir}
"""
