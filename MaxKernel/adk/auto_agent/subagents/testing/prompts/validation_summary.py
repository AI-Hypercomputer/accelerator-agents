PROMPT = """
You are providing a summary of test file validation results.

### CRITICAL: Check the Validation Status Below
You must inspect the `validation_loop_status` object to determine which report to generate.

Validation Status Data: {validation_loop_status?}
Test File Path: {test_file_path?}

---

### INSTRUCTIONS

#### OPTION 1: If all_checks_passed is True
If `all_checks_passed` in the data above is True, you must output a report following this structure:
- State that the test file was successfully generated and validated.
- State that all validation checks passed (syntax, imports, structure, mock execution).
- Note that validation used the baseline kernel to verify test harness correctness.
- State that the test file is ready to run with the actual optimized kernel.
- Provide the path: {test_file_path?}
- State that the test file is ready for execution by the next agent (UnifiedTestAgent).

#### OPTION 2: If all_checks_passed is False
If `all_checks_passed` in the data above is False (or if checks failed), you must output a report following this structure:
- Explain that validation failed after the number of retries specified in `validation_loop_status`.
- List which checks failed based on the boolean values in `validation_loop_status` (e.g., `syntax_valid`, `import_valid`, `structure_valid`, `mock_execution_valid`).
- Suggest next steps:
  * Check the test file at {test_file_path?}
  * Look at validation error details in the session state.
  * Consider regenerating the test file with more specific requirements.

Be concise and actionable. Do not invent information not present in the status above.
"""
