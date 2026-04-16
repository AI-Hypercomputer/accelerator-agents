PROMPT = """
You are providing a summary of test file validation results.

### CRITICAL: Check the Validation Status Below
You must inspect the value of `ALL_CHECKS_PASSED` to determine which report to generate.

- ALL_CHECKS_PASSED: {validation_loop_status.all_checks_passed?}
- SUCCESS: {validation_loop_status.success?}
- RETRIES: {validation_loop_status.retries?}
- TEST_FILE_PATH: {test_file_path?}

---

### INSTRUCTIONS

#### OPTION 1: If ALL_CHECKS_PASSED is True
If the value above is True, you must output a report following this structure:
- State that the test file was successfully generated and validated.
- State that all validation checks passed (syntax, imports, structure, mock execution).
- Note that validation used the baseline kernel to verify test harness correctness.
- State that the test file is ready to run with the actual optimized kernel.
- Provide the path: {test_file_path?}
- State that the test file is ready for execution by the next agent (UnifiedTestAgent).

#### OPTION 2: If ALL_CHECKS_PASSED is False
If the value above is False (or if checks failed), you must output a report following this structure:
- Explain that validation failed after {validation_loop_status.retries?} retry attempts.
- List which checks failed based on these values:
  * Syntax: {validation_loop_status.syntax_valid?}
  * Imports: {validation_loop_status.import_valid?}
  * Structure: {validation_loop_status.structure_valid?}
  * Mock Execution: {validation_loop_status.mock_execution_valid?}
- Suggest next steps:
  * Check the test file at {test_file_path?}
  * Look at validation error details in the session state.
  * Consider regenerating the test file with more specific requirements.

Be concise and actionable. Do not invent information not present in the status above.
"""
