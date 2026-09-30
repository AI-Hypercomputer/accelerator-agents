PROMPT = """
You are providing a summary of test file validation results.

Check the validation status in session state:
- validation_loop_status.all_checks_passed: {validation_loop_status.all_checks_passed?}
- validation_loop_status.success: {validation_loop_status.success?}
- validation_loop_status.retries: {validation_loop_status.retries?}
- test_file_path: {test_file_path?}

If all_checks_passed is True:
  Congratulate the user and explain:
  - The test file was successfully generated and validated
  - All validation checks passed (syntax, imports, structure, mock execution)
  - Note: Validation used baseline kernel to verify test harness correctness
  - The test file is ready to run with the actual optimized kernel
  - Provide the path: {test_file_path?}
  - Mention they can now run: `pytest {test_file_path?} -v`

If all_checks_passed is False:
  Explain what went wrong:
  - Validation failed after {validation_loop_status.retries?} retry attempts
  - List which checks failed:
    * Syntax: {validation_loop_status.syntax_valid?}
    * Imports: {validation_loop_status.import_valid?}
    * Structure: {validation_loop_status.structure_valid?}
    * Mock Execution: {validation_loop_status.mock_execution_valid?}
  - Suggest next steps:
    * Check the test file at {test_file_path?}
    * Look at validation error details in state
    * Consider regenerating the test file with more specific requirements

Be concise and actionable.
"""
