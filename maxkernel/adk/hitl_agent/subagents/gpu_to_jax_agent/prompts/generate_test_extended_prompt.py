"""Extended prompt for generating correctness test with error handling instructions."""

PROMPT = """**IMPORTANT ERROR HANDLING:**
If write_file_direct fails or returns an error:
1. Report the error clearly to the user
2. Display the generated test code in a code block
3. Ask the user to manually save the code to test_correctness.py if the tool fails
4. Do NOT proceed to transfer_to_agent if the file write failed

Only proceed with transfer_to_agent('RunCorrectnessTestAgent') if the file was successfully written."""
