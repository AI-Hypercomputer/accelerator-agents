"""Extended prompt for fixing conversion errors with routing logic."""

PROMPT = """
After fixing the code and writing it to converted_jax.py, transfer back to ValidateSyntaxAgent to re-validate:
transfer_to_agent('ValidateSyntaxAgent')

Note: If you've attempted fixes multiple times and errors persist, you may choose to proceed anyway:
transfer_to_agent('ValidateCompilationAgent')"""
