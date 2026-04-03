"""Extended prompt for generating and writing conversion summary."""

PROMPT = """
After generating the summary, you MUST write it to a file using write_file_direct.

CRITICAL FILE PATH INSTRUCTIONS:
- Use EXACTLY the filename "CONVERSION_SUMMARY.md" - NO directory paths, NO prefixes
- The file will be written to the same directory as the user-provided GPU code file
- Example: If the user provided "/home/user/project/kernel.cu", all output files (including CONVERSION_SUMMARY.md) will be written to "/home/user/project/"
- ONLY use: name="CONVERSION_SUMMARY.md"

Call write_file_direct with:
- path="CONVERSION_SUMMARY.md"
- content=<the complete summary you just generated>

After successfully writing the file, note that the tool will return the full path where the file was written.

Respond with: "Conversion complete! All output files have been written to the same directory as your GPU code file.

Files created:
- SIMPLIFICATION_PLAN.md (approved plan)
- simplified_code.[cu|py|txt] (GPU code with hardware optimizations removed)
- SIMPLIFICATION_SUMMARY.md (simplification details)
- converted_jax.py (runnable JAX code)
- test_correctness.py (validation test script)
- CONVERSION_SUMMARY.md (full conversion summary)"

This is the final step of the GPU-to-JAX conversion workflow.

After completing this step, transfer control back to the parent orchestrator:
transfer_to_agent('GpuToJaxAgent')"""
