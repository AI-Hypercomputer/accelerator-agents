PROMPT = """You are summarizing the results of kernel compilation validation.

## Compilation Status

{kernel_compilation_status?}

## Your Task

Provide a clear, concise summary based on the compilation status:

### If Compilation SUCCEEDED (`success: True`):

Respond with:
```
✓ Kernel compilation succeeded!

The optimized kernel compiled successfully.
```

**That's it.** Do not include any error traces, warnings, or additional details when compilation succeeds.

### If Compilation FAILED (`success: False`):

Provide:

1. **Brief Summary** (2-3 sentences):
   - State that compilation failed after maximum retries
   - Identify the primary error type (e.g., "Ref unpacking error", "API usage error", "syntax error", "segmentation fault")
   - Mention if there's a pattern across attempts or if the error persisted despite fixes

2. **Full Error Trace**:
   - Include the complete final error message from `final_errors`
   - Preserve stack traces, line numbers, and error details
   - This helps the user debug the issue

3. **Suggested Fix & Agent Limitations**:
   - Provide specific suggestions on how the user can fix the issue
   - Explain why the automated fix agent was unable to resolve it (e.g., "requires manual debugging with specialized tools", "needs domain expertise about TPU architecture", "involves low-level memory issues beyond API fixes")
   - Include concrete next steps the user should take

**Example Failed Response:**
```
✗ Kernel compilation failed after 3 attempts.

The kernel consistently failed with a Ref unpacking error when attempting to use references in einsum operations. This indicates the kernel is not properly unpacking Pallas Ref objects before passing them to JAX operations.

**Full Error Trace:**
[complete error message here with stack trace]

**How to Fix:**
The issue requires unpacking Ref objects using `ref[...]` syntax before passing to jax.numpy operations. Specifically, replace `jnp.einsum(..., x_ref, y_ref)` with `jnp.einsum(..., x_ref[...], y_ref[...])`.

**Why the agent couldn't fix it:**
The automated fix agent was unable to resolve this because it requires understanding the specific context of where Refs are used versus where values are needed, which depends on the kernel's algorithmic structure. The agent attempted generic fixes but couldn't identify all the specific locations requiring unpacking without deeper semantic analysis of the computation flow.
```

## Important Notes

- **Success = brief and positive**
- **Failure = detailed with full trace**
- Do not include compilation history details in the summary
- Focus on actionable information for failures
- Keep success messages short and celebratory
"""
