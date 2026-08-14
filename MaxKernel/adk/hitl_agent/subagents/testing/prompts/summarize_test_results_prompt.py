PROMPT = """Analyze the pytest test results and provide a comprehensive summary with actionable recommendations.

## Test Results

{test_results?}

## Your Task

Analyze these test results and provide a comprehensive report with the following sections:

### 1. Overall Status
- Clear statement: Did all tests pass, or were there failures?
- Quick overview: compilation status, correctness status, performance status

### 2. Test Breakdown
Provide detailed analysis for each test category:

**Compilation Tests:**
- Did the kernels compile successfully?
- Were there any compilation errors or warnings?

**Correctness Tests:**
- Did the optimized kernel produce correct results?
- Were there numerical accuracy issues (tolerance problems)?
- Did outputs match the baseline across different input sizes?

**Performance Tests:**
- What was the performance comparison between base and optimized kernels?
- Was there a speedup? How much?
- Did performance meet expectations?

### 3. Detailed Error Analysis
If any test failed:
- Include the **FULL traceback** and error message
- Identify the root cause of the failure
- Explain what the error means in plain language

### 4. Recommendations

Based on the test results, provide **specific, actionable recommendations** for next steps.

**Use your tools to research recommendations:**

1. **`retrieval_tool`**: Use this EXTENSIVELY to look up:
   - Common error patterns and their solutions (e.g., "BlockSpec errors", "JAX shape mismatch")
   - Best practices for fixing compilation issues
   - Numerical accuracy guidelines for Pallas kernels
   - Performance optimization patterns for TPU
   - TPU-specific error debugging (memory constraints, MXU usage, etc.)
   - Debugging strategies for specific error types

2. **`search_api_tool`**: Use this to find:
   - Specific API documentation for error-prone functions
   - Known issues and workarounds

**Recommendation Guidelines:**

- If tests **passed**: Suggest next steps (auto-tuning parameters like block sizes using `AutotuneAgent`, profiling for bottlenecks, testing with more input sizes, production deployment considerations)
- If **compilation failed**: Provide specific fixes based on the error (API signature issues, import problems, syntax errors)
- If **correctness failed**: Suggest debugging approaches (check block boundaries, verify reduction operations, inspect memory access patterns, adjust tolerances)
- If **performance is poor**: Suggest optimization opportunities (block size tuning using `AutotuneAgent`, memory layout optimization, pipelining, prefetching)

**Important**:
- Research the specific error types using your tools before making recommendations
- Provide code examples or specific changes when possible
- Reference documentation or examples you find via retrieval_tool
- Prioritize recommendations by impact and ease of implementation

### Output Format

Structure your response as:

```
## Test Summary

[Overall status and quick overview]

## Detailed Results

### Compilation
[Compilation test results]

### Correctness
[Correctness test results]

### Performance
[Performance test results]

## Error Analysis

[If failures occurred, full tracebacks and explanations]

## Recommendations

[Numbered list of specific, actionable recommendations with code examples where applicable]
```

Provide a clear, actionable summary that helps the user understand what happened and what to do next.
"""
