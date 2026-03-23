PROMPT = """
Your goal is to provide the results. Your response should have three parts:

1) Compilation Results: Describes the results of the compilation of the Pallas kernel. First provide a summary, then the full results. If compilation failed, provide the error message. If compilation was successful, feel free to keep the section short.
2) Correctness Results: Describes the results of the correctness evaluation of the Pallas kernel. First provide a summary, then the full results. If correctness evaluation failed, provide the error message. If correctness evaluation was successful, feel free to keep the section short.
3) Performance Results: Describes the results of the performance evaluation of the Pallas kernel. First provide a summary, then the full results. If performance evaluation failed, provide the error message. If performance evaluation was successful share the performance metrics.

## General Format:

1. **Compilation Results**:
    - Summary: A brief summary of the compilation results.
    - Full Results: The complete compilation results, including any error messages if applicable.

2. **Correctness Results**:
    - Summary: A brief summary of the correctness evaluation results.
    - Full Results: The complete correctness evaluation results, including any error messages if applicable.

3. **Performance Results**:
    - Summary: A brief summary of the performance evaluation results.
    - Full Results: The complete performance evaluation results, including any error messages if applicable.

## Relevant context:

To help you generate the results, here are the results of each step:
{compilation_results?}
{correctness_test_results?}
{performance_test_results?}
"""
