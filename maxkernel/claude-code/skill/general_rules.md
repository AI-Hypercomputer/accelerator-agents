# MaxKernel General Rules

1. **Python Environment**: NEVER use system `python3`. Always use the prepared virtual environment: `{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/<tool_name>.py`.
2. **Strict Boundaries**: All generated artifacts belong in `<run_dir>`. Do NOT run `find`, `Glob`, or `Grep` across the repository or the home directory. Reading the MaxKernel TPU knowledge base under `{{MAXKERNEL_ROOT}}/*.md` by explicit path is allowed and expected.
3. **Fail Fast**: If a required file (e.g. `<run_dir>/state.json`) is missing, terminate immediately with an explicit error. Do NOT hallucinate parameters or fake test results.
4. **TPU Execution**: Submitting to TPU must ONLY be done via `tpu_client.py`. NEVER use `gcloud compute tpus tpu-vm ssh` directly.
5. **Debug Protocol**: If you encounter an infrastructure failure, gracefully terminate and report the explicit error back to the caller so it can be logged in `<run_dir>/maxkernel_debug_history.md`.
6. **Preserve User Input Parameters**: NEVER modify user-specified input parameters or tensor shapes (e.g. in `get_inputs()`, `max_seq`, batch size, problem dimensions). All kernel optimizations must strictly preserve the user's input parameter values.
7. **Report, don't decide**: You run one phase of a larger loop. Finish your phase, write your artifact, and return a short report. Do not dispatch other agents and do not try to advance the loop yourself.
