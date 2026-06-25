#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Change to the script directory so Python files are always found
cd "$SCRIPT_DIR" || exit 1

# Export PYTHONPATH pointing to MaxKernel root
export PYTHONPATH="$(cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"

# Detect Python executable
if [ -f "$SCRIPT_DIR/../../.venv/bin/python3" ]; then
    PYTHON_EXE="$(cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)/.venv/bin/python3"
    echo "Using virtualenv Python: $PYTHON_EXE"
else
    PYTHON_EXE="python3"
    echo "Using system Python"
fi

# Check if mock mode is requested
if [ "${MOCK_COMPILER,,}" = "true" ]; then
    TPU_SERVER_SCRIPT="mock_tpu_server.py"
    echo "Running in Mock Compiler Mode"
else
    TPU_SERVER_SCRIPT="tpu_server.py"
    echo "Running in Real TPU Compiler Mode"
fi

if [ "$1" = "--start-tpu" ]; then
    # Start TPU server on port 5463
    nohup "$PYTHON_EXE" "$TPU_SERVER_SCRIPT" > output_tpu_server.txt 2>&1 &

    echo "TPU server started successfully on port 5463 using $TPU_SERVER_SCRIPT"
elif [ "$1" = "--start-cpu" ]; then
    # Start CPU server on port 5464
    nohup "$PYTHON_EXE" cpu_server.py > output_cpu_server.txt 2>&1 &

    echo "CPU server started successfully on port 5464"
elif [ "$1" = "--start-eval" ]; then
    # Start eval server on port 1245
    nohup "$PYTHON_EXE" eval_server.py > output_eval_server.txt 2>&1 &

    echo "Eval server started successfully on port 1245"
elif [ "$1" = "--start-all" ]; then
    # Start all servers
    nohup "$PYTHON_EXE" "$TPU_SERVER_SCRIPT" > output_tpu_server.txt 2>&1 &
    nohup "$PYTHON_EXE" cpu_server.py > output_cpu_server.txt 2>&1 &
    nohup "$PYTHON_EXE" eval_server.py > output_eval_server.txt 2>&1 &

    echo "All servers started successfully (TPU script: $TPU_SERVER_SCRIPT)"
elif [ "$1" = "--end" ]; then

    # Kill Python processes for all servers
    pkill -f "mock_tpu_server.py"
    pkill -f "tpu_server.py"
    pkill -f "cpu_server.py"
    pkill -f "eval_server.py"

    echo "Server(s) stopped successfully"
else
    echo "Usage: $0 --start-tpu|--start-cpu|--start-eval|--start-all|--end"
fi