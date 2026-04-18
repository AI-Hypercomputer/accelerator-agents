#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Change to the script directory so Python files are always found
cd "$SCRIPT_DIR" || exit 1

if [ "$1" = "--start-tpu" ]; then
    # Start TPU server on port 5463
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &

    echo "TPU server started successfully on port 5463"
elif [ "$1" = "--start-cpu" ]; then
    # Start CPU server on port 5464
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &

    echo "CPU server started successfully on port 5464"
elif [ "$1" = "--start-eval" ]; then
    # Start eval server on port 1245
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "Eval server started successfully on port 1245"
elif [ "$1" = "--start-all" ]; then
    # Start all servers
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "All servers started successfully"
elif [ "$1" = "--end" ]; then
    # Kill Python processes for all servers
    pkill -f "tpu_server.py"
    pkill -f "cpu_server.py"
    pkill -f "eval_server.py"

    echo "Server(s) stopped successfully"
else
    echo "Usage: $0 --start-tpu|--start-cpu|--start-eval|--start-all|--end"
fi