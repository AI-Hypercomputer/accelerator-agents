#!/bin/bash

if [ "$1" = "--start-tpu" ]; then
    # Start TPU server on port 5463
    cd kernel_gen_agent/kernel_eval
    python tpu_server.py > output_tpu_server.txt 2>&1 &

    echo "TPU server started successfully on port 5463"
elif [ "$1" = "--start-cpu" ]; then
    # Start CPU server on port 5464
    cd kernel_gen_agent/kernel_eval
    python cpu_server.py > output_cpu_server.txt 2>&1 &

    echo "CPU server started successfully on port 5464"
elif [ "$1" = "--start-eval" ]; then
    # Start eval server on port 1245
    cd kernel_gen_agent/kernel_eval
    python eval_server.py > output_eval_server.txt 2>&1 &

    echo "Eval server started successfully on port 1245"
elif [ "$1" = "--end" ]; then
    # Kill Python processes for all servers
    pkill -f "tpu_server.py"
    pkill -f "cpu_server.py"
    pkill -f "eval_server.py"

    echo "Server(s) stopped successfully"
else
    echo "Usage: $0 --start-tpu|--start-cpu|--start-eval|--end"
fi