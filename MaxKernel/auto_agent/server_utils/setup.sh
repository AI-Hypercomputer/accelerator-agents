#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

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
    # Kill existing eval_server and restart (picks up any eval_config.yaml changes)
    pkill -f "eval_server.py" 2>/dev/null || true
    sleep 1
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "Eval server started successfully on port 1245"
elif [ "$1" = "--start-all" ]; then
    # Start all local servers
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "All local servers started successfully"
elif [ "$1" = "--start-remote-tpus" ]; then
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
    fi
    REMOTE_PATH="${EXTRA_TPU_REMOTE_PATH:-~/accelerator-agents/MaxKernel}"
    if [ -z "$EXTRA_TPU_IPS" ]; then
        echo "No EXTRA_TPU_IPS configured in ${ENV_FILE}. Nothing to start."
        exit 0
    fi
    IFS=',' read -ra IPS <<< "$EXTRA_TPU_IPS"
    for ip in "${IPS[@]}"; do
        ip=$(echo "$ip" | tr -d '[:space:]')
        echo "Starting TPU server on $ip (${REMOTE_PATH})..."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
            "cd ${REMOTE_PATH} && bash auto_agent/server_utils/setup.sh --start-tpu" &
    done
    wait
    echo "Remote TPU servers start commands sent to: ${IPS[*]}"
elif [ "$1" = "--stop-remote-tpus" ]; then
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
    fi
    if [ -z "$EXTRA_TPU_IPS" ]; then
        echo "No EXTRA_TPU_IPS configured. Nothing to stop."
        exit 0
    fi
    IFS=',' read -ra IPS <<< "$EXTRA_TPU_IPS"
    for ip in "${IPS[@]}"; do
        ip=$(echo "$ip" | tr -d '[:space:]')
        echo "Stopping TPU server on $ip..."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
            "pkill -f tpu_server.py 2>/dev/null || true; echo 'TPU server stopped on $ip'" &
    done
    wait
    echo "Remote TPU server stop commands sent."
elif [ "$1" = "--restart-all" ]; then
    echo "Stopping local servers..."
    pkill -f "tpu_server.py" 2>/dev/null || true
    pkill -f "cpu_server.py" 2>/dev/null || true
    pkill -f "eval_server.py" 2>/dev/null || true
    sleep 2

    echo "Starting local servers..."
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "Starting remote TPU servers..."
    bash "$0" --start-remote-tpus

    echo "All servers restarted."
elif [ "$1" = "--end" ]; then
    # Kill all local server processes
    pkill -f "tpu_server.py" 2>/dev/null || true
    pkill -f "cpu_server.py" 2>/dev/null || true
    pkill -f "eval_server.py" 2>/dev/null || true

    echo "Local server(s) stopped successfully"
else
    echo "Usage: $0 --start-tpu|--start-cpu|--start-eval|--start-all|--start-remote-tpus|--stop-remote-tpus|--restart-all|--end"
fi
