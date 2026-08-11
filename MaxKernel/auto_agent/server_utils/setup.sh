#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Change to the script directory so Python files are always found
cd "$SCRIPT_DIR" || exit 1

# Load all configurations and ports dynamically from server_config.py
load_config() {
    eval "$(python3 "$SCRIPT_DIR/server_config.py" 2>/dev/null)"
}

# Health check helper function
wait_for_server_health() {
    local name="$1"
    local port="$2"
    local log_file="$3"
    local max_retries="${4:-15}"
    
    echo "Waiting for $name on port $port to become healthy..."
    for ((i=1; i<=max_retries; i++)); do
        if curl -s --max-time 2 "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "$name started successfully and is healthy on port $port!"
            return 0
        fi
        sleep 1
    done
    
    echo "========================================================="
    echo " ERROR: $name failed to become healthy on port $port."
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo " Last 10 lines of $log_file:"
        tail -n 10 "$log_file"
    fi
    echo "========================================================="
    return 1
}

if [ "$1" = "--start-tpu" ]; then
    load_config
    if [ -z "$LOCAL_TPU_PORT" ]; then
        echo "Note: No local TPU backend configured in eval_config.yaml (remote TPU VM or CPU only)."
        exit 0
    fi
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    wait_for_server_health "TPU server" "$LOCAL_TPU_PORT" "output_tpu_server.txt" || exit 1

elif [ "$1" = "--start-cpu" ]; then
    load_config
    if [ -z "$LOCAL_CPU_PORT" ]; then
        echo "Note: No local CPU backend configured in eval_config.yaml."
        exit 0
    fi
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    wait_for_server_health "CPU server" "$LOCAL_CPU_PORT" "output_cpu_server.txt" || exit 1

elif [ "$1" = "--start-eval" ]; then
    load_config
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &
    wait_for_server_health "Eval server" "$EVAL_PORT" "output_eval_server.txt" || exit 1

elif [ "$1" = "--start-gke" ]; then
    load_config
    
    if [ -n "$BASTION_NAME" ]; then
        # Check if tunnel/eval server is already running on the local port
        if curl -s --max-time 2 http://localhost:${BASTION_LOCAL_PORT}/health >/dev/null 2>&1; then
            echo "An active Evaluation server is already reachable on local port ${BASTION_LOCAL_PORT}."
            echo "No need to restart the SSH tunnel."
            exit 0
        fi

        echo "GKE/Bastion configuration detected in gke_config.yaml."
        echo "Starting SSH tunnel to Bastion VM '$BASTION_NAME' mapping local port $BASTION_LOCAL_PORT to remote port $BASTION_REMOTE_PORT..."
        
        # Build gcloud command
        CMD=("gcloud" "compute" "ssh" "$BASTION_NAME")
        if [ -n "$BASTION_ZONE" ]; then
            CMD+=("--zone=$BASTION_ZONE")
        fi
        if [ -n "$BASTION_PROJECT" ]; then
            CMD+=("--project=$BASTION_PROJECT")
        fi
        CMD+=("--" "-N" "-L" "${BASTION_LOCAL_PORT}:localhost:${BASTION_REMOTE_PORT}")
        
        # Start the tunnel in the background
        nohup "${CMD[@]}" > output_bastion_tunnel.txt 2>&1 &
        wait_for_server_health "GKE Evaluation tunnel" "$BASTION_LOCAL_PORT" "output_bastion_tunnel.txt" || exit 1
    else
        echo "Error: No bastion_vm configuration found in gke_config.yaml."
        echo "Please add a 'bastion_vm' section with 'name' to gke_config.yaml."
        exit 1
    fi
elif [ "$1" = "--start-local" ] || [ "$1" = "--start-gce" ]; then
    load_config
    # Start all local execution/evaluation servers (needed for local or GCE cases)
    echo "Starting local background servers (CPU, TPU, Eval)..."
    if [ -n "$LOCAL_TPU_PORT" ]; then
        nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    fi
    if [ -n "$LOCAL_CPU_PORT" ]; then
        nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    fi
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    if [ -n "$LOCAL_TPU_PORT" ]; then
        wait_for_server_health "TPU server" "$LOCAL_TPU_PORT" "output_tpu_server.txt" || exit 1
    fi
    if [ -n "$LOCAL_CPU_PORT" ]; then
        wait_for_server_health "CPU server" "$LOCAL_CPU_PORT" "output_cpu_server.txt" || exit 1
    fi
    wait_for_server_health "Eval server" "$EVAL_PORT" "output_eval_server.txt" || exit 1
elif [ "$1" = "--end-gke" ]; then
    load_config
    echo "Stopping GKE SSH tunnel..."
    if [ -n "$BASTION_LOCAL_PORT" ] && [ -n "$BASTION_REMOTE_PORT" ]; then
        if pkill -f "${BASTION_LOCAL_PORT}:localhost:${BASTION_REMOTE_PORT}"; then
            echo "GKE tunnel stopped successfully."
        else
            echo "No active GKE tunnel was running on port ${BASTION_LOCAL_PORT}."
        fi
    fi
elif [ "$1" = "--end-local" ] || [ "$1" = "--end-gce" ]; then
    echo "Stopping local background servers..."
    servers_stopped=false
    pkill -f "tpu_server.py" && servers_stopped=true || true
    pkill -f "cpu_server.py" && servers_stopped=true || true
    pkill -f "eval_server.py" && servers_stopped=true || true
    
    if [ "$servers_stopped" = true ]; then
        echo "Local background servers stopped successfully."
    else
        echo "No active local background servers were running."
    fi
elif [ "$1" = "--end" ]; then
    load_config
    echo "Stopping all background servers and tunnels..."
    
    tunnel_stopped=false
    if [ -n "$BASTION_LOCAL_PORT" ] && [ -n "$BASTION_REMOTE_PORT" ]; then
        pkill -f "${BASTION_LOCAL_PORT}:localhost:${BASTION_REMOTE_PORT}" && tunnel_stopped=true || true
    fi
    
    servers_stopped=false
    pkill -f "tpu_server.py" && servers_stopped=true || true
    pkill -f "cpu_server.py" && servers_stopped=true || true
    pkill -f "eval_server.py" && servers_stopped=true || true

    if [ "$tunnel_stopped" = true ] || [ "$servers_stopped" = true ]; then
        echo "Successfully stopped:"
        [ "$tunnel_stopped" = true ] && echo "  - GKE SSH tunnel"
        [ "$servers_stopped" = true ] && echo "  - Local background servers"
    else
        echo "No active servers or tunnels were running."
    fi
else
    echo "Usage: $0 --start-local|--start-gce|--start-gke|--start-tpu|--start-cpu|--start-eval|--end|--end-local|--end-gce|--end-gke"
fi