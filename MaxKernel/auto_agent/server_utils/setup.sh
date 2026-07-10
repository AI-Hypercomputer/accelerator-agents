#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Change to the script directory so Python files are always found
cd "$SCRIPT_DIR" || exit 1

# Helper to check if a bastion_vm is configured in gke_config.yaml
get_bastion_config() {
    if [ -f "gke_config.yaml" ]; then
        python3 -c '
import yaml
try:
    with open("gke_config.yaml") as f:
        cfg = yaml.safe_load(f)
    b = cfg.get("bastion_vm", {})
    if b:
        name = b.get("name", "")
        zone = b.get("zone", "")
        project = b.get("project", "")
        local_port = b.get("local_port", b.get("port", 1245))
        remote_port = b.get("remote_port", b.get("port", 1245))
        print(f"{name}|{zone}|{project}|{local_port}|{remote_port}")
except:
    pass
'
    fi
}

# Lazy loader for bastion configuration to avoid running python on every script call
load_bastion_config() {
    if [ -n "$BASTION_NAME" ]; then
        return # Already loaded
    fi
    
    local config_info=$(get_bastion_config)
    if [ -n "$config_info" ]; then
        IFS='|' read -r BASTION_NAME BASTION_ZONE BASTION_PROJECT BASTION_LOCAL_PORT BASTION_REMOTE_PORT <<< "$config_info"
    fi
    BASTION_LOCAL_PORT=${BASTION_LOCAL_PORT:-1245}
    BASTION_REMOTE_PORT=${BASTION_REMOTE_PORT:-1245}
}

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
elif [ "$1" = "--start-gke" ]; then
    load_bastion_config
    
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
        
        # Test the tunnel connection
        echo "Waiting for SSH tunnel to establish..."
        TUNNEL_SUCCESS=false
        for i in {1..15}; do
            if curl -s --max-time 2 http://localhost:${BASTION_LOCAL_PORT}/health >/dev/null 2>&1; then
                TUNNEL_SUCCESS=true
                break
            fi
            sleep 1
        done
        
        if [ "$TUNNEL_SUCCESS" = true ]; then
            echo "SSH tunnel established and remote Evaluation server is reachable!"
        else
            echo "========================================================="
            echo " WARNING: SSH tunnel established but Evaluation server is not reachable on localhost:${BASTION_LOCAL_PORT}."
            echo " Please verify that the eval_server.py is running on the Bastion VM."
            echo "========================================================="
            exit 1
        fi
    else
        echo "Error: No bastion_vm configuration found in gke_config.yaml."
        echo "Please add a 'bastion_vm' section with 'name' to gke_config.yaml."
        exit 1
    fi
elif [ "$1" = "--start-local" ] || [ "$1" = "--start-gce" ]; then
    # Start all local execution/evaluation servers (needed for local or GCE cases)
    echo "Starting local background servers (CPU, TPU, Eval)..."
    nohup python3 tpu_server.py > output_tpu_server.txt 2>&1 &
    nohup python3 cpu_server.py > output_cpu_server.txt 2>&1 &
    nohup python3 eval_server.py > output_eval_server.txt 2>&1 &

    echo "Local background servers started successfully."
elif [ "$1" = "--end-gke" ]; then
    load_bastion_config
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
    pkill -f "tpu_server.py" && servers_stopped=true
    pkill -f "cpu_server.py" && servers_stopped=true
    pkill -f "eval_server.py" && servers_stopped=true
    
    if [ "$servers_stopped" = true ]; then
        echo "Local background servers stopped successfully."
    else
        echo "No active local background servers were running."
    fi
elif [ "$1" = "--end" ]; then
    load_bastion_config
    echo "Stopping all background servers and tunnels..."
    
    tunnel_stopped=false
    if [ -n "$BASTION_LOCAL_PORT" ] && [ -n "$BASTION_REMOTE_PORT" ]; then
        pkill -f "${BASTION_LOCAL_PORT}:localhost:${BASTION_REMOTE_PORT}" && tunnel_stopped=true
    fi
    
    servers_stopped=false
    pkill -f "tpu_server.py" && servers_stopped=true
    pkill -f "cpu_server.py" && servers_stopped=true
    pkill -f "eval_server.py" && servers_stopped=true

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