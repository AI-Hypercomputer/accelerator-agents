#!/bin/bash

# Script to start the kernel agent with UI or CLI options
# Usage: ./run_hitl_agent.sh [--ui] [--reset] [--session <session_id>]
#
# Note: CLI and UI modes use different session storage:
#   - CLI mode: JSON files (*.session.json) 
#   - UI mode: SQLite database (sessions.db)
# Sessions are NOT interchangeable between modes.

set -e

UI_MODE=false
RESET_MODE=false
OUTPUT_FILE="output.txt"
UI_PORT=1430
SESSION_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ui)
            UI_MODE=true
            shift
            ;;
        --reset)
            RESET_MODE=true
            shift
            ;;
        --session)
            SESSION_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--ui] [--reset] [--session <session_id>]"
            echo ""
            echo "Options:"
            echo "  --ui              Start with web UI on port $UI_PORT"
            echo "  --reset           Kill existing instances and restart"
            echo "  --session <id>    Specify session ID to resume (CLI mode only)"
            echo "  --help            Show this help message"
            echo ""
            echo "Note: CLI and UI modes use different session storage mechanisms:"
            echo "  - CLI mode: JSON files (*.session.json) in current directory"
            echo "  - UI mode: SQLite database (sessions.db) - manage via web interface"
            echo "  Sessions cannot be shared between CLI and UI modes."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to kill existing adk processes
kill_existing_processes() {
    echo "Checking for existing adk processes..."
    
    # Kill adk processes
    pkill -f "adk web" || true
    pkill -f "adk run" || true
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill if still running
    pkill -9 -f "adk web" || true
    pkill -9 -f "adk run" || true
    
    echo "Existing processes terminated."
}

# Function to select or manage session
select_session() {
    local mode=$1  # "ui" or "cli"
    
    # Find all session files in current directory
    local session_files=(*.session.json)
    
    # If no session files found
    if [ ! -e "${session_files[0]}" ]; then
        echo "No existing sessions found. Starting fresh..."
        SESSION_ID="hitl_session_$(date +%Y%m%d_%H%M%S)"
        return 0
    fi
    
    # If session ID was provided via command line, validate it
    if [ -n "$SESSION_ID" ]; then
        if [ -f "${SESSION_ID}.session.json" ]; then
            echo "Resuming session: $SESSION_ID"
            return 0
        else
            echo "Warning: Session file ${SESSION_ID}.session.json not found."
            echo ""
        fi
    fi
    
    # Display available sessions
    echo ""
    echo "Available sessions:"
    local i=1
    local session_ids=()
    for file in "${session_files[@]}"; do
        local sid="${file%.session.json}"
        session_ids+=("$sid")
        local mod_time=$(stat -c '%y' "$file" 2>/dev/null || stat -f '%Sm' "$file" 2>/dev/null || echo "Unknown")
        echo "  $i) $sid (Last modified: ${mod_time:0:19})"
        ((i++))
    done
    
    echo "  $i) Start new session"
    echo "  $((i+1))) Delete all sessions and start fresh"
    echo ""
    
    # Get user choice
    read -p "Enter your choice (1-$((i+1))): " choice
    
    # Validate input
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        echo "Invalid input. Starting new session..."
        SESSION_ID="hitl_session_$(date +%Y%m%d_%H%M%S)"
        return 0
    fi
    
    # Process choice
    if [ "$choice" -ge 1 ] && [ "$choice" -lt "$i" ]; then
        # Resume existing session
        local idx=$((choice - 1))
        SESSION_ID="${session_ids[$idx]}"
        echo "Resuming session: $SESSION_ID"
    elif [ "$choice" -eq "$i" ]; then
        # Start new session
        SESSION_ID="hitl_session_$(date +%Y%m%d_%H%M%S)"
        echo "Starting new session: $SESSION_ID"
    elif [ "$choice" -eq "$((i+1))" ]; then
        # Delete all and start fresh
        echo "Deleting all session files..."
        rm -f *.session.json
        SESSION_ID="hitl_session_$(date +%Y%m%d_%H%M%S)"
        echo "Starting fresh with session: $SESSION_ID"
    else
        echo "Invalid choice. Starting new session..."
        SESSION_ID="hitl_session_$(date +%Y%m%d_%H%M%S)"
    fi
}

# Function to start UI mode
start_ui_mode() {
    echo "Starting kernel agent in UI mode on port $UI_PORT..."
    echo "(Sessions stored in SQLite database: hitl_kernel_gen_agent/sessions.db)"
    echo ""
    
    # Use SQLite for session persistence in UI mode
    local session_db="hitl_kernel_gen_agent/sessions.db"
    
    echo "Output will be logged to hitl_kernel_gen_agent/$OUTPUT_FILE"
    echo "You can access the web interface at: http://localhost:$UI_PORT"
    echo ""
    echo "All sessions will be automatically saved to the database."
    echo "Resume any session from the UI by selecting it in the web interface."
    
    # Start in background and redirect output
    cd ..
    nohup adk web --port $UI_PORT --session_service_uri "sqlite:///$session_db" > hitl_kernel_gen_agent/$OUTPUT_FILE 2>&1 &
    ADK_PID=$!
    cd hitl_kernel_gen_agent

    # Get the PID
    echo "Started with PID: $ADK_PID"
    
    # Wait a moment and check if process is still running
    sleep 2
    if kill -0 $ADK_PID 2>/dev/null; then
        echo "Kernel agent UI started successfully!"
        echo "To stop the agent, run: kill $ADK_PID"
        echo "To view logs, run: tail -f $OUTPUT_FILE"
    else
        echo "Failed to start kernel agent UI. Check $OUTPUT_FILE for errors."
        exit 1
    fi
}

# Function to start CLI mode
start_cli_mode() {
    echo "Starting kernel agent in CLI mode..."
    echo "(Sessions stored as JSON files in hitl_kernel_gen_agent/ directory)"
    echo ""
    
    # Handle session selection
    select_session "cli"
    
    echo "Starting adk run with session saving..."
    echo "Session ID: $SESSION_ID"
    
    # Check if resuming an existing session
    if [ -f "hitl_kernel_gen_agent/${SESSION_ID}.session.json" ]; then
        echo "Resuming from saved session file..."
        cd ..
        PYTHONWARNINGS="ignore" adk run hitl_kernel_gen_agent --resume "hitl_kernel_gen_agent/${SESSION_ID}.session.json" --save_session --session_id "$SESSION_ID"
    else
        echo "Starting new session..."
        cd ..
        PYTHONWARNINGS="ignore" adk run hitl_kernel_gen_agent --save_session --session_id "$SESSION_ID"
    fi
}

# Main execution
echo "=== Kernel Agent Startup Script ==="
echo "Mode: $([ "$UI_MODE" = true ] && echo "UI" || echo "CLI")"
echo "Reset: $([ "$RESET_MODE" = true ] && echo "Yes" || echo "No")"
echo ""

# Kill existing processes if reset is requested
if [ "$RESET_MODE" = true ]; then
    kill_existing_processes
    kill_existing_processes
    echo ""
fi

# Copy env file to parent directory for adk commands
cp .env ../.env

# Start the appropriate mode
if [ "$UI_MODE" = true ]; then
    start_ui_mode
else
    start_cli_mode
fi