#!/bin/bash

set -e  # Exit on any error

# Handle Ctrl+C gracefully
trap 'echo -e "\n\033[0;31m[ERROR]\033[0m Setup interrupted by user. Exiting..."; exit 130' INT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Repository root path (assuming this script is in repository root)
REPO_ROOT=$(pwd)
# Configuration env file path (save env config in the repository root directory)
CONFIG_FILE="$REPO_ROOT/.env"

# Function to get primary IP address (cross-platform Linux and macOS)
get_hostname_ip() {
    local ip=""
    
    # 1. Try Python socket connection (most reliable cross-platform method)
    if command -v python3 &>/dev/null; then
        ip=$(python3 -c '
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
    s.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
' 2>/dev/null || true)
    fi

    # 2. Linux fallback: hostname -I / hostname -i
    if [ -z "$ip" ] || [ "$ip" = "127.0.0.1" ]; then
        if command -v hostname &>/dev/null; then
            local host_ips
            host_ips=$(hostname -I 2>/dev/null || hostname -i 2>/dev/null || true)
            for candidate in $host_ips; do
                if [[ "$candidate" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [ "$candidate" != "127.0.0.1" ]; then
                    ip="$candidate"
                    break
                fi
            done
        fi
    fi

    # 3. macOS fallback: ipconfig getifaddr
    if [ -z "$ip" ] || [ "$ip" = "127.0.0.1" ]; then
        if command -v ipconfig &>/dev/null; then
            for iface in en0 en1 eth0 wlan0; do
                local candidate
                candidate=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
                if [ -n "$candidate" ]; then
                    ip="$candidate"
                    break
                fi
            done
        fi
    fi

    # 4. Fallback default
    if [ -z "$ip" ]; then
        ip="127.0.0.1"
    fi

    echo "$ip"
}

# Function to prompt user about environment setup
prompt_env_setup() {
    print_info "Python environment setup"
    echo ""
    echo "Choose how you want to set up your Python environment:"
    echo "1) Create a .venv virtual environment (Recommended)"
    echo "2) Use your current Python environment"
    echo ""
    echo -n "Enter choice (1-2): "
    read -r choice
    
    case $choice in
        1)
            return 0  # Create venv
            ;;
        2)
            print_info "Using your own environment. Make sure you have Python 3.11+ and pip installed."
            return 1  # Skip environment setup
            ;;
        *)
            print_error "Invalid choice. Please select 1-2"
            exit 1
            ;;
    esac
}

# Function to create and activate venv
create_venv() {
    print_info "Creating .venv virtual environment..."
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not available. Please install Python 3.11+ first."
        exit 1
    fi
    
    # Create venv
    python3 -m venv .venv
    
    # Activate venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success ".venv created and activated successfully"
}

# Function to check if a package is installed
is_package_installed() {
    pip show "$1" >/dev/null 2>&1
}

# Function to check if packages from requirements file are installed
check_requirements_installed() {
    local req_file="$1"
    if [ ! -f "$req_file" ]; then
        return 1
    fi
    
    # Parse requirements file and check each package
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Extract package name (before ==, >=, <=, >, <, [, or whitespace)
        package_name=$(echo "$line" | sed -E 's/^([^=<>\[[:space:]]+).*/\1/' | tr -d '[:space:]')
        
        if [ -n "$package_name" ] && ! is_package_installed "$package_name"; then
            return 1  # At least one package is not installed
        fi
    done < "$req_file"
    
    return 0  # All packages are installed
}

# Function to install repository dependencies
install_dependencies() {
    print_info "Checking repository dependencies..."
    DEPENDENCY_ROOT="$REPO_ROOT/dependency"

    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed or not in PATH. Please install pip first."
        exit 1
    fi
    
    # Check and install main requirements
    MAIN_REQUIREMENTS="$DEPENDENCY_ROOT/main_requirements.txt"
    if [ -f "$MAIN_REQUIREMENTS" ]; then
        if check_requirements_installed "$MAIN_REQUIREMENTS"; then
            print_info "Main requirements already installed, skipping..."
        else
            print_info "Installing main requirements.txt... (this may take several minutes)"
            pip install -r "$MAIN_REQUIREMENTS"
        fi
    else
        print_warning "Main requirements.txt not found"
    fi
    
    # Check and install agent requirements
    AGENT_REQUIREMENTS="$DEPENDENCY_ROOT/agent_requirements.txt"
    if [ -f "$AGENT_REQUIREMENTS" ]; then
        if check_requirements_installed "$AGENT_REQUIREMENTS"; then
            print_info "Agent requirements already installed, skipping..."
        else
            print_info "Installing agent requirements.txt..."
            pip install -r "$AGENT_REQUIREMENTS"
        fi
    else
        print_warning "Agent requirements.txt not found"
    fi

    # Check if the package is already installed in editable mode
    if pip show hitl-agent >/dev/null 2>&1; then
        print_info "Repo package already installed, skipping..."
    else
        print_info "Installing package in editable mode..."
        pip install -e "$REPO_ROOT"
    fi

    # Check if npx is installed
    if ! command -v npx &> /dev/null; then
        print_info "npx not found. Installing nodejs and npm via nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
        nvm install 20
        
        # Verify npx is now available
        if command -v npx &> /dev/null; then
            print_success "nodejs, npm, and npx installed successfully"
            return 1  # Signal that nvm was installed and restart is needed
        else
            print_warning "Node.js installed but npx not found in PATH."
            return 1  # Signal that nvm was installed and restart is needed
        fi
    else
        print_info "npx is already installed"
    fi

    print_success "Dependencies check completed successfully"
    return 0  # Dependencies installed successfully
}

# Function to load existing config
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_info "Loading existing configuration from $CONFIG_FILE"
        source "$CONFIG_FILE"
    else
        print_info "No existing configuration found, will create new config file"
    fi
}

# Function to prompt for environment variables
prompt_env_vars() {
    print_info "Setting up environment variables..."
    
    # GEMINI_API_KEY
    if [ -z "$GEMINI_API_KEY" ]; then
        echo -n "Enter your GEMINI_API_KEY: "
        read -r GEMINI_API_KEY
        # Strip quotes and whitespace
        GEMINI_API_KEY=$(echo "$GEMINI_API_KEY" | sed -e "s/^['\"]*//" -e "s/['\"]*$//" -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')
        if [ -z "$GEMINI_API_KEY" ]; then
            print_error "GEMINI_API_KEY is required"
            exit 1
        fi
    else
        print_info "Using existing GEMINI_API_KEY"
    fi
    
    # WORKDIR
    if [ -z "$WORKDIR" ]; then
        echo -n "Enter your WORKDIR (default: $(pwd)): "
        read -r WORKDIR_INPUT
        if [ -z "$WORKDIR_INPUT" ]; then
            WORKDIR="$(pwd)"
        else
            # Strip quotes and whitespace
            WORKDIR=$(echo "$WORKDIR_INPUT" | sed -e "s/^['\"]*//" -e "s/['\"]*$//" -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')
        fi
    else
        print_info "Using existing WORKDIR: $WORKDIR"
    fi
    
    # Create WORKDIR if it doesn't exist
    if [ ! -d "$WORKDIR" ]; then
        mkdir -p "$WORKDIR"
        print_info "Created WORKDIR: $WORKDIR"
    fi
    
    # TPU_VERSION
    if [ -z "$TPU_VERSION" ]; then
        echo "Select TPU version:"
        echo "1) TPU v4"
        echo "2) TPU v5e"
        echo "3) TPU v5p"
        echo "4) TPU v6e"
        echo "5) TPU v7x"
        echo -n "Enter choice (1-5): "
        read -r choice

        case $choice in
            1) TPU_VERSION="TPU v4" ;;
            2) TPU_VERSION="TPU v5e" ;;
            3) TPU_VERSION="TPU v5p" ;;
            4) TPU_VERSION="TPU v6e" ;;
            5) TPU_VERSION="TPU 7x" ;;
            *)
                print_error "Invalid choice. Please select 1-5"
                exit 1
                ;;
        esac
    else
        print_info "Using existing TPU_VERSION: $TPU_VERSION"
    fi

    # GOOGLE_CLOUD_PROJECT (mandatory - needed for Vertex AI)
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        echo -n "Enter your GOOGLE_CLOUD_PROJECT (default: tpu-kernel-assist-sandbox): "
        read -r GOOGLE_CLOUD_PROJECT_INPUT
        # Strip quotes and whitespace
        GOOGLE_CLOUD_PROJECT_INPUT=$(echo "$GOOGLE_CLOUD_PROJECT_INPUT" | sed -e "s/^['\"]*//" -e "s/['\"]*$//" -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')
        if [ -z "$GOOGLE_CLOUD_PROJECT_INPUT" ]; then
            GOOGLE_CLOUD_PROJECT="tpu-kernel-assist-sandbox"
            print_info "Using default GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
        else
            # Validate project ID format (lowercase letters, digits, hyphens)
            if [[ "$GOOGLE_CLOUD_PROJECT_INPUT" =~ ^[a-z0-9-]+$ ]]; then
                GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT_INPUT"
            else
                print_error "Invalid GOOGLE_CLOUD_PROJECT format. Project IDs must contain only lowercase letters, digits, and hyphens."
                exit 1
            fi
        fi
    else
        print_info "Using existing GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
    fi

    # GOOGLE_CLOUD_LOCATION (mandatory - needed for Vertex AI)
    if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
        echo "Select Google Cloud location:"
        echo "1) global (default)"
        echo "2) us-central1"
        echo "3) us-east1"
        echo "4) europe-west1"
        echo "5) asia-northeast1"
        echo "6) other (enter custom location)"
        echo -n "Enter choice (1-6, or press Enter for default): "
        read -r location_choice

        case $location_choice in
            1|"")
                GOOGLE_CLOUD_LOCATION="global"
                print_info "Using GOOGLE_CLOUD_LOCATION: $GOOGLE_CLOUD_LOCATION"
                ;;
            2) GOOGLE_CLOUD_LOCATION="us-central1" ;;
            3) GOOGLE_CLOUD_LOCATION="us-east1" ;;
            4) GOOGLE_CLOUD_LOCATION="europe-west1" ;;
            5) GOOGLE_CLOUD_LOCATION="asia-northeast1" ;;
            6)
                echo -n "Enter custom Google Cloud location: "
                read -r CUSTOM_LOCATION
                # Strip quotes and whitespace
                CUSTOM_LOCATION=$(echo "$CUSTOM_LOCATION" | sed -e "s/^['\"]*//" -e "s/['\"]*$//" -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')
                if [ -z "$CUSTOM_LOCATION" ]; then
                    print_error "Location cannot be empty"
                    exit 1
                fi
                GOOGLE_CLOUD_LOCATION="$CUSTOM_LOCATION"
                ;;
            *)
                print_error "Invalid choice. Please select 1-6"
                exit 1
                ;;
        esac
    else
        print_info "Using existing GOOGLE_CLOUD_LOCATION: $GOOGLE_CLOUD_LOCATION"
    fi

    # INCLUDE_THOUGHTS (optional - enable reasoning traces)
    if [ -z "$INCLUDE_THOUGHTS" ]; then
        echo "Show reasoning traces in responses?"
        echo "1) Yes (show thinking process)"
        echo "2) No (hide thinking process)"
        echo -n "Enter choice (1-2, or press Enter for default): "
        read -r thinking_choice

        case $thinking_choice in
            1|"")
                INCLUDE_THOUGHTS="true"
                print_info "Reasoning traces enabled"
                ;;
            2)
                INCLUDE_THOUGHTS="false"
                ;;
            *)
                print_error "Invalid choice. Please select 1-2"
                exit 1
                ;;
        esac
    else
        print_info "Using existing INCLUDE_THOUGHTS: $INCLUDE_THOUGHTS"
    fi

    # RAG_CORPUS (optional - for Vertex AI RAG Engine)
    if [ -z "$RAG_CORPUS" ]; then
        echo -n "Enter your Vertex AI RAG corpus path (optional, press Enter to use default): "
        read -r RAG_CORPUS_INPUT
        # Strip quotes and whitespace
        RAG_CORPUS_INPUT=$(echo "$RAG_CORPUS_INPUT" | sed -e "s/^['\"]*//" -e "s/['\"]*$//" -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')
        if [ -n "$RAG_CORPUS_INPUT" ]; then
            # Validate RAG corpus path format
            if [[ "$RAG_CORPUS_INPUT" =~ ^projects/.+/locations/.+/ragCorpora/.+$ ]]; then
                RAG_CORPUS="$RAG_CORPUS_INPUT"
                print_info "Using RAG_CORPUS: $RAG_CORPUS"
            else
                print_error "Invalid RAG_CORPUS format. Expected format: projects/{project}/locations/{location}/ragCorpora/{corpus_id}"
                exit 1
            fi
        else
            # Use default corpus
            RAG_CORPUS="projects/tpu-kernel-assist-sandbox/locations/us-west1/ragCorpora/7991637538768945152"
            print_info "Using default RAG_CORPUS: $RAG_CORPUS"
        fi
    else
        print_info "Using existing RAG_CORPUS: $RAG_CORPUS"
    fi
}

# Function to save configuration
save_config() {
    print_info "Saving configuration to $CONFIG_FILE"
    
    # Always use Vertex AI since GOOGLE_CLOUD_PROJECT is now mandatory
    USE_VERTEXAI="TRUE"

    cat > "$CONFIG_FILE" << EOF
# Agent configuration file
# Generated on $(date)

GEMINI_API_KEY="$GEMINI_API_KEY"
WORKDIR="$WORKDIR"
TPU_VERSION="$TPU_VERSION"
GOOGLE_GENAI_USE_VERTEXAI=$USE_VERTEXAI
GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT"
GOOGLE_CLOUD_LOCATION="$GOOGLE_CLOUD_LOCATION"
RAG_CORPUS="$RAG_CORPUS"
INCLUDE_THOUGHTS="$INCLUDE_THOUGHTS"
EOF
    
    print_success "Configuration saved to $CONFIG_FILE"
}


# Function to create eval_config.yaml for both auto_agent and hitl_agent
create_eval_config() {
    print_info "Creating eval_config.yaml for evaluation servers..."
    
    HOSTNAME_IP=$(get_hostname_ip)
    
    if [ -z "$HOSTNAME_IP" ]; then
        print_warning "Could not determine local IP, defaulting to 127.0.0.1"
        HOSTNAME_IP="127.0.0.1"
    fi
    
    # Write eval_config.yaml to server_utils for both agents
    local config_targets=(
        "$REPO_ROOT/auto_agent/server_utils/eval_config.yaml"
        "$REPO_ROOT/hitl_agent/server_utils/eval_config.yaml"
    )

    for target in "${config_targets[@]}"; do
        local target_dir
        target_dir="$(dirname "$target")"
        if [ -d "$target_dir" ]; then
            cat > "$target" << EOF
backends:
  - name: tpu-0
    ip: $HOSTNAME_IP
    port: 5463
    type: tpu
  - name: cpu-0
    ip: $HOSTNAME_IP
    port: 5464
    type: cpu
EOF
        fi
    done

    print_success "Created eval_config.yaml for both auto_agent and hitl_agent (TPU:5463, CPU:5464, IP: $HOSTNAME_IP)"
}


# Main execution
main() {
    print_info "Starting MaxKernel Agent setup..."
    
    # Step 1: Setup Python environment
    # Check if already in a venv (VIRTUAL_ENV is set when venv is active)
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Already in virtual environment: $VIRTUAL_ENV"
    # Check if .venv exists but not activated
    elif [ -d ".venv" ]; then
        # Check if venv is properly set up (has bin/activate)
        if [ -f ".venv/bin/activate" ]; then
            print_info "Found existing .venv, activating..."
            source .venv/bin/activate
        else
            print_warning "Found incomplete .venv directory (missing bin/activate). Removing and recreating..."
            rm -rf .venv
            create_venv
        fi
    # No environment detected, ask user what they want
    else
        set +e  # Temporarily disable exit on error to capture return code
        prompt_env_setup
        env_choice=$?
        set -e  # Re-enable exit on error
        
        if [ $env_choice -eq 0 ]; then
            # Create venv
            create_venv
        else
            # Using own environment
            print_info "Proceeding with your current Python environment..."
        fi
    fi
    
    # Step 2: Install dependencies
    if ! install_dependencies; then
        print_warning "Please restart your terminal and run this script again to continue with the setup"
        exit 0
    fi
    
    # Step 3: Handle environment variables
    load_config
    prompt_env_vars
    save_config
    
    # Export environment variables for current session
    export GEMINI_API_KEY
    export WORKDIR
    export TPU_VERSION
    export GOOGLE_CLOUD_PROJECT
    export GOOGLE_CLOUD_LOCATION
    export RAG_CORPUS
    export INCLUDE_THOUGHTS

    # Step 4: Create eval_config.yaml
    create_eval_config
    
    print_success "Setup completed successfully!"
    print_info "Configuration saved to: $CONFIG_FILE"
    print_info ""
    
    # Provide instructions based on environment setup
    print_info "Next steps:"
    print_info "  If using .venv:    source .venv/bin/activate"
    print_info ""
    print_info "To run HITL Agent:  bash run_hitl_agent.sh"
    print_info "To run Auto Agent:  bash run_auto_agent.sh"
}

# Function to show manual setup steps
show_manual_steps() {
    cat << 'EOF'
================================================================================
MANUAL SETUP STEPS FOR MAXKERNEL AGENTS
================================================================================

STEP 1: Python Environment Setup (.venv)
----------------------------------------
python3 -m venv .venv
source .venv/bin/activate

STEP 2: Install Dependencies
-----------------------------
pip install -r dependency/main_requirements.txt
pip install -r dependency/agent_requirements.txt
pip install -e .

STEP 3: Set Environment Variables
----------------------------------
cat > .env << 'ENVEOF'
GEMINI_API_KEY="your_api_key_here"
WORKDIR="/path/to/your/workdir"
TPU_VERSION="TPU v5e"
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_CLOUD_LOCATION="global"
RAG_CORPUS="projects/tpu-kernel-assist-sandbox/locations/us-west1/ragCorpora/7991637538768945152"
INCLUDE_THOUGHTS="true"
ENVEOF

STEP 4: Create eval_config.yaml
--------------------------------
backends:
  - name: tpu-0
    ip: 127.0.0.1
    port: 5463
    type: tpu
  - name: cpu-0
    ip: 127.0.0.1
    port: 5464
    type: cpu

STEP 5: Run the Agent
----------------------
# HITL Agent:
bash run_hitl_agent.sh

# Auto Agent:
bash run_auto_agent.sh
================================================================================
EOF
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help|--manual]"
        echo "Sets up the MaxKernel environment for HITL and Auto agents"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --manual      Display manual setup instructions without running anything"
        exit 0
        ;;
    --manual)
        show_manual_steps
        exit 0
        ;;
    *)
        main
        ;;
esac