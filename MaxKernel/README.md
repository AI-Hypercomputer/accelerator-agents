# Human-in-the-Loop (HITL) Kernel Generation Agent

An intelligent, interactive agent system for generating, optimizing, testing, and profiling TPU kernels with JAX/Pallas. This agent orchestrates a multi-stage workflow that keeps you in control at every step, from initial planning through implementation, testing, and performance optimization.

## Overview

The HITL Kernel Gen Agent provides a conversational interface for TPU kernel development with:

- **Plan-Driven Development**: Creates detailed optimization plans before implementation, allowing you to review and refine the approach
- **Automated Testing**: Generates and executes comprehensive pytest test suites with compilation, correctness, and performance validation
- **Performance Profiling**: Identifies bottlenecks and provides data-driven optimization recommendations
- **GPU-to-JAX Conversion**: Automatically converts CUDA, Triton, and PyTorch GPU code to JAX/Pallas
- **RAG-Enhanced**: Leverages documentation retrieval for accurate, context-aware code generation
- **Safety-First**: Scoped file system access with configurable work directories

## Features

### 🎯 Core Capabilities

1. **Interactive Kernel Planning**
   - Creates detailed optimization plans for Pallas kernels
   - Automatic approval workflow with revision support
   - Includes tiling strategies, memory optimization, and performance targets

2. **Kernel Implementation**
   - Implements kernels following approved plans
   - Supports various optimization techniques (tiling, pipelining, memory management)
   - Generates clean, idiomatic JAX/Pallas code

3. **Comprehensive Testing**
   - Automatic pytest test file generation
   - Compilation validation
   - Numerical correctness testing
   - Performance benchmarking
   - Full traceback reporting for debugging

4. **Performance Profiling**
   - DMA and memory transfer analysis
   - Compute vs memory ratio profiling
   - Bottleneck identification with actionable recommendations

5. **GPU-to-JAX Conversion**
   - Converts CUDA, Triton, PyTorch CUDA code to JAX
   - Strips hardware-specific optimizations
   - Includes syntax validation and numerical correctness testing
   - See [GPU-to-JAX Agent README](gpu_to_jax_agent/README.md) for details

### 🛡️ Safety & Control

- **Scoped Permissions**: Agent operates only within designated work directory
- **User Approval Required**: All implementations require explicit plan approval
- **Transparent Operations**: All file operations are logged and visible
- **Session Persistence**: Save and resume your work across sessions

## Architecture

### Agent Hierarchy

```
KernelGenerationOrchestrationAgent (root_agent)
├── ExplanationAgent - Explains TPU/Pallas concepts
├── PlanKernelAgent - Creates/revises optimization plans
├── ImplementKernelAgent - Implements approved plans
├── ValidatedTestGenerationAgent
│   ├── GenerateTestFileAgent - Creates pytest test files
│   ├── TestValidationLoopAgent - Validates test syntax/structure
│   └── ValidationSummaryAgent - Reports validation results
├── UnifiedTestAgent
│   ├── ReadFileForTestingAgent - Locates test files
│   ├── RunTestsAgent - Executes pytest with server management
│   └── SummarizeTestResultsAgent - Analyzes and reports results
├── ProfileAgentOrchestrator
│   ├── ReadFileForProfilingAgent - Locates kernel files
│   ├── GenerateProfilingScriptAgent - Creates profiling scripts
│   ├── EvalProfileAgent - Executes profiling
│   └── SummarizeProfileAgent - Analyzes bottlenecks
└── GpuToJaxAgent - GPU-to-JAX conversion pipeline
    └── (10-step conversion pipeline - see gpu_to_jax_agent/README.md)
```

### Directory Structure

```
hitl_agent/
├── hitl_agent
│   ├── agent.py                        # Main orchestration logic
│   ├── callbacks.py                    # Agent callbacks
│   ├── config.py                       # Configuration management   
│   ├── constants.py                    # Agent constants   
│   ├── custom_types.py                 # Custom types   
│   ├── dependency                      # Agent dependencies   
│   │   ├── adk_cli_patch.py
│   │   ├── agent_requirements.txt
│   │   └── main_requirements.txt
│   ├── isolate_object.py
│   ├── knowledge_base.                 # Knowledge base for pallas docs
│   │   ├── pallas_docs.py
│   │   └── pallas_profiling_docs.py
│   ├── prompts                         # Main interactive prompt
│   │   └── interactive_prompt.py
│   ├── server_utils                    # Server management utilities
│   │   ├── cpu_server.py
│   │   ├── eval_config.yaml
│   │   ├── eval_server.py
│   │   ├── server_manager_mixin.py
│   │   ├── setup.sh
│   │   └── tpu_server.py
│   ├── subagents                       # Specialized subagents
│   │   ├── explanation                 # Explanation agent
│   │   │   ├── agent.py
│   │   │   └── prompts
│   │   ├── gpu_to_jax_agent            # GPU-to-JAX conversion subagent
│   │   │   ├── README.md
│   │   │   ├── agent.py
│   │   │   ├── constants.py
│   │   │   ├── evaluators
│   │   │   │   ├── compilation_checker.py
│   │   │   │   ├── correctness_checker.py
│   │   │   │   ├── jax_syntax_checker.py
│   │   │   │   └── shape_validator.py
│   │   │   ├── prompts
│   │   │   └── test_agent.py
│   │   ├── kernel_writing              # Kernel planning & implementation
│   │   │   ├── agent.py
│   │   │   ├── kernel_compilation.py
│   │   │   └── prompts
│   │   ├── profiling                  # Performance profiling
│   │   │   ├── agent.py
│   │   │   ├── kernel_profile.py
│   │   │   ├── offline_tools.py
│   │   │   └── prompts
│   │   └── testing                   # Test generation & execution
│   │       ├── agent.py
│   │       └── prompts
│   ├── tests
│   │   ├── conftest.py
│   │   ├── test_compilation_validation_loop.py
│   │   └── test_validate_kernel_compilation_agent.py
│   ├── tools                         # Agent tools
│   │   ├── analyze_profile.py
│   │   ├── api_rag
│   │   │   └── get_apis.py
│   │   ├── search_api_tool.py
│   │   └── tools.py
│   └── tpu_specs.json
├── prepare_hitl_agent.sh            # Hitl agent setup script
├── run_hitl_agent.sh                # Launch script (CLI or UI mode)
└── setup.py
```
## Getting Started

### Prerequisites

1. **Python Environment**: Python 3.9+ with JAX and dependencies installed
2. **Google Cloud**: Vertex AI access for the agent and RAG retrieval
3. **TPU Access**: For actual kernel execution and testing

### Installation

1. **Navigate to the directory of this README file**:

2. **Run the setup script**:
   ```bash
   bash prepare_hitl_agent.sh
   ```

   This script will:
   - Prompt you to choose Python environment setup (Miniconda, venv, or your own)
   - Install required dependencies
   - Set up environment variables
   - Configure your work directory
   - Create the `.env` file with your settings

3. **Configure Environment Variables**:
   
   The setup script creates a `.env` file. You can edit it manually to customize:
   
   ```bash
   # Required
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_GENAI_API_KEY=your-api-key
   
   # Optional - defaults provided
   WORKDIR=/path/to/your/work/directory  # Default: example_workdir
   TPU_VERSION=v5e                        # Default: v5e
   SESSION_ID=hitl_session                # Default: hitl_session
   
   # RAG Configuration (optional)
   VERTEX_AI_RAG_CORPUS=your-corpus-name
   GOOGLE_CLOUD_REGION=us-central1
   ```

### Running the Agent

#### Option 1: CLI Mode (Recommended for Development)

```bash
bash run_hitl_agent.sh
```

**CLI Features**:
- Interactive command-line interface
- Session stored as JSON files (`*.session.json`)
- Easy to debug and inspect
- Lower overhead

**CLI Options**:
```bash
# Start with default session
bash run_hitl_agent.sh

# Start with specific session ID
bash run_hitl_agent.sh --session my_session

# Reset and start fresh
bash run_hitl_agent.sh --reset
```

#### Option 2: Web UI Mode

```bash
bash run_hitl_agent.sh --ui
```

**UI Features**:
- Web interface on port 1430
- Session stored in SQLite database
- Visual conversation history
- Better for demos and non-technical users

**Important**: CLI and UI modes use different session storage mechanisms and **cannot share sessions**.

### Resuming Sessions

**CLI Mode**:
```bash
# Sessions are saved as JSON files
bash run_hitl_agent.sh --session my_session
```

**UI Mode**:
```bash
# Sessions managed through web interface
bash run_hitl_agent.sh --ui
# Browse to http://localhost:1430 and select session
```

## Work Directory Configuration

### What is a Work Directory?

The work directory is where the agent:
- Reads your input files (kernels, GPU code, etc.)
- Writes generated files (plans, implementations, tests, profiles)
- Executes operations (testing, profiling)

**Security**: The agent's file access is **scoped** to this directory - it cannot access files outside it.

### Setting Up Your Work Directory

1. **During Initial Setup**:
   ```bash
   bash prepare_hitl_agent.sh
   # Follow prompts to set WORKDIR
   ```

2. **Manual Configuration**:
   Edit the generated `.env` file:
   ```bash
   WORKDIR=/absolute/path/to/your/work/directory
   ```
