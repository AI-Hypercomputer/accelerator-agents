# Accelerator Agents Extension - Agent Instructions

This document provides instructions for you, the AI agent, on how to use the
tools provided by the "accelerator-agents-dev" extension.

## Available Tools

The following tools are available via the "dev-server":

### 1. run_migration_workflow

*   **Purpose:** Runs the migration agent to migrate code from PyTorch to JAX.
*   **Arguments:**
    *   `prompt`: A string to send to the agent.
*   **Usage:** Use this endpoint to migrate code from PyTorch to JAX.

### 2. run_evaluation_workflow

*   **Purpose:** Runs the evaluation agent to generate model configurations,
    generate Oracle evaluation data, or write equivalence tests.
*   **Arguments:**
    *   `prompt`: A string to send to the agent.
*   **Usage:** Use this endpoint to generate model configurations, generate
    Oracle evaluation data, or write equivalence tests.
