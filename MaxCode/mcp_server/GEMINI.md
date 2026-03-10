# Accelerator Agents Extension - Agent Instructions

This document provides instructions for you, the AI agent, on how to use the
tools provided by the "accelerator-agents-dev" extension.

## Available Tools

The following tools are available via the "dev-server":

### 1. run_agent

*   **Purpose:** Runs the orchestrator agent with a user-provided prompt. The
    agent can use its internal tools, like `migrate_module`, to fulfill the
    request.
*   **Arguments:**
    *   `prompt`: A string to send to the agent.
*   **Usage:** Use this endpoint to interact with the agent.
