# Accelerator Agents

**Accelerator Agents** is a collection of AI-powered tools designed to accelerate machine learning development on Google Cloud TPUs. This repository hosts agents that assist with code migration, kernel optimization, and performance tuning, enabling developers to leverage the full power of TPUs with greater velocity.

> **Disclaimer:** This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

## Overview

As machine learning models grow in complexity, optimizing them for specific hardware accelerators like TPUs becomes increasingly challenging. This project provides a suite of "Agents"—specialized AI tools powered by Gemini—to automate and assist with these complex tasks.

This initial release includes two primary agents:

### 1. PyTorch to JAX Migration Agent

The **PyTorch to JAX Migration Agent** facilitates the conversion of existing PyTorch models and codebases into JAX. It is designed to help users migrate their workloads to run efficiently on TPUs, leveraging high-performance frameworks like [MaxText](https://github.com/google/maxtext).

**Key Features:**
*   **Automated Conversion:** Converts functional code blocks and model layers from PyTorch to JAX.
*   **MaxText Integration:** Generates JAX code compatible with the MaxText framework for immediate training and inference on TPUs.
*   **Human-in-the-Loop:** Designed to draft initial implementations that developers can review and refine.

### 2. Kernel Agent

The **Kernel Agent**  is a specialized tool for high-performance kernel development on TPUs. It assists engineers in writing, optimizing, and debugging custom kernels, specifically focusing on **Pallas** (JAX's kernel language).

**Key Features:**
*   **Kernel Writing:** Drafts Pallas kernels from scratch or based on JAX reference implementations.
*   **CUDA to Pallas Conversion:** Assists in porting custom CUDA/GPU kernels to run optimally on TPUs.
*   **Optimization & Profiling:** Provides profiling insights and optimization suggestions to improve kernel performance (MFU).
*   **Test Harness Generation:** Automatically generates boilerplate code for correctness testing and compilation checks.

## Getting Started

### Prerequisites
*   A Google Cloud TPU VM (recommended for running the Kernel Agent).
*   Python 3.10+
*   Access to Gemini API (for agent reasoning capabilities).

### Installation

Clone the repository:
```bash
git clone https://github.com/google/accelerator-agents.git
cd accelerator-agents
```

*(Note: Specific installation instructions for each agent can be found in their respective subdirectories.)*

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/contributing.md) for details on how to submit pull requests, report issues, and contribute to the project.

### License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
