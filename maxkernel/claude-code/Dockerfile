# Production Dockerfile for running MaxKernel Claude Code on Trillium (TPU v6e) containers.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MK_ROOT=/accelerator-agents/maxkernel/claude-code

# Install system dependencies and repository prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x and npm
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install Python 3.12 from deadsnakes PPA
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Configure Python 3.12 as default python3 and install pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /accelerator-agents/maxkernel/claude-code

# Copy requirements and install Python dependencies including JAX with Trillium libtpu and anthropic[vertex]
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "anthropic[vertex]" && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source files
COPY . .

# Ensure entrypoint runner is executable
RUN chmod +x tools/run_in_pod.sh

ENTRYPOINT ["/accelerator-agents/maxkernel/claude-code/tools/run_in_pod.sh"]
