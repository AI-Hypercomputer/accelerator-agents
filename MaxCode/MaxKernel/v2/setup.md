<!-- disableFinding(LINE_OVER_80) -->

# MaxKernel TPU VM & Local Setup Guide

This guide describes how to set up the Python virtual environment
(`maxkernel_venv`) and install dependencies on both your **local Cloudtop** and
the **TPU VM**.

--------------------------------------------------------------------------------

## Part 1: Local Cloudtop Setup

Perform these steps on your local Cloudtop.

### Step 1: Verify Python Version

Ensure you have Python 3.12 installed:

```bash
python3 --version
```

If you need to install it:

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

### Step 2: Verify if Virtual Environment Exists

Run the following command to check if `maxkernel_venv` already exists and is
valid:

```bash
if [ -x ~/maxkernel_venv/bin/python3 ]; then echo "✅ maxkernel_venv exists"; else echo "❌ maxkernel_venv NOT found"; fi
```

If it does not exist, create it:

```bash
python3.12 -m venv ~/maxkernel_venv
```

### Step 3: Install Dependencies

Activate the virtual environment and install the required packages:

```bash
# Activate the venv
source ~/maxkernel_venv/bin/activate

# Upgrade pip
pip install --upgrade pip --index-url https://pypi.org/simple

# Install dependencies from the combined requirements.txt
pip install -r third_party/py/accelerator_agents/MaxKernel/v2/requirements.txt --index-url https://pypi.org/simple
```

### Step 4: Verification

Verify the installation:

```bash
# 1. Verify we are using the venv python
which python3
# Expected: /usr/local/google/home/<username>/maxkernel_venv/bin/python3 (or similar path under your home)

# 2. Verify JAX version is 0.11.0
python3 -c "import jax; print(jax.__version__)"
# Expected: 0.11.0
```

--------------------------------------------------------------------------------

## Part 2: TPU VM Setup

Perform these steps after SSH-ing into the TPU VM.

### Step 1: SSH into TPU VM

Ask the user what command to use to SSH into the TPU VM, then replace the `<TPU_SSH_COMMAND>` variable placeholder with the user's command.

Use the following command to SSH into the designated TPU VM:

```bash
<TPU_SSH_COMMAND>
```

### Step 2: Verify if Virtual Environment Exists

On the TPU VM, run the same check:

```bash
if [ -x ~/maxkernel_venv/bin/python3 ]; then echo "✅ maxkernel_venv exists"; else echo "❌ maxkernel_venv NOT found"; fi
```

If it does not exist, create it (Python 3.12 is required):

```bash
# Ensure Python 3.12 and venv module are installed on VM
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Create the virtual environment
python3.12 -m venv ~/maxkernel_venv
```

### Step 3: Create requirements.txt on TPU VM

Create the `requirements.txt` file in your home directory on the TPU VM:

```bash
cat > ~/requirements.txt << 'EOF'
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
torch
google-cloud-bigquery
langchain-google-community
tensorflow
transformers
jax[tpu]==0.11.0
matplotlib
pandas
tabulate
protobuf
google-adk~=1.0
pygithub
langchain-community
google-cloud-aiplatform[adk,agent_engines]
cloudpickle
xprof
tpu-info
pytest-asyncio
google-genai
PyYAML
EOF
```

### Step 4: Install Dependencies

Activate the virtual environment and install the required packages:

```bash
# Activate the venv
source ~/maxkernel_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from the newly created requirements.txt
pip install -r ~/requirements.txt
```

### Step 5: Verification

Verify the installation on the TPU VM:

```bash
# 1. Verify we are using the venv python
which python3
# Expected: /home/cathygao/maxkernel_venv/bin/python3

# 2. Verify JAX version is 0.11.0
python3 -c "import jax; print(jax.__version__)"
# Expected: 0.11.0
```
