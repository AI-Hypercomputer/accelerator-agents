# MaxKernel TPU Configuration & Client Guide

How to point MaxKernel at a TPU, and how to drive the TPU client. Reference
material for use after the Install steps in `README.md`.

Building the Python environment is a separate, one-time job and is **not**
covered here — see the Install section of `README.md`.

Throughout this guide, `$MAXKERNEL` is the path to your clone of this
repository, and the virtualenv is the one the agents run tools with (the
default `install.py` uses; override it with `install.py --venv <path>`):

```bash
export MAXKERNEL=~/maxkernel      # wherever you cloned it
```

--------------------------------------------------------------------------------

## Part 1: TPU Configuration (Local vs Remote VM)

You can run the agent either on a **separate host machine** (accessing a remote TPU VM) or **directly on the TPU VM** (local execution).

### Step 1: Create / Register TPU Configurations

To safely register TPU machines into `tpu_config.json` without risk of race conditions or file corruption, use the `tpu_client.py` CLI command:

```bash
~/maxkernel_venv/bin/python $MAXKERNEL/tools/tpu_client.py --add_tpu '{"tpu_name": "<TPU_NAME>", "zone": "<ZONE>", "project": "<PROJECT>"}'
```
*This command uses file-locking (`fcntl.flock`) and atomic replace (`os.replace`) to safely deduplicate, auto-assign dynamic local ports, and auto-cache hardware specs (`tpu_spec`).*

#### Manual / Initial Config File Examples (`$MAXKERNEL/tpu_config.json`):

##### Option A: Multi-TPU Mode (Pool of Multiple TPUs)
```json
{
    "tpus": [
        {
            "tpu_name": "tpu-v6e-8-1",
            "zone": "us-east5-b",
            "project": "tpu-prod-env-multipod",
            "tpu_version": "TPU v6e",
            "tpu_spec": {
                "device_kind": "TPU v6 lite",
                "device_count": 8
            }
        },
        {
            "tpu_name": "tpu-v6e-8-2",
            "zone": "us-east5-b",
            "project": "tpu-prod-env-multipod",
            "tpu_version": "TPU v6e",
            "tpu_spec": {
                "device_kind": "TPU v6 lite",
                "device_count": 8
            }
        }
    ]
}
```

##### Option B: Remote TPU VM Mode (Single TPU)
```json
{
    "mode": "remote",
    "tpu_name": "<YOUR_TPU_NAME>",
    "zone": "<YOUR_TPU_ZONE>",
    "project": "<YOUR_GCP_PROJECT>",
    "tpu_version": "TPU v6e"
}
```

##### Option C: Local TPU VM Mode (Agent running on the TPU VM directly)
```json
{
    "mode": "local",
    "tpu_version": "TPU v6e"
}
```

### Step 2: Running with `tpu_client.py`

The `tpu_client.py` will automatically read `tpu_config.json` and start the server daemon (locally or via remote SSH/SCP depending on mode). You can also override the target mode using the `--mode` CLI flag (`--mode local` or `--mode remote`).

--------------------------------------------------------------------------------

## Part 2: Async Job Queue & Client Usage

The TPU Server features an asynchronous job queue for handling execution requests without conflicts when the TPU is busy.

### Standard Synchronous Call with Queue Waiting
Submits job to queue and automatically polls until completion:
```bash
~/maxkernel_venv/bin/python $MAXKERNEL/tools/tpu_client.py --action correctness_test --code_file path/to/script.py
```

### Non-blocking Async Submission
Submits job and returns immediately with a `job_id`:
```bash
~/maxkernel_venv/bin/python $MAXKERNEL/tools/tpu_client.py --action autotune --code_file payload.json --submit_only
```

### Checking Job Status & Results
```bash
~/maxkernel_venv/bin/python $MAXKERNEL/tools/tpu_client.py --check_job job_1700000000000_abc123
```

### Inspecting TPU Server Queue
```bash
~/maxkernel_venv/bin/python $MAXKERNEL/tools/tpu_client.py --queue
```
