# Example Work Directory

This is an example work directory for the HITL Kernel Gen Agent.

When you use agents, they will treat this directory as the root directory. This means that their scope of permissions are limited to this directory and its subdirectories.

## Changing the Work Directory

Make sure you are in the `hitl_kernel_gen_agent` directory:
```bash
cd tpu_kernel_gen/tpu_kernel_gen/agents/hitl_kernel_gen_agent
```

Then, to configure a different work directory, run the setup script:

```bash
bash tpu_kernel_gen/agents/hitl_kernel_gen_agent/prepare_hitl_agent.sh
```

The script will prompt you to set the `WORKDIR` environment variable to your desired location. If you already set your `WORKDIR` environment, directly edit the generated `.env` file.