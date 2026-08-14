# TPU VM Management and Troubleshooting

This document tracks the status of TPU VMs and provides troubleshooting
guidelines for resource conflicts.

## TPU VM Status Tracking

When using a TPU VM, update this table to reflect the current usage. If you are
using a VM, set "In Use" to `True` and record your `conversation_id` or
`agent_id`. When you are finished, set it back to `False` and clear the ID.

TPU Name            | SSH Command                                                                                          | In Use  | Current User (Conv/Agent ID)
:------------------ | :--------------------------------------------------------------------------------------------------- | :------ | :---------------------------

*Note: If a subagent attempts to use the TPU, they must update this table with
their active conversation ID or agent ID.*

## Troubleshooting

### TPU Already in Use Error

If you encounter the following error during execution:

```
RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid <PID>. Not attempting to load libtpu.so in this process.
```

This indicates that another process is currently utilizing the TPU hardware.

#### Strict Rules:

1.  **NEVER attempt to kill the occupying process** (e.g., do NOT run `kill`,
    `killall`, or `fuser` commands to free up the TPU).
2.  **Abandon the current VM**: Immediately stop trying to use this TPU VM.
3.  **Fallback to next available VM**: Check the tracking table above for the
    next available TPU VM (where "In Use" is `False`), update its status, and
    switch your execution to that VM. If no other VMs are available, report the
    conflict to the user.
4.  **Release on Completion**: When you finish using a TPU VM (either upon
    successful completion of the workflow or if you abort due to errors), you
    must immediately update the tracking table to set "In Use" to `False` and
    clear your ID.
