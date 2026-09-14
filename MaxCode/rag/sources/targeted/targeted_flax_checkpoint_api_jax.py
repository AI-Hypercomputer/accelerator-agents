"""
TARGETED JAX PATTERN: Flax Checkpoint and TensorBoard APIs

CRITICAL: Several Flax APIs are deprecated or removed in newer versions.
When converting training utilities, use current stable APIs.

## WRONG: Using deprecated flax.training.checkpoints

    # WRONG! This API is deprecated and may be removed.
    from flax.training.checkpoints import save_checkpoint, restore_checkpoint

    save_checkpoint(ckpt_dir, target=state, step=epoch)
    state = restore_checkpoint(ckpt_dir, target=state)

## CORRECT: Use flax.serialization for simple cases

    import flax.serialization

    # Save
    state_bytes = flax.serialization.to_bytes(state)
    with open(path, 'wb') as f:
        f.write(state_bytes)

    # Load
    with open(path, 'rb') as f:
        state_bytes = f.read()
    state = flax.serialization.from_bytes(state, state_bytes)

## CORRECT: Use orbax for production checkpointing

    import orbax.checkpoint as ocp

    # Save
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, state)

    # Load
    state = checkpointer.restore(path, target=state)

## WRONG: Using flax.metrics.tensorboard

    # WRONG! This module may not exist in newer Flax versions.
    from flax.metrics.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)

## CORRECT: Use tensorboardX or standard TensorBoard

    # Option 1: tensorboardX (most common in JAX ecosystem)
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir)
    writer.add_scalar('train/loss', loss_val, step)

    # Option 2: Use the source's TensorBoard pattern faithfully
    # If the PyTorch source uses torch.utils.tensorboard.SummaryWriter,
    # convert to tensorboardX which has the same API:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(tensorboard_dir)
    for name, value in epoch_metrics.items():
        writer.add_scalar(f'train/{name}', float(value), epoch)
    writer.close()

## Why this matters:

1. **Import errors**: Deprecated APIs cause ImportError at runtime, making the
   converted code non-functional without manual fixes.
2. **API stability**: orbax and tensorboardX are the recommended replacements
   and are actively maintained.
3. **Source fidelity**: If the source has TensorBoard logging, the conversion
   should preserve it using the correct JAX-ecosystem equivalent.
"""
