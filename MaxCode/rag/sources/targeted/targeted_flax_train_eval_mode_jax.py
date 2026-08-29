"""
TARGETED JAX PATTERN: Train/Eval Mode in Flax — Use deterministic Flag

CRITICAL: Flax nn.Module objects do NOT have a .train attribute like PyTorch.
Setting model.train = True or model.train = False does nothing in Flax and
will silently produce incorrect behavior. Flax controls train vs eval mode
through a `deterministic` argument passed to __call__.

## WRONG: Setting .train attribute on Flax module (PyTorch habit)

    # WRONG! Flax modules have no .train attribute. This sets a random
    # Python attribute that NO Flax module reads. Dropout, noise, and
    # other stochastic layers will NOT change behavior.
    model = MixtureOfExperts(config)

    # Training loop
    model.train = True   # <-- DOES NOTHING! Silently ignored.
    output = model(x, deterministic=False)

    # Eval loop
    model.train = False  # <-- DOES NOTHING! Silently ignored.
    output = model(x, deterministic=True)

## WRONG: Using PyTorch's model.eval() / model.train() pattern

    # WRONG! Flax modules do not have .eval() or .train() methods.
    # This will raise an AttributeError.
    model.eval()
    model.train()

## CORRECT: Use the deterministic flag on __call__

    # In Flax, train/eval mode is controlled by passing `deterministic`
    # to the module's __call__ method. Each submodule (Dropout, etc.)
    # checks this flag to decide whether to apply stochastic behavior.

    model = MixtureOfExperts(config)

    # Training: deterministic=False enables dropout, noise, etc.
    output = model.apply(
        {'params': params},
        x,
        deterministic=False,
        rngs={'dropout': dropout_rng}
    )

    # Evaluation: deterministic=True disables all stochastic behavior.
    output = model.apply(
        {'params': params},
        x,
        deterministic=True
        # No rngs needed in eval mode
    )

## CORRECT: Training loop pattern

    # The training loop should NOT set any attribute on the model.
    # Instead, pass deterministic=False to train_step and deterministic=True
    # to eval_step via the model.apply call.

    for epoch in range(num_epochs):
        # Training: pass deterministic=False
        for batch in train_loader:
            state, metrics = train_step(state, batch)  # uses deterministic=False internally

        # Evaluation: pass deterministic=True
        for batch in val_loader:
            metrics = eval_step(state, batch)  # uses deterministic=True internally

## Why this matters:

1. **Silent failure**: Setting model.train = True/False creates a new Python attribute
   but no Flax code reads it. The model behaves identically in both cases.
2. **Dropout stays on/off**: Without the deterministic flag, nn.Dropout either always
   drops (if deterministic defaults to False) or never drops. This corrupts training
   dynamics or evaluation metrics.
3. **Router noise**: Routers that add noise during training (for load balancing) use
   the deterministic flag to decide whether to inject noise. Without it, noise is
   either always on (noisy eval) or always off (no exploration during training).
4. **Functional paradigm**: Flax follows JAX's functional style — behavior is controlled
   by function arguments, not by mutable object state.
"""
