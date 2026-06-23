"""
TARGETED JAX PATTERN: No Explicit Initializer for Bare nn.Linear / nn.Conv1d

CRITICAL: When converting bare PyTorch layers that use only framework defaults
(no explicit nn.init call), the JAX conversion must NOT add explicit initializer
arguments. Flax defaults (lecun_normal for kernel, zeros for bias) are the
accepted equivalent of PyTorch defaults (kaiming_uniform for weight, uniform for
bias). Adding explicit kaiming_uniform or uniform locks in a specific
initialization that may not match downstream usage.

## WRONG: Adding explicit kaiming_uniform to bare nn.Conv1d

    # PyTorch source:
    #   self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    #   (no nn.init call anywhere for conv1)

    # WRONG! Source uses the default init, but conversion adds explicit kaiming.
    conv1 = nn.Conv(
        features=out_channels,
        kernel_size=(1,),
        use_bias=False,
        kernel_init=nn.initializers.kaiming_uniform(),  # NOT in source!
    )

## WRONG: Adding explicit kaiming_uniform and uniform to bare nn.Linear

    # PyTorch source:
    #   self.fc = nn.Linear(in_features, out_features)
    #   (no nn.init call anywhere for fc)

    # WRONG! Source uses the default init, but conversion adds explicit inits.
    fc = nn.Dense(
        features=out_features,
        kernel_init=nn.initializers.kaiming_uniform(),  # NOT in source!
        bias_init=nn.initializers.uniform(),            # NOT in source!
    )

## WRONG: Adding explicit kaiming_uniform to a gate projection

    # PyTorch source:
    #   self.gate = nn.Linear(hidden_size, num_heads, bias=False)
    #   (no nn.init call)

    # WRONG!
    gate = nn.Dense(
        features=num_heads,
        use_bias=False,
        kernel_init=nn.initializers.kaiming_uniform(),  # NOT in source!
    )

## CORRECT: Bare nn.Conv1d -> bare nn.Conv (no explicit init args)

    # PyTorch source:
    #   self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    # CORRECT: No explicit initializer. Flax default (lecun_normal) is the
    # accepted equivalent of PyTorch's default (kaiming_uniform).
    conv1 = nn.Conv(
        features=out_channels,
        kernel_size=(1,),
        use_bias=False,
    )

## CORRECT: Bare nn.Linear -> bare nn.Dense (no explicit init args)

    # PyTorch source:
    #   self.fc = nn.Linear(in_features, out_features)

    # CORRECT: No explicit initializer. Flax defaults (lecun_normal for kernel,
    # zeros for bias) are the accepted equivalent of PyTorch's defaults.
    fc = nn.Dense(features=out_features)

## CORRECT: Only use explicit init when the source explicitly initializes

    # PyTorch source HAS an explicit init call:
    #   self.fc = nn.Linear(in_features, out_features)
    #   nn.init.xavier_uniform_(self.fc.weight)
    #   nn.init.zeros_(self.fc.bias)

    # CORRECT: Mirror the explicit init from source.
    fc = nn.Dense(
        features=out_features,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros_init(),
    )

## Why this matters:

1. **PyTorch default != Flax default, but both are accepted**: PyTorch uses
   kaiming_uniform by default; Flax uses lecun_normal. These are DIFFERENT
   distributions, but both are reasonable defaults. Adding explicit kaiming
   to Flax code locks in a specific choice the source author never made.
2. **Bare layers signal "use framework default"**: When the source writes
   `nn.Linear(in, out)` with no init call, the intent is "use whatever the
   framework provides". The JAX equivalent of that intent is `nn.Dense(out)`
   with no init args.
3. **Explicit init adds noise to verification**: Adding kaiming_uniform gets
   flagged as a deviation from source faithfulness, even though the source
   never specified any initializer.
4. **Weight loading overrides init anyway**: For inference or fine-tuning from
   pretrained weights, the initializer is irrelevant because weights are loaded
   from a checkpoint. Adding an explicit init is pure noise.
5. **Rule of thumb**: Only add kernel_init / bias_init to nn.Dense or nn.Conv
   when the PyTorch source has an explicit nn.init.* call for that parameter.
"""
