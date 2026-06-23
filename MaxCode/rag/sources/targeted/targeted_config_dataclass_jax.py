"""
TARGETED JAX PATTERN: Model Config as a Python Dataclass

Every model conversion MUST include a Config dataclass at the top of the file.
This dataclass mirrors the PyTorch model's configuration class and provides
typed, defaulted fields for all hyperparameters. Without it, modules use
`config: Any` which loses type safety, IDE support, and default values.

## WRONG: No Config dataclass, using Any

    class Qwen3NextAttention(nn.Module):
        config: Any  # No type info, no defaults, can't instantiate standalone
        layer_idx: int

    # WHY THIS IS WRONG:
    # - Cannot create a default config for testing: config = ???
    # - No IDE autocomplete for config.hidden_size, config.num_attention_heads
    # - No documentation of what fields the config requires
    # - Cannot validate config values at construction time

## CORRECT: Full Config dataclass with all fields

    import dataclasses
    from typing import Any, Dict, List

    @dataclasses.dataclass
    class Qwen3NextConfig:
        # Vocabulary and embeddings
        vocab_size: int = 151936
        hidden_size: int = 4096
        intermediate_size: int = 22016

        # Attention
        num_attention_heads: int = 32
        num_key_value_heads: int = 32
        head_dim: int = 128
        num_key_value_groups: int = 1

        # Sequence
        max_position_embeddings: int = 32768
        rms_norm_eps: float = 1e-6
        initializer_range: float = 0.02

        # Layer configuration
        num_hidden_layers: int = 32
        layer_types: List[str] = dataclasses.field(
            default_factory=lambda: ["full_attention"] * 32
        )
        rope_parameters: Dict[str, Any] = dataclasses.field(
            default_factory=lambda: {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            }
        )

        # Gated DeltaNet (linear attention)
        gated_delta_rule_chunk_size: int = 64
        v_head_dim: int = 128
        conv_size: int = 4
        num_v_heads: int = 16
        qk_nope_head_dim: int = 128

        # MoE
        num_experts: int = 64
        num_experts_per_tok: int = 4
        decoder_sparse_step: int = 1
        moe_intermediate_size: int = 1408
        shared_expert_intermediate_size: int = 5632
        norm_topk_prob: bool = False
        router_aux_loss_coef: float = 0.001
        output_router_logits: bool = False

        # MLP-only layers
        mlp_only_layers: List[int] = dataclasses.field(default_factory=list)

        # Misc
        attention_bias: bool = False
        attention_dropout: float = 0.0
        hidden_act: str = "silu"
        tie_word_embeddings: bool = True

    # Then use it in modules:
    class Qwen3NextAttention(nn.Module):
        config: Qwen3NextConfig  # Typed, not Any!
        layer_idx: int

## KEY POINTS:
## - ALWAYS include a @dataclasses.dataclass Config class at the top of the file
## - Use dataclasses.field(default_factory=...) for mutable defaults (lists, dicts)
## - Mirror ALL fields from the PyTorch config class
## - Use the Config type (not Any) in module annotations
## - Default values should match the PyTorch model's defaults
"""
