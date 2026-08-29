"""
TARGETED RAG: Preserve .detach() as jax.lax.stop_gradient() in JAX/Flax
=========================================================================

When converting PyTorch code that calls .detach() on a tensor, you MUST
use jax.lax.stop_gradient() in the JAX version. Omitting this changes
the gradient flow and training dynamics.

This is especially common for:
- Positional embeddings (sinusoidal or learned) that should not receive gradients
- Target values in loss computation
- Codebook entries in VQ-VAE
- Teacher outputs in knowledge distillation

WRONG -- Omitting stop_gradient when source uses .detach():
------------------------------------------------------------
    # PyTorch source:
    #   def forward(self, input):
    #       ...
    #       return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    # WRONG! Missing stop_gradient -- gradients will flow through positional embeddings
    def __call__(self, input):
        ...
        return weights[positions]

WHY THIS IS WRONG:
- .detach() in PyTorch severs the tensor from the computation graph
- Without it, gradients propagate back through the embedding lookup
- For sinusoidal positional embeddings this is especially wrong because:
  1. The embeddings are deterministic functions of position, not learnable
  2. Gradient flow through them wastes compute and can cause instability
  3. The PyTorch source author explicitly chose to block gradients here
- Omitting .detach() silently changes training behavior with no error or warning

CORRECT -- Use jax.lax.stop_gradient() wherever source uses .detach():
-----------------------------------------------------------------------
    # CORRECT: stop_gradient preserves the .detach() semantics
    def __call__(self, input):
        ...
        return jax.lax.stop_gradient(weights[positions])

PATTERN MATCHING:
-----------------
When you see ANY of these patterns in PyTorch, add jax.lax.stop_gradient():

  PyTorch pattern 1: `tensor.detach()`
  JAX equivalent:    `jax.lax.stop_gradient(tensor)`

  PyTorch pattern 2: `tensor.detach().clone()`
  JAX equivalent:    `jax.lax.stop_gradient(tensor).copy()`

  PyTorch pattern 3: `with torch.no_grad(): result = ...`
  JAX equivalent:    `result = jax.lax.stop_gradient(...)`

  PyTorch pattern 4: `x.data`  (accessing raw data, no grad tracking)
  JAX equivalent:    `jax.lax.stop_gradient(x)`

FULL EXAMPLE -- Sinusoidal Positional Embedding:
-------------------------------------------------
    # PyTorch source:
    class SinusoidalPositionalEmbedding(nn.Module):
        def forward(self, input):
            bsz, seq_len = input.size()
            max_pos = self.padding_idx + 1 + seq_len
            weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
            positions = make_positions(input, self.padding_idx, self.left_pad)
            return weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    # CORRECT JAX conversion:
    class SinusoidalPositionalEmbedding(nn.Module):
        embedding_dim: int
        padding_idx: int = 0
        left_pad: int = 0

        @nn.compact
        def __call__(self, input):
            bsz, seq_len = input.shape
            max_pos = self.padding_idx + 1 + seq_len
            weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
            positions = make_positions(input, self.padding_idx, self.left_pad)
            # CRITICAL: preserve .detach() as stop_gradient
            return jax.lax.stop_gradient(weights[positions.reshape(-1)].reshape(bsz, seq_len, -1))

RULE: Every .detach() in the source MUST become a jax.lax.stop_gradient() in JAX.
This is not optional -- it changes the mathematical gradient computation.
"""
