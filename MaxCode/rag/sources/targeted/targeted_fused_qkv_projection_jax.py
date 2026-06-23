"""
TARGETED RAG: Fused QKV Projection in JAX/Flax
================================================

When converting fairseq-style MultiheadAttention that uses a single
`in_proj_weight` of shape [3*embed_dim, embed_dim] with sliced projection
methods (in_proj_qkv, in_proj_q, in_proj_kv), preserve this fused design
in JAX. Do NOT split into 3 separate nn.Dense layers.

WRONG -- 3 separate Dense layers:
-----------------------------------
class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, query, key, value):
        q = nn.Dense(self.embed_dim, name='q_proj')(query)    # WRONG
        k = nn.Dense(self.embed_dim, name='k_proj')(key)      # WRONG
        v = nn.Dense(self.embed_dim, name='v_proj')(value)     # WRONG
        ...

WHY THIS IS WRONG:
- Breaks weight compatibility with PyTorch checkpoints that store a single
  in_proj_weight tensor of shape [3*D, D]
- Loses the qkv_same_embed_dim / kv_same_embed_dim optimization paths
  where Q,K,V are projected from the same input in a single matmul
- Cannot faithfully represent in_proj_q (query-only), in_proj_kv
  (key+value only) projection methods used for cross-attention

CORRECT -- Single fused [3*D, D] parameter with sliced projection:
-------------------------------------------------------------------
import jax
import jax.numpy as jnp
import flax.linen as nn

class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    kdim: int = None
    vdim: int = None
    add_bias_kv: bool = False
    add_zero_attn: bool = False
    attn_dropout: float = 0.0
    deterministic: bool = False

    def _get_dims(self):
        kdim = self.kdim if self.kdim is not None else self.embed_dim
        vdim = self.vdim if self.vdim is not None else self.embed_dim
        head_dim = self.embed_dim // self.num_heads
        qkv_same = (kdim == self.embed_dim and vdim == self.embed_dim)
        kv_same = (kdim == vdim)
        return kdim, vdim, head_dim, qkv_same, kv_same

    @nn.compact
    def __call__(self, query, key, value, attn_mask=None, need_weights=True):
        kdim, vdim, head_dim, qkv_same, kv_same = self._get_dims()
        bsz = query.shape[1]  # (T, B, D) time-first layout

        # === Fused QKV weight: single [3*D, D] parameter ===
        if qkv_same:
            in_proj_weight = self.param(
                'in_proj_weight',
                nn.initializers.xavier_uniform(),
                (3 * self.embed_dim, self.embed_dim),
            )
            in_proj_bias = self.param(
                'in_proj_bias',
                nn.initializers.zeros_init(),
                (3 * self.embed_dim,),
            )
        else:
            # Separate weights when dims differ (cross-attention)
            q_weight = self.param('q_proj_weight', nn.initializers.xavier_uniform(),
                                  (self.embed_dim, self.embed_dim))
            k_weight = self.param('k_proj_weight', nn.initializers.xavier_uniform(),
                                  (self.embed_dim, kdim))
            v_weight = self.param('v_proj_weight', nn.initializers.xavier_uniform(),
                                  (self.embed_dim, vdim))
            q_bias = self.param('q_proj_bias', nn.initializers.zeros_init(), (self.embed_dim,))
            k_bias = self.param('k_proj_bias', nn.initializers.zeros_init(), (self.embed_dim,))
            v_bias = self.param('v_proj_bias', nn.initializers.zeros_init(), (self.embed_dim,))

        out_proj = nn.Dense(self.embed_dim, name='out_proj',
                            kernel_init=nn.initializers.xavier_uniform())

        # === Sliced projection methods (matching fairseq) ===
        def _in_proj(x, weight, bias, start=0, end=None):
            \"\"\"Project x using a slice of the fused weight and bias.\"\"\"
            w = weight[start:end]
            b = bias[start:end] if bias is not None else None
            out = x @ w.T
            if b is not None:
                out = out + b
            return out

        def in_proj_qkv(x):
            \"\"\"Project Q, K, V from the same input (self-attention).\"\"\"
            D = self.embed_dim
            return (_in_proj(x, in_proj_weight, in_proj_bias, 0, D),
                    _in_proj(x, in_proj_weight, in_proj_bias, D, 2*D),
                    _in_proj(x, in_proj_weight, in_proj_bias, 2*D, 3*D))

        def in_proj_q(x):
            \"\"\"Project Q only (used in cross-attention).\"\"\"
            if qkv_same:
                return _in_proj(x, in_proj_weight, in_proj_bias, 0, self.embed_dim)
            else:
                return x @ q_weight.T + q_bias

        def in_proj_kv(x):
            \"\"\"Project K and V together (used in cross-attention).\"\"\"
            D = self.embed_dim
            if qkv_same:
                return (_in_proj(x, in_proj_weight, in_proj_bias, D, 2*D),
                        _in_proj(x, in_proj_weight, in_proj_bias, 2*D, 3*D))
            elif kv_same:
                return (x @ k_weight.T + k_bias, x @ v_weight.T + v_bias)
            else:
                return (x @ k_weight.T + k_bias, x @ v_weight.T + v_bias)

        # === Usage in forward pass ===
        if qkv_same and (query is key is value):
            # Self-attention: single fused projection
            q, k, v = in_proj_qkv(query)
        else:
            # Cross-attention: separate Q and KV projections
            q = in_proj_q(query)
            k, v = in_proj_kv(key)  # key == value typically

        # Reshape: (T, B, D) -> (B*H, T, head_dim)
        T_q, T_kv = q.shape[0], k.shape[0]
        q = q.reshape(T_q, bsz * self.num_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(T_kv, bsz * self.num_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(T_kv, bsz * self.num_heads, head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn_weights = attn_weights.astype(q.dtype)
        attn_weights = nn.Dropout(rate=self.attn_dropout)(
            attn_weights, deterministic=self.deterministic)

        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 0, 2).reshape(T_q, bsz, self.embed_dim)
        attn_output = out_proj(attn_output)

        if need_weights:
            attn_weights = attn_weights.reshape(bsz, self.num_heads, T_q, T_kv)
            attn_weights = attn_weights.mean(axis=1)  # avg over heads
        return attn_output, attn_weights

KEY POINTS:
-----------
1. Single `in_proj_weight` param of shape [3*embed_dim, embed_dim] -- matches PyTorch
2. Sliced access via in_proj_qkv(), in_proj_q(), in_proj_kv() -- matches fairseq API
3. Falls back to separate weights when kdim != embed_dim or vdim != embed_dim
4. Xavier uniform initialization matches PyTorch's default for MultiheadAttention
5. Weight loading from PyTorch is trivial: just copy in_proj_weight directly
"""
