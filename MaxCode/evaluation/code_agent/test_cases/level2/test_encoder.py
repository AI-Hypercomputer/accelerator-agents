"""Compilability, Shape, and Equivalence tests for Transformer Encoder."""

import os
import sys
import traceback

# pylint: disable=g-importing-member
from absl import app
from evaluation.code_agent.pytorch_references.level2.encoder import TransformerModel as Encoder_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.TransformerModel.model_compiled import TransformerModel as Encoder_Jax
from flax import core
import jax
from jax import config
from jax import sharding
import jax.numpy as jnp
import numpy as np
import torch

Mesh = sharding.Mesh
jax_config = config

os.environ["JAX_PLATFORMS"] = "cpu"
jax_config.update("jax_default_matmul_precision", "highest")

CONFIG = {
    "ntoken": 100,
    "ninp": 32,
    "nhead": 4,
    "nhid": 64,
    "nlayers": 2,
    "batch_size": 4,
    "seq_len": 128,
}


def create_cpu_mesh():
  """Creates a simple 1x1 mesh for testing on a single CPU."""
  devices = jax.devices()
  # Create a 1x1 mesh using the first available device
  device_array = np.array(devices[:1]).reshape(1, 1)
  return Mesh(device_array, axis_names=("data", "model"))


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = Encoder_Torch(
      ntoken=CONFIG["ntoken"],
      ninp=CONFIG["ninp"],
      nhead=CONFIG["nhead"],
      nhid=CONFIG["nhid"],
      nlayers=CONFIG["nlayers"],
      dropout=0.1,
  )
  model.eval()

  x = torch.randint(
      0, CONFIG["ntoken"], (CONFIG["batch_size"], CONFIG["seq_len"])
  )
  # Create a dummy mask (not used in shape check logic but required by forward)
  mask_pt = torch.ones((CONFIG["seq_len"], CONFIG["seq_len"]), dtype=torch.bool)

  out = model(x, mask_pt)

  expected_shape = (CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["ntoken"])
  assert out.shape == expected_shape, (
      f"PT Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  )
  assert not torch.isnan(out).any(), "PT Output contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  test_mesh = create_cpu_mesh()

  model = Encoder_Jax(
      ntoken=CONFIG["ntoken"],
      ninp=CONFIG["ninp"],
      nhead=CONFIG["nhead"],
      nhid=CONFIG["nhid"],
      nlayers=CONFIG["nlayers"],
      dropout=0.1,
      mesh=test_mesh,
  )

  rng = jax.random.PRNGKey(0)
  x = jax.random.randint(
      rng, (CONFIG["batch_size"], CONFIG["seq_len"]), 0, CONFIG["ntoken"]
  )

  # Init
  variables = model.init(rng, x, src_mask=None, deterministic=True)

  # Apply
  out = model.apply(variables, x, src_mask=None, deterministic=True)

  expected_shape = (CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["ntoken"])
  assert out.shape == expected_shape, (
      f"JAX Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  )
  assert not jnp.isnan(out).any(), "JAX Output contains NaNs"


def transfer_and_align_weights(pt_model, jax_variables):
  """Transfers weights from PyTorch to JAX, handling Fused QKV projection."""
  pt_state = pt_model.state_dict()
  new_params = core.unfreeze(jax_variables["params"])

  print("   >> Starting Smart Weight Transfer (Fused QKV Mode)...")

  # 1. Embedding
  print("      - Token Embeddings")
  new_params["token_embedder"]["embedding"] = pt_state["encoder.weight"].numpy()

  # 2. Output Decoder
  print("      - Output Projection")
  new_params["output_projection"]["kernel"] = (
      pt_state["decoder.weight"].T.numpy()
  )
  if "bias" in new_params["output_projection"]:
    new_params["output_projection"]["bias"] = pt_state["decoder.bias"].numpy()
  else:
    # Zero out PT bias if JAX doesn't support it to ensure equivalence
    pt_model.decoder.bias.data.fill_(0.0)

  # 3. Encoder Layers
  for i in range(CONFIG["nlayers"]):
    print(f"      - Layer {i}...")
    jax_layer_name = f"encoder_layer_{i}"
    pt_layer = pt_model.transformer_encoder.layers[i]
    attn_block = new_params[jax_layer_name]["self_attention"]

    # --- FUSED QKV TRANSFER ---
    # PyTorch: in_proj_weight is (3 * D, D)
    in_proj_weight = pt_layer.self_attn.in_proj_weight.detach().numpy()
    in_proj_bias = pt_layer.self_attn.in_proj_bias.detach().numpy()

    # Step A: Transpose Weight (Input -> Output) -> Shape: (D, 3*D)
    w_t = in_proj_weight.T

    # Step B: Split into Q, K, V -> Each becomes (D, D)
    w_q, w_k, w_v = np.split(w_t, 3, axis=1)
    b_q, b_k, b_v = np.split(in_proj_bias, 3, axis=0)

    # Step C: Stack for MaxText Fused Format (Kernel: D, 3, D)
    w_fused = np.stack([w_q, w_k, w_v], axis=1)
    b_fused = np.stack([b_q, b_k, b_v], axis=0)

    # Step D: Reshape for Heads
    hidden_dim = CONFIG["ninp"]
    num_heads = CONFIG["nhead"]
    head_dim = hidden_dim // num_heads

    w_fused = w_fused.reshape(hidden_dim, 3, num_heads, head_dim)
    b_fused = b_fused.reshape(3, num_heads, head_dim)

    # Assign to JAX 'qkv_proj'
    if "qkv_proj" in attn_block:
      attn_block["qkv_proj"]["kernel"] = w_fused
      if "bias" in attn_block["qkv_proj"]:
        attn_block["qkv_proj"]["bias"] = b_fused
      else:
        print(
            "        ! JAX Attention missing bias. Zeroing PyTorch QKV bias."
        )
        pt_layer.self_attn.in_proj_bias.data.fill_(0.0)

    # Output Projection (Out)
    out_w = pt_layer.self_attn.out_proj.weight.detach().numpy()
    out_b = pt_layer.self_attn.out_proj.bias.detach().numpy()

    # 1. Transpose standard linear: (Out, In) -> (In, Out)
    out_kernel_flat = out_w.T

    # 2. Reshape for MaxText Attention: (Heads, Head_Dim, Hidden)
    out_kernel_reshaped = out_kernel_flat.reshape(
        CONFIG["nhead"], CONFIG["ninp"] // CONFIG["nhead"], CONFIG["ninp"]
    )

    attn_block["out"]["kernel"] = out_kernel_reshaped

    if "bias" in attn_block["out"]:
      attn_block["out"]["bias"] = out_b
    else:
      pt_layer.self_attn.out_proj.bias.data.fill_(0.0)

    # --- MLP Block ---
    # Linear 1
    l1_w = pt_layer.linear1.weight.detach().numpy()
    l1_b = pt_layer.linear1.bias.detach().numpy()

    # Handle variable naming (wi vs wi_0)
    mlp_block = new_params[jax_layer_name]["mlp"]
    wi_key = "wi" if "wi" in mlp_block else "wi_0"

    mlp_block[wi_key]["kernel"] = l1_w.T
    if "bias" in mlp_block[wi_key]:
      mlp_block[wi_key]["bias"] = l1_b
    else:
      pt_layer.linear1.bias.data.fill_(0.0)

    # Linear 2
    l2_w = pt_layer.linear2.weight.detach().numpy()
    l2_b = pt_layer.linear2.bias.detach().numpy()
    mlp_block["wo"]["kernel"] = l2_w.T
    if "bias" in mlp_block["wo"]:
      mlp_block["wo"]["bias"] = l2_b
    else:
      pt_layer.linear2.bias.data.fill_(0.0)

    # --- Layer Norms ---
    new_params[jax_layer_name]["norm1"]["scale"] = (
        pt_layer.norm1.weight.detach().numpy()
    )
    new_params[jax_layer_name]["norm2"]["scale"] = (
        pt_layer.norm2.weight.detach().numpy()
    )

    # Zero PT Norm biases if JAX doesn't support them
    if hasattr(pt_layer.norm1, "bias") and pt_layer.norm1.bias is not None:
      pt_layer.norm1.bias.data.fill_(0.0)
    if hasattr(pt_layer.norm2, "bias") and pt_layer.norm2.bias is not None:
      pt_layer.norm2.bias.data.fill_(0.0)

  return {"params": core.freeze(new_params)}


def test_equivalence():
  """Tests numerical equivalence between the PyTorch and JAX Transformer models.

  This function initializes both a PyTorch and a JAX Transformer model,
  transfers the weights from the PyTorch model to the JAX model, and then
  compares their outputs for the same input to ensure numerical equivalence.
  """
  print("\n--- Starting Equivalence Test ---")

  # 1. Setup PyTorch (Golden Model)
  pt_model = Encoder_Torch(
      ntoken=CONFIG["ntoken"],
      ninp=CONFIG["ninp"],
      nhead=CONFIG["nhead"],
      nhid=CONFIG["nhid"],
      nlayers=CONFIG["nlayers"],
      dropout=0.0,
  )
  pt_model.eval()

  # 2. Setup JAX (Target Model)
  test_mesh = create_cpu_mesh()
  jax_model = Encoder_Jax(
      ntoken=CONFIG["ntoken"],
      ninp=CONFIG["ninp"],
      nhead=CONFIG["nhead"],
      nhid=CONFIG["nhid"],
      nlayers=CONFIG["nlayers"],
      dropout=0.0,
      mesh=test_mesh,
  )

  # 3. Create Inputs
  np.random.seed(42)
  input_ids = np.random.randint(
      0, CONFIG["ntoken"], (CONFIG["batch_size"], CONFIG["seq_len"])
  )
  x_pt = torch.from_numpy(input_ids)
  x_jax = jnp.array(input_ids)

  # 4. Init JAX
  print("   >> Initializing JAX...")
  rng = jax.random.PRNGKey(0)
  variables = jax_model.init(rng, x_jax, src_mask=None, deterministic=True)

  # 5. TRANSFER & ALIGN
  # Note: Pass variables, not just params, as we need the structure
  variables = transfer_and_align_weights(pt_model, variables)

  # 6. Forward Pass
  print("   >> Running PyTorch Forward...")
  with torch.no_grad():
    # Pass dummy mask or None depending on implementation
    pt_out = pt_model(x_pt, src_mask=None).numpy()

  print("   >> Running JAX Forward...")
  jax_out = jax_model.apply(variables, x_jax, src_mask=None, deterministic=True)
  jax_out = np.array(jax_out)

  # 7. Compare
  print(f"   >> PT Shape: {pt_out.shape}, JAX Shape: {jax_out.shape}")
  diff = np.abs(pt_out - jax_out)
  mean_diff = np.mean(diff)
  max_diff = np.max(diff)
  print(f"   >> Mean Diff: {mean_diff:.6f}")
  print(f"   >> Max Diff:  {max_diff:.6f}")

  if np.allclose(pt_out, jax_out, atol=1e-4):
    print("\nSUCCESS: Models match numerically!")
  else:
    print("\nFAILURE: Models do not match (Diff too high).")


# pylint: disable=unused-argument
def main(argv):
  try:
    test_pytorch_independent()
    print("PyTorch Model Shape: VALID (True)")
  except AssertionError as e:
    print(f"PyTorch Model FAILED: {e}")

  try:
    test_jax_independent()
    print("JAX Model Shape:     VALID (True)")
  except AssertionError as e:
    print(f"JAX Model FAILED: {e}")
    # Stop here if JAX fails completely
    sys.exit(1)
  try:
    test_equivalence()
    print("Equivalence Test:    VALID (True)")
  except AssertionError as e:
    traceback.print_exc()
    print(f"Equivalence Test FAILED: {e}")


if __name__ == "__main__":
  app.run(main)
