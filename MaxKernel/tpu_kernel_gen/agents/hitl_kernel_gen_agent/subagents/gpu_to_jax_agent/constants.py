"""Configuration constants for GPU to JAX conversion agent."""

MODEL_NAME = "gemini-3-pro-preview"
MAX_ITERATIONS = 5
LLM_GEN_RETRY_COUNT = 3
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 5
CONVERSION_TIMEOUT = 180  # Timeout for conversion attempts
EVAL_SERVER_PORT = 1245
NUMERICAL_TOLERANCE = 1e-5  # Tolerance for numerical correctness checks

# Backend selection for evaluation
# Options: "cpu", "tpu", or None (use any available backend)
PREFERRED_BACKEND = "cpu"  # Default to CPU for GPU-to-JAX conversion testing

# Supported frameworks for conversion
SUPPORTED_FRAMEWORKS = ["CUDA", "Triton", "PyTorch CUDA", "CuDNN", "cuBLAS"]
