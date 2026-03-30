"""Configuration constants for GPU to JAX conversion agent."""
from tpu_kernel_gen.agents.kernel_gen_agent import constants

MODEL_NAME = constants.MODEL_NAME
MAX_ITERATIONS = constants.MAX_ITERATIONS
LLM_GEN_RETRY_COUNT = constants.LLM_GEN_RETRY_COUNT
TEMPERATURE = constants.TEMPERATURE
TOP_P = constants.TOP_P
TOP_K = constants.TOP_K
CONVERSION_TIMEOUT = 180  # Timeout for conversion attempts
EVAL_SERVER_PORT = constants.EVAL_SERVER_PORT
NUMERICAL_TOLERANCE = 1e-5  # Tolerance for numerical correctness checks

# Backend selection for evaluation
# Options: "cpu", "tpu", or None (use any available backend)
PREFERRED_BACKEND = "cpu"  # Default to CPU for GPU-to-JAX conversion testing

# Supported frameworks for conversion
SUPPORTED_FRAMEWORKS = ["CUDA", "Triton", "PyTorch CUDA", "CuDNN", "cuBLAS"]
