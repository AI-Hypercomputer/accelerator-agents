"""TPU hardware specifications and auto-detection."""

TPU_SPECS = {
    'v5e': {
        'peak_tflops_bf16': 197.0,
        'hbm_bandwidth_gbs': 819.0,
    },
    'v6e': {
        'peak_tflops_bf16': 918.0,
        'hbm_bandwidth_gbs': 1600.0,
    },
}


def detect_tpu():
    """Auto-detect TPU type from JAX devices. Returns 'v5e', 'v6e', etc."""
    try:
        import jax
        devices = jax.devices()
        if not devices:
            raise RuntimeError("No JAX devices found")
        kind = devices[0].device_kind.lower()
        for tpu_type in TPU_SPECS:
            if tpu_type in kind:
                return tpu_type
        raise RuntimeError(f"Unknown TPU type: {devices[0].device_kind}")
    except Exception as e:
        raise RuntimeError(f"Could not detect TPU type: {e}")


def get_tpu_spec(tpu='auto'):
    """Return TPU spec dict. If tpu='auto', auto-detects from JAX devices."""
    if tpu == 'auto':
        tpu = detect_tpu()
    if tpu not in TPU_SPECS:
        raise ValueError(f"Unknown TPU type '{tpu}'. Available: {list(TPU_SPECS.keys())}")
    return tpu, TPU_SPECS[tpu]


def get_peak_tflops(tpu='auto'):
    """Return peak TFLOPS for the given TPU target."""
    _, spec = get_tpu_spec(tpu)
    return spec['peak_tflops_bf16']
