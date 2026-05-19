"""Dynamic module loading utilities."""

import importlib.util
import os
import sys


def load_module(path, module_name=None):
    """Import a Python file as a module.

    Registers in sys.modules to fix dataclass __module__ resolution.
    Returns the loaded module.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found: {path}")

    if module_name is None:
        module_name = os.path.splitext(os.path.basename(path))[0]

    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_kernel(path):
    """Load an agent-provided kernel file.

    The file must export a workload(*inputs) function.
    Returns the loaded module.
    """
    mod = load_module(path, f"agent_kernel_{os.path.basename(path)}")

    if not hasattr(mod, 'workload'):
        raise ValueError(
            f"Kernel file {path} must define a workload(*inputs) function"
        )

    if not callable(mod.workload):
        raise ValueError(
            f"workload in {path} is not callable"
        )

    return mod
