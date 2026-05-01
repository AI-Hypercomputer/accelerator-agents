"""JAXBench workload discovery and loading."""

import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def list_workloads():
    """Return sorted list of workload names (e.g. ['1p_Flash_Attention', ...])."""
    workloads = []
    for d in os.listdir(BENCHMARK_DIR):
        full = os.path.join(BENCHMARK_DIR, d)
        if (os.path.isdir(full)
                and not d.startswith(('_', '.'))
                and os.path.exists(os.path.join(full, 'baseline.py'))):
            workloads.append(d)
    return sorted(workloads, key=_sort_key)


def get_workload_dir(name):
    """Return absolute path to workload directory. Raises ValueError if not found."""
    path = os.path.join(BENCHMARK_DIR, name)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'baseline.py')):
        raise ValueError(f"Workload '{name}' not found in {BENCHMARK_DIR}")
    return path


def has_optimized(name):
    """Check if a workload has an optimized.py variant."""
    return os.path.exists(os.path.join(BENCHMARK_DIR, name, 'optimized.py'))


def _sort_key(name):
    """Sort by numeric prefix (1p before 18k, etc.)."""
    num = ''
    for c in name:
        if c.isdigit():
            num += c
        else:
            break
    return int(num) if num else 999
