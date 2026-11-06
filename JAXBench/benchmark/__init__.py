"""JAXBench workload discovery and loading."""

import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS = ('level1', 'level2')


def _get_level_dir(level):
    """Return the benchmark directory for a supported workload level."""
    if level not in LEVELS:
        raise ValueError(f"Unknown workload level '{level}'; choose from {LEVELS}")
    return os.path.join(BENCHMARK_DIR, level)


def list_workloads(level='level1'):
    """Return numerically sorted workload names within the selected level."""
    level_dir = _get_level_dir(level)
    workloads = []
    for d in os.listdir(level_dir):
        full = os.path.join(level_dir, d)
        if (os.path.isdir(full)
                and not d.startswith(('_', '.'))
                and os.path.exists(os.path.join(full, 'baseline.py'))):
            workloads.append(d)
    return sorted(workloads, key=_sort_key)


def get_workload_dir(name, level='level1'):
    """Return absolute path to workload directory. Raises ValueError if not found."""
    level_dir = _get_level_dir(level)
    path = os.path.join(level_dir, name)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'baseline.py')):
        raise ValueError(f"Workload '{name}' not found in {level_dir}")
    return path


def has_optimized(name, level='level1'):
    """Check if a workload has an optimized.py variant."""
    return os.path.exists(os.path.join(_get_level_dir(level), name, 'optimized.py'))


def _sort_key(name):
    """Sort by numeric prefix (1p before 18k, etc.)."""
    num = ''
    for c in name:
        if c.isdigit():
            num += c
        else:
            break
    return int(num) if num else 999
