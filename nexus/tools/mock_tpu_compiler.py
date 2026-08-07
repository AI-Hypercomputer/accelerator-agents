#!/usr/bin/env python3
"""
Mock TPU Compiler and Kernel Profiler for Claude PoC.
Simulates Pallas/TPU kernel compilation and execution.
All metrics and error logs are explicitly labeled as Mock / Simulated.
"""
import sys
import os
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/mock_tpu_compiler.py <path_to_kernel_file>")
        sys.exit(1)

    kernel_file = sys.argv[1]
    if not os.path.exists(kernel_file):
        print(f"[Simulated Error] File not found: {kernel_file}")
        sys.exit(1)

    with open(kernel_file, "r") as f:
        content = f.read()

    # Parse BLOCK_SIZE and USE_RING_ATTENTION from the target file
    block_size_match = re.search(r"BLOCK_SIZE\s*=\s*(\d+)", content)
    ring_match = re.search(r"USE_RING_ATTENTION\s*=\s*(True|False)", content)

    block_size = int(block_size_match.group(1)) if block_size_match else 256
    use_ring = (ring_match.group(1) == "True") if ring_match else False

    print("=" * 65)
    print(f"[Mock TPU Compiler] Compiling {os.path.basename(kernel_file)} for TPU v5e...")
    print(f"[Mock Target Config] BLOCK_SIZE={block_size}, USE_RING_ATTENTION={use_ring}")
    print("=" * 65)

    if block_size > 64 or not use_ring:
        print("[Simulated Error] VMEM OOM (Out Of Memory) in Pallas Scratchpad Allocation.")
        print(f"[Simulated Diagnostics] BLOCK_SIZE={block_size} exceeds available VMEM per core when USE_RING_ATTENTION={use_ring}.")
        print("[Mock Recommendation] Reduce BLOCK_SIZE to <= 64 and set USE_RING_ATTENTION = True.")
        sys.exit(1)

    print("[Mock Execution Success] Kernel compiled and verified on mock TPU v5e mesh.")
    print("[Mock Execution Latency] 4.8ms (Simulated - compared to baseline 15.2ms)")
    print("[Mock Metric] Peak VMEM Utilization: 78% (Simulated)")
    sys.exit(0)

if __name__ == "__main__":
    main()
