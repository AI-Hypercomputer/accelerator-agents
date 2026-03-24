#!/usr/bin/env python3
"""
Kernel Table Generator

A script that uses the kernel parser to find all Pallas kernels in a directory
and generates a CSV table with kernel definitions and call patterns.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from kernel_parser import PallasKernel, PallasKernelFinder


def format_kernel_code(kernel: PallasKernel) -> str:
  """Format kernel definition and call into a single code string."""
  code_parts = []

  # Add kernel definition if available
  if kernel.definition_code:
    code_parts.append("# Kernel Definition:")
    code_parts.append(kernel.definition_code.strip())
    code_parts.append("")

  # Add pallas call
  if kernel.call_code:
    code_parts.append("# Pallas Call:")
    code_parts.append(kernel.call_code.strip())

  return "\n".join(code_parts)


def detect_framework(code: str) -> str:
  """Detect framework based on code content."""
  if "pallas" in code.lower():
    return "jax"
  return "unknown"


def generate_kernel_table(kernels: List[PallasKernel], output_path: Path):
  """Generate a CSV table of kernels with their code and metadata."""

  fieldnames = [
    "file_path",
    "base_directory",
    "kernel_name",
    "call_lines",
    "def_lines",
    "framework",
    "code",
  ]

  # Check if file exists to determine if we should append or create new
  file_exists = output_path.exists()
  mode = "a" if file_exists else "w"

  with open(output_path, mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Only write header if creating a new file
    if not file_exists:
      writer.writeheader()

    for kernel in kernels:
      # Format line ranges
      call_lines = f"{kernel.call_lines.start}-{kernel.call_lines.end}"
      def_lines = ""
      if kernel.definition_lines:
        # Join multiple definition line ranges with semicolons
        line_ranges = [f"{lines.start}-{lines.end}" for lines in kernel.definition_lines]
        def_lines = "; ".join(line_ranges)

      # Extract base directory (first directory component)
      file_path_str = str(kernel.file_path)
      path_parts = Path(file_path_str).parts
      base_directory = path_parts[0] if len(path_parts) > 1 else ""

      # Generate the combined code string
      code = format_kernel_code(kernel)

      # Detect framework
      framework = detect_framework(code)

      row = {
        "file_path": file_path_str,
        "base_directory": base_directory,
        "kernel_name": kernel.function_name,
        "call_lines": call_lines,
        "def_lines": def_lines,
        "framework": framework,
        "code": code,
      }

      writer.writerow(row)


def print_table_summary(kernels: List[PallasKernel], output_path: Path):
  """Print a summary of the generated table."""
  print(f"\n📊 Generated kernel table: {output_path}")
  # Statistics
  with_definitions = sum(1 for k in kernels if k.definition_code)
  base_directories = {}

  for kernel in kernels:
    # Count base directories
    path_parts = Path(str(kernel.file_path)).parts
    base_dir = path_parts[0] if len(path_parts) > 1 else "."
    base_directories[base_dir] = base_directories.get(base_dir, 0) + 1

  print(f"🔧 Kernels with definitions: {with_definitions}/{len(kernels)}")
  print(f"📂 Base directories: {dict(base_directories)}")

  # File distribution
  files = set(k.file_path for k in kernels)
  print(f"📁 Files containing kernels: {len(files)}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate a CSV table of JAX Pallas kernels found in source code",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python kernel_table_gen.py /path/to/project
  python kernel_table_gen.py /path/to/project --output my_kernels.csv
  python kernel_table_gen.py /path/to/project --no-recursive
        """,
  )

  parser.add_argument("directory", type=Path, help="Directory to search for Pallas kernels")

  parser.add_argument(
    "--output",
    type=Path,
    default="pallas_kernels.csv",
    help="Output CSV file (default: pallas_kernels.csv)",
  )

  parser.add_argument(
    "--no-recursive",
    action="store_true",
    help="Do not search subdirectories recursively",
  )

  parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

  args = parser.parse_args()

  # Validate input directory
  if not args.directory.exists():
    print(f"Error: Directory {args.directory} does not exist")
    sys.exit(1)

  if not args.directory.is_dir():
    print(f"Error: {args.directory} is not a directory")
    sys.exit(1)

  # Find kernels using the existing parser
  print(f"🔍 Searching for Pallas kernels in {args.directory}")
  if args.verbose:
    print(f"📂 Recursive: {not args.no_recursive}")

  finder = PallasKernelFinder()
  kernels = finder.find_kernels_in_directory(args.directory, recursive=not args.no_recursive)

  if not kernels:
    print("❌ No Pallas kernels found.")
    sys.exit(0)

  # Generate the CSV table
  print("📝 Generating table...")
  generate_kernel_table(kernels, args.output)

  # Print summary
  print_table_summary(kernels, args.output)

  if args.verbose:
    print("\n📄 Sample entries:")
    for i, kernel in enumerate(kernels[:3]):  # Show first 3
      print(f"  {i + 1}. {kernel.function_name} in {kernel.file_path.name}")


if __name__ == "__main__":
  main()
