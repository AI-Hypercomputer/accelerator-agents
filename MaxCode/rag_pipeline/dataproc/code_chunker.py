"""Spark job for AST-aware Python code chunking."""

import sys
from typing import Any, Dict, List

from pyspark.sql import SparkSession
import pyspark.sql.functions as f

# pylint: disable=g-multiple-import
from pyspark.sql.types import ArrayType, MapType, StringType, StructField, StructType


def chunk_python_file(content: str) -> List[Dict[str, Any]]:
  """AST-aware chunking for Python files."""
  # pylint: disable=g-import-not-at-top
  import tree_sitter_python
  from tree_sitter import Language, Parser

  py_language = Language(tree_sitter_python.language())
  try:
    parser = Parser(py_language)
  except TypeError:
    parser = Parser()
    parser.set_language(py_language)

  tree = parser.parse(bytes(content, "utf8"))
  root_node = tree.root_node

  chunks = []

  # Identify top-level definitions (functions, classes)
  for child in root_node.children:
    if child.type in ["function_definition", "class_definition"]:
      chunk_content = content[child.start_byte : child.end_byte]
      # Only include chunks that have substantial content
      if len(chunk_content.strip()) > 50:
        chunks.append({
            "code_chunk": chunk_content,
            "chunk_type": child.type,
            "metadata": {
                "start_line": str(child.start_point[0] + 1),
                "end_line": str(child.end_point[0] + 1),
            },
        })
    elif (
        child.type == "comment"
        and len(content[child.start_byte : child.end_byte]) > 200
    ):
      # Catch large docblocks as well
      chunks.append({
          "code_chunk": content[child.start_byte : child.end_byte],
          "chunk_type": "docblock",
          "metadata": {
              "start_line": str(child.start_point[0] + 1),
              "end_line": str(child.end_point[0] + 1),
          },
      })

  # If no major chunks found or file is small, take the whole file
  if not chunks and len(content.strip()) > 0:
    chunks.append({
        "code_chunk": content,
        "chunk_type": "file",
        "metadata": {
            "start_line": "1",
            "end_line": str(content.count("\n") + 1),
        },
    })

  return chunks


def process_file_content(path: str, content: str) -> List[Dict[str, Any]]:
  """Route file to appropriate chunker."""
  try:
    if path.endswith(".py"):
      return chunk_python_file(content)
    elif path.endswith((".yml", ".yaml")):
      # For YAML configs, keep the whole file as a single chunk
      return [{
          "code_chunk": content,
          "chunk_type": "config",
          "metadata": {
              "start_line": "1",
              "end_line": str(content.count("\n") + 1),
          },
      }]
    else:
      return []
  # pylint: disable=broad-exception-caught
  except Exception as e:
    return [{
        "code_chunk": content,
        "chunk_type": "error_fallback",
        "metadata": {"error": str(e)},
    }]


def main():
  spark = SparkSession.builder.appName("MultiRepoCodeChunker").getOrCreate()

  input_path = sys.argv[1]
  output_path = sys.argv[2]
  repo_name = sys.argv[3]
  commit_hash = sys.argv[4]
  branch_name = sys.argv[5]

  # Read all *.py, *.yml, and *.yaml files recursively from GCS directory
  df = (
      spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true")
      .option("pathGlobFilter", "*.{py,yml,yaml}")
      .load(input_path)
  )

  df = df.withColumn("content", f.col("content").cast("string"))
  df = df.withColumnRenamed("path", "file_path")

  # Define the output schema for the UDF
  schema = ArrayType(
      StructType([
          StructField("code_chunk", StringType(), False),
          StructField("chunk_type", StringType(), True),
          StructField("metadata", MapType(StringType(), StringType()), True),
      ])
  )

  # Register UDF
  chunk_udf = f.udf(process_file_content, schema)

  # Process and Explode
  processed_df = df.withColumn(
      "chunks", chunk_udf(f.col("file_path"), f.col("content"))
  )
  exploded_chunks_df = processed_df.select(
      f.col("file_path"), f.explode(f.col("chunks")).alias("chunk_struct")
  )
  exploded_df = exploded_chunks_df.select(
      f.col("file_path"),
      f.col("chunk_struct.code_chunk").alias("code_chunk"),
      f.col("chunk_struct.chunk_type").alias("chunk_type"),
      f.col("chunk_struct.metadata").alias("metadata"),
  )

  # Add Repository Metadata
  final_df = (
      exploded_df.withColumn("repository", f.lit(repo_name))
      .withColumn("commit_hash", f.lit(commit_hash))
      .withColumn("branch", f.lit(branch_name))
  )

  # Write to Parquet on GCS
  final_df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
  main()
