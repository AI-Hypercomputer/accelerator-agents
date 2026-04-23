import os
import sys
import uuid
from typing import List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, udf, col, lit
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, MapType

def chunk_python_file(content: str) -> List[Dict[str, Any]]:
    """AST-aware chunking for Python files."""
    import tree_sitter_python
    from tree_sitter import Language, Parser
    
    PY_LANGUAGE = Language(tree_sitter_python.language())
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    
    tree = parser.parse(bytes(content, "utf8"))
    root_node = tree.root_node
    
    chunks = []
    
    # Identify top-level definitions (functions, classes)
    for child in root_node.children:
        if child.type in ["function_definition", "class_definition"]:
            chunk_content = content[child.start_byte:child.end_byte]
            # Only include chunks that have substantial content
            if len(chunk_content.strip()) > 50:
                chunks.append({
                    "code_chunk": chunk_content,
                    "chunk_type": child.type,
                    "metadata": {"start_line": child.start_point[0] + 1, "end_line": child.end_point[0] + 1}
                })
        elif child.type == "comment" and len(content[child.start_byte:child.end_byte]) > 200:
            # Catch large docblocks as well
            chunks.append({
                "code_chunk": content[child.start_byte:child.end_byte],
                "chunk_type": "docblock",
                "metadata": {"start_line": child.start_point[0] + 1, "end_line": child.end_point[0] + 1}
            })
            
    # If no major chunks found or file is small, take the whole file
    if not chunks and len(content.strip()) > 0:
        chunks.append({
            "code_chunk": content,
            "chunk_type": "file",
            "metadata": {"start_line": 1, "end_line": content.count('\n') + 1}
        })
        
    return chunks



def process_file_content(path: str, content: str) -> List[Dict[str, Any]]:
    """Route file to appropriate chunker."""
    try:
        if path.endswith(".py"):
            return chunk_python_file(content)
        else:
            return []
    except Exception as e:
        return [{"code_chunk": content, "chunk_type": "error_fallback", "metadata": {"error": str(e)}}]

def main():
    spark = SparkSession.builder.appName("MultiRepoCodeChunker").getOrCreate()
    
    # Enable recursive file reading for subdirectories
    spark.conf.set("spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive", "true")
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    repo_name = sys.argv[3]
    commit_hash = sys.argv[4]
    branch_name = sys.argv[5]

    # Read only .py files from GCS using wildcard to avoid partition confusion
    df = spark.read.text(input_path + "**/*.py", wholetext=True).withColumnRenamed("value", "content")
    df = df.withColumn("file_path", input_file_name())

    # Define the output schema for the UDF
    schema = ArrayType(StructType([
        StructField("code_chunk", StringType(), False),
        StructField("chunk_type", StringType(), True),
        StructField("metadata", MapType(StringType(), StringType()), True)
    ]))

    # Register UDF
    chunk_udf = udf(lambda path, content: process_file_content(path, content), schema)

    # Process and Explode
    processed_df = df.withColumn("chunks", chunk_udf(col("file_path"), col("content")))
    exploded_df = processed_df.select(
        col("file_path"),
        col("chunks")
    ).select(
        col("file_path"),
        col("chunks.code_chunk").alias("code_chunk"),
        col("chunks.chunk_type").alias("chunk_type"),
        col("chunks.metadata").alias("metadata")
    )

    # Add Repository Metadata
    final_df = exploded_df.withColumn("repository", lit(repo_name)) \
                          .withColumn("commit_hash", lit(commit_hash)) \
                          .withColumn("branch", lit(branch_name))

    # Write to Parquet on GCS
    final_df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    main()
