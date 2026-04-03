import argparse
import json
import os
import shutil

import pandas as pd
import torch
from tqdm import tqdm

from tpu_kernel_gen.unixcoder import UniXcoder


def load_model(model_name="microsoft/unixcoder-base"):
  """Load and initialize the UniXcoder model"""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  model = UniXcoder(model_name)
  model.to(device)
  return model, device


def get_code_embedding(model, device, code_text, max_length=512):
  """Get embedding for a single piece of code"""
  try:
    # Tokenize and encode
    tokens_ids = model.tokenize([code_text], max_length=max_length, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, code_embedding = model(source_ids)

    # Normalize embedding
    norm_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)

    # Convert to numpy and flatten
    return norm_embedding.cpu().detach().numpy().flatten()

  except Exception as e:
    print(f"Error processing code: {e}")
    return None


def process_csv_with_embeddings(
  input_path,
  content_column,
  model_name="microsoft/unixcoder-base",
  max_length=512,
  batch_size=1,
  backup=True,
):
  """
  Process CSV file and add embedding columns in place

  Args:
      input_path: Path to input CSV file (will be modified)
      content_column: Name of column containing code
      model_name: UniXcoder model name
      max_length: Maximum token length for encoding
      batch_size: Batch size for processing (currently only supports 1)
      backup: Whether to create a backup of the original file
  """

  # Create backup if requested
  if backup:
    backup_path = input_path + ".backup"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(input_path, backup_path)

  # Load model
  print("Loading UniXcoder model...")
  model, device = load_model(model_name)

  # Load CSV
  print(f"Loading CSV from {input_path}...")
  df = pd.read_csv(input_path)

  if content_column not in df.columns:
    raise ValueError(f"Column '{content_column}' not found in CSV. Available columns: {list(df.columns)}")

  # Check if embeddings already exist
  if "embedding" in df.columns:
    print("Found existing embedding column")
    response = input("Embedding already exists. Overwrite? (y/N): ").strip().lower()
    if response != "y":
      print("Operation cancelled.")
      return
    # Remove existing embedding column and metadata
    df = df.drop(columns=["embedding"])
    metadata_cols = ["embedding_model"]
    existing_metadata = [col for col in metadata_cols if col in df.columns]
    if existing_metadata:
      df = df.drop(columns=existing_metadata)

  print(f"Processing {len(df)} rows...")

  # Process embeddings
  embeddings = []
  failed_indices = []

  for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
    code_text = str(row[content_column])
    code_text = str(row[content_column])

    embedding = get_code_embedding(model, device, code_text, max_length)

    if embedding is not None:
      # Convert numpy array to list and then to JSON string
      embedding_json = json.dumps(embedding.tolist())
      embeddings.append(embedding_json)
    else:
      # Handle failed embeddings by using zeros
      print(f"Failed to process row {idx}, using zero embedding")
      failed_indices.append(idx)
      # Use a zero embedding with the same dimension as successful ones
      if embeddings:
        # Get dimension from first successful embedding
        first_embedding = json.loads(embeddings[0])
        zero_embedding = [0.0] * len(first_embedding)
        embeddings.append(json.dumps(zero_embedding))
      else:
        # If this is the first row and it fails, we need to determine embedding size
        # Try with a simple example to get the dimension
        dummy_embedding = get_code_embedding(model, device, "def dummy(): pass", max_length)
        if dummy_embedding is not None:
          zero_embedding = [0.0] * len(dummy_embedding)
          embeddings.append(json.dumps(zero_embedding))
        else:
          raise RuntimeError("Could not determine embedding dimension")

  # Add embedding column to DataFrame
  if embeddings:
    # Get embedding dimension from first embedding
    first_embedding = json.loads(embeddings[0])
    embedding_dim = len(first_embedding)
    print(f"Embedding dimension: {embedding_dim}")

    # Add embedding column
    df["embedding"] = embeddings

    # Add metadata columns
    df["embedding_model"] = model_name

    # Save result back to original file
    print(f"Saving results back to {input_path}...")
    df.to_csv(input_path, index=False)

    print(f"Successfully processed {len(df)} rows")
    print(f"Failed embeddings: {len(failed_indices)}")
    print(f"File updated: {input_path}")
    if backup:
      print(f"Backup saved: {backup_path}")

  else:
    raise RuntimeError("No embeddings could be computed")


def main():
  parser = argparse.ArgumentParser(description="Add code embeddings to CSV file using UniXcoder")
  parser.add_argument("input_csv", help="Path to input CSV file (will be modified in place)")
  parser.add_argument(
    "--content_column",
    help="Name of column containing code",
  )
  parser.add_argument("--model", default="microsoft/unixcoder-base", help="UniXcoder model name")
  parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Maximum token length (default: 512)",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for processing (default: 1)",
  )
  parser.add_argument(
    "--no-backup",
    action="store_true",
    help="Don't create a backup of the original file",
  )

  args = parser.parse_args()

  # Validate input file exists
  if not os.path.exists(args.input_csv):
    raise FileNotFoundError(f"Input file not found: {args.input_csv}")

  # Process the CSV
  process_csv_with_embeddings(
    args.input_csv,
    args.content_column,
    args.model,
    args.max_length,
    args.batch_size,
    backup=not args.no_backup,
  )


if __name__ == "__main__":
  main()
