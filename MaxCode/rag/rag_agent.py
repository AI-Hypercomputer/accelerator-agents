"""Tool for performing retrieval augmented generation."""

import os
import sqlite3
from typing import Any, Dict, List

import models
from agents import base
from rag import embedding
from rag import prompts
from rag import vector_db
import numpy as np


# We use a hardcoded character limit for the full code context to avoid
# exceeding the model's token limit. While the Gemini API does not provide a
# way to get the max context length in characters, 20000 characters
# (roughly 5000-7000 tokens) is a safe limit for models with 32k token limits,
# when considering that the prompt sends file content in two fields.
_MAX_CONTEXT_LENGTH = 20000


class RAGAgent(base.Agent):
  """Tool for performing retrieval augmented generation."""

  def __init__(
      self,
      model: Any,
      embedding_model_name: models.EmbeddingModel,
      db_path: str = vector_db.RAG_DB_FILE,
      api_key: str | None = None,
  ):
    """Initializes the agent.

    Args:
      model: The base model to use for generation.
      embedding_model_name: Name of the embedding model to use.
      db_path: Path to the RAG SQLite database.
      api_key: The API key for Google AI services.
    """
    super().__init__(model=model)
    self._db_path = db_path
    self._embedding_agent = embedding.EmbeddingAgent(
        model_name=embedding_model_name.value, api_key=api_key
    )
    vector_db.create_db(db_path)
    (
        self._ids,
        self._names,
        self._texts,
        self._files,
        self._index,
    ) = vector_db.make_embedding_index(db_path)

  def build_from_directory(self, source_path: str):
    """Builds RAG database from files in a source directory."""
    for root, _, files in os.walk(source_path):
      for filename in files:
        if filename.endswith(".py"):
          file_path = os.path.join(root, filename)
          try:
            with open(file_path, "r") as f:
              content = f.read()
            doc_name = os.path.relpath(file_path, source_path)
            print(f"Adding {doc_name} to RAG database...")
            description = self.generate(
                prompts.CODE_DESCRIPTION,
                {
                    "code_block": content,
                    "full_code_context": content[:_MAX_CONTEXT_LENGTH],
                },
            )
            embedding_vector = self._embedding_agent.embed(description)
            vector_db.save_document(
                name=doc_name,
                text=content,
                desc=description,
                file=file_path,
                embedding=np.array(embedding_vector),
                db_path=self._db_path,
            )
          except (OSError, sqlite3.Error) as e:
            print(f"Skipping {file_path}: {e}")
    # Refresh index
    self._ids, self._names, self._texts, self._files, self._index = (
        vector_db.make_embedding_index(self._db_path)
    )
    print("Finished building RAG database.")

  def retrieve_context(
      self, query: str, top_k: int = 3
  ) -> List[Dict[str, Any]]:
    """Retrieves relevant context from the vector DB based on the query.

    Args:
      query: The query string to search for.
      top_k: The number of top results to return.

    Returns:
      A list of dictionaries, each containing 'name', 'text', 'file',
      and 'distance' for a retrieved document.
    """
    query_embedding = self._embedding_agent.embed(query)
    results = vector_db.search_embedding(
        np.array(query_embedding), self._index, self._texts, top_k=top_k
    )
    retrieved_context = []
    for text, distance, i in results:
      retrieved_context.append({
          "name": self._names[i],
          "text": text,
          "file": self._files[i],
          "distance": distance,
      })
    return retrieved_context

  def run(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Runs RAG to retrieve context for a query."""
    return self.retrieve_context(query, top_k)
