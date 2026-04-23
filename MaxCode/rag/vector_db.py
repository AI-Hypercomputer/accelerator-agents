"""Functions for managing a SQLite database for RAG."""

import os
import pickle
import sqlite3
import numpy as np

RAG_DB_FILE = os.path.join(os.environ["HOME"], "rag_store.db")


def create_db(db_path: str = RAG_DB_FILE):
  """Create the SQLite database and `documents` table if they do not exist.

  Args:
    db_path: Path to the SQLite database file.
  """
  conn = sqlite3.connect(db_path)
  cur = conn.cursor()
  cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        text TEXT NOT NULL,
        desc TEXT NOT NULL,
        file TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """)
  conn.commit()
  conn.close()


def save_document(
    name: str,
    text: str,
    desc: str,
    file: str,
    embedding: np.ndarray,
    db_path: str = RAG_DB_FILE,
):
  """Insert a document and its embedding into the database.

  Args:
      name: Logical name/identifier for the document.
      text: Raw text content of the document.
      desc: Short description or summary of the document.
      file: File path or source identifier for the document.
      embedding: Dense vector representation of the document with shape (dim,)
        and dtype convertible to float32.
      db_path: Path to the SQLite database file.
  """
  conn = sqlite3.connect(db_path)
  cur = conn.cursor()
  emb_binary = pickle.dumps(embedding.astype(np.float32))
  cur.execute(
      "INSERT INTO documents (name,text,desc,file, embedding) VALUES (?,"
      " ?,?,?,?)",
      (name, text, desc, file, emb_binary),
  )
  conn.commit()
  conn.close()


def load_all_documents(
    db_path: str = RAG_DB_FILE,
) -> tuple[list[int], list[str], list[str], list[str], np.ndarray]:
  """Load all documents and embeddings from the database.

  Args:
    db_path: Path to the SQLite database file.

  Returns:
      tuple[list[int], list[str], list[str], list[str], numpy.ndarray]:
          - ids: Row IDs for each document.
          - names: Names for each document.
          - texts: Text content for each document.
          - files: Source file paths/identifiers.
          - embeddings: Array of shape (num_docs, dim) with dtype float32.
  """
  conn = sqlite3.connect(db_path)
  cur = conn.cursor()
  cur.execute("SELECT id,name, text,file, embedding FROM documents")
  rows = cur.fetchall()
  conn.close()

  ids, names, texts, files, embeddings = [], [], [], [], []
  for r in rows:
    ids.append(r[0])
    names.append(r[1])
    texts.append(r[2])
    files.append(r[3])
    embeddings.append(pickle.loads(r[4]))
  return ids, names, texts, files, np.array(embeddings, dtype=np.float32)


def build_numpy_index(embeddings: np.ndarray) -> np.ndarray | None:
  """Return embeddings if valid, else None."""
  if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    return None
  return embeddings


def search_embedding(
    query_embedding: np.ndarray,
    index: np.ndarray,
    texts: list[str],
    top_k: int = 3,
) -> list[tuple[str, float, int]]:
  """Search the index for nearest neighbors to a query embedding using numpy.

  Args:
      query_embedding: Vector of shape (dim,) convertible to float32.
      index: A numpy array of document embeddings (num_docs, dim).
      texts: Texts aligned with vectors in the index.
      top_k: Number of nearest neighbors to retrieve.

  Returns:
      list[tuple[str, float, int]]: For each neighbor, a tuple of (text,
      distance, index_in_corpus).
        Distances are squared L2 (Euclidean) norms; smaller values indicate
        greater similarity.
  """
  if index is None:
    return []
  # Calculate squared L2 distance
  distances = np.sum((index - query_embedding) ** 2, axis=1)
  # Get top_k indices
  top_k_indices = np.argsort(distances)[:top_k]
  results = [
      (texts[i], distances[i], i) for i in top_k_indices if i < len(texts)
  ]
  return results


def make_embedding_index(
    db_path: str = RAG_DB_FILE,
) -> tuple[list[int], list[str], list[str], list[str], np.ndarray | None]:
  """Load all documents and return embeddings as index.

  Args:
    db_path: Path to the SQLite database file.

  Returns:
      tuple[list[int], list[str], list[str], list[str], np.ndarray | None]:
          (ids, names, texts, files, index)
  """
  ids, names, texts, files, embeddings = load_all_documents(db_path)
  index = build_numpy_index(embeddings)
  return ids, names, texts, files, index
