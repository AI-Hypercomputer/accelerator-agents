"""Interface for interacting with embedding models."""

import logging
import time

from google import genai


logger = logging.getLogger("__name__")


class EmbeddingAgent:
  """A class to manage interactions with a Google Gemini model for embeddings."""

  def __init__(self, model_name="models/embedding-001", api_key=None):
    """Initializes the EmbeddingAgent.

    Args:
      model_name: The name of the embedding model to use.
      api_key: The API key for Google AI services. If None, the key is expected
        to be set in the GOOGLE_API_KEY environment variable.
    """
    self.client = genai.Client(api_key=api_key)
    self.model = model_name
    self._cache = {}

  def embed(self, text: str) -> list[float]:
    """Generates an embedding for the given text.

    Args:
      text: The text to embed.

    Returns:
      The embedding vector as a list of floats.
    """
    if text in self._cache:
      return self._cache[text]

    time.sleep(2)
    embedding = (
        self.client.models.embed_content(model=self.model, contents=text)
        .embeddings[0]
        .values
    )
    self._cache[text] = embedding
    return embedding
