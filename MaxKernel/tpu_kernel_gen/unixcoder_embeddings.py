from typing import List

import torch
from langchain_core.embeddings.embeddings import Embeddings

from tpu_kernel_gen.unixcoder import UniXcoder


class UniXcoderEmbeddings(Embeddings):
  """UniXcoder embeddings using the UniXcoder model."""

  def __init__(self, model_name: str = "microsoft/unixcoder-base", device: str = None):
    """Initialize UniXcoder embeddings.

    Args:
        model_name: HuggingFace model name for UniXcoder
        device: Device to run the model on (cuda/cpu). If None, auto-detect.
    """
    self.model = UniXcoder(model_name)
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.model.eval()

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Embed search docs.

    Args:
        texts: List of text to embed.

    Returns:
        List of embeddings.
    """
    return self._get_embeddings(texts)

  def embed_query(self, text: str) -> List[float]:
    """Embed query text.

    Args:
        text: Text to embed.

    Returns:
        Embedding.
    """
    embeddings = self._get_embeddings([text])
    return embeddings[0]

  def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    with torch.no_grad():
      # Tokenize the texts
      tokens_ids = self.model.tokenize(texts, mode="<encoder-only>", padding=True)
      source_ids = torch.tensor(tokens_ids).to(self.device)

      # Get sentence embeddings
      _, sentence_embeddings = self.model(source_ids)

      # Convert to list of lists
      embeddings = sentence_embeddings.cpu().numpy().tolist()

    return embeddings
