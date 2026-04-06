"""LLM models for agent use."""

import enum
import logging
import random
import time
import requests


class GeminiModel(enum.Enum):
  """Enum for Gemini model names."""

  GEMINI_2_5_PRO = "gemini-2.5-pro"
  GEMINI_2_5_FLASH = "gemini-2.5-flash"
  GEMINI_3_0_PRO = "gemini-3.0-pro"
  GEMINI_3_0_FLASH = "gemini-3-flash-preview"
  GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"


class EmbeddingModel(enum.Enum):
  """Enum for Embedding model names."""

  EMBEDDING_001 = "models/embedding-001"
  GEMINI_EMBEDDING_001 = "models/gemini-embedding-001"
  TEXT_EMBEDDING_004 = "models/text-embedding-004"


class GeminiTool:
  """Tool for interacting with the Gemini API."""

  def __init__(
      self,
      model_name: GeminiModel | str = GeminiModel.GEMINI_3_0_FLASH,
      system_instruction=None,
      api_key=None,
  ):
    """Initializes the GeminiTool with a specific system instruction.

    Args:
      model_name: The Gemini model to use.
      system_instruction (str, optional): The system instruction to prepend to
        user prompts. Defaults to None.
      api_key: The API key for Google AI services. If None, the key is expected
        to be set in the GOOGLE_API_KEY environment variable.
    """
    if not api_key:
      raise ValueError("API key must be provided for GeminiTool.")
    if isinstance(model_name, str):
      try:
        self.model_name = GeminiModel(model_name)
      except ValueError as exc:
        raise ValueError(
            f"Invalid model name string: {model_name}. Must be one of"
            f" {[m.value for m in GeminiModel]}"
        ) from exc
    else:
      self.model_name = model_name
    self.system_instruction = system_instruction
    self.api_key = api_key
    self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name.value}:generateContent?key={self.api_key}"

  def __call__(self, user_prompt: str):
    """Generates a response from the Gemini API based on the user prompt.

    Args:
      user_prompt (str): The user's prompt.

    Returns:
        str: The generated text response from the Gemini API.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": user_prompt}], "role": "user"}]}
    if self.system_instruction:
      payload["system_instruction"] = {
          "parts": [{"text": self.system_instruction}]
      }

    max_retries = 5
    for attempt in range(max_retries):
      try:
        time.sleep(2 if attempt == 0 else min(30, 2 ** (attempt + 1)))
        response = requests.post(
            self.endpoint, headers=headers, json=payload, timeout=600
        )
        response.raise_for_status()
        json_response = response.json()
        return json_response["candidates"][0]["content"]["parts"][0]["text"]
      except requests.exceptions.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (429, 500, 503) and attempt < max_retries - 1:
          wait = min(60, 2 ** (attempt + 2)) + random.uniform(0, 2)
          logging.warning(
              "Gemini API %s, retrying in %.1fs (attempt %d/%d)...",
              status,
              wait,
              attempt + 1,
              max_retries,
          )
          time.sleep(wait)
          continue
        logging.exception("Error calling Gemini API: %s", e)
        raise
      except (KeyError, IndexError) as e:
        logging.exception("Error parsing Gemini API response: %s", e)
        raise ValueError("Could not parse response from Gemini API.") from e

  def generate(self, user_prompt: str):
    """Generates a response from the Gemini API.

    This method is an alias for `__call__` to support agents expecting a
    `generate` method.

    Args:
      user_prompt (str): The user's prompt.

    Returns:
        str: The generated text response from the Gemini API.
    """
    return self(user_prompt)
