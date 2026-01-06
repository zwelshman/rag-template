"""
Hugging Face Provider
Implementation for Hugging Face Inference API with open source models.
Optimized for latency with fast, lightweight models.
"""

from typing import Optional, Generator
import os
from .base import BaseLLMClient


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face Inference API client for open source LLM inference."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ):
        """
        Initialize Hugging Face client.

        Args:
            api_key: Hugging Face API token (defaults to HF_API_KEY or HUGGINGFACE_API_KEY env var)
            model: Model to use (default: Mistral-7B-Instruct for good latency/quality balance)
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

        self.api_key = api_key or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")

        self.model = model
        self._client = InferenceClient(token=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from Hugging Face."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.chat_completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from Hugging Face."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat_completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
