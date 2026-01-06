"""
Hugging Face Provider
Implementation for Hugging Face Inference API with open source models.
Optimized for latency with fast, lightweight models.
"""

from typing import Optional, Generator, List
import os
import logging
from .base import BaseLLMClient

logger = logging.getLogger("rag_app.llm.huggingface")


class DeprecatedModelError(Exception):
    """Raised when a model is deprecated and no longer available."""
    pass


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face Inference API client for open source LLM inference."""

    FALLBACK_MODELS: List[str] = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ):
        """
        Initialize Hugging Face client.

        Args:
            api_key: Hugging Face API token (defaults to HF_API_KEY or HUGGINGFACE_API_KEY env var)
            model: Model to use (default: Meta-Llama-3.1-8B-Instruct for good latency/quality balance)
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
        self._InferenceClient = InferenceClient

    def _is_deprecated_error(self, error: Exception) -> bool:
        """Check if the error indicates a deprecated model."""
        error_str = str(error).lower()
        return (
            "410" in error_str or
            "gone" in error_str or
            "deprecated" in error_str or
            "no longer supported" in error_str
        )

    def _try_with_fallback(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ):
        """Try the request with fallback models if the primary model is deprecated."""
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Attempting request with model: {model}")
                if stream:
                    return self._client.chat_completion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ), model
                else:
                    response = self._client.chat_completion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if model != self.model:
                        logger.warning(
                            f"Model '{self.model}' is deprecated. Successfully fell back to '{model}'. "
                            f"Consider updating your configuration to use '{model}' as the default."
                        )
                        self.model = model
                    return response, model
            except Exception as e:
                last_error = e
                if self._is_deprecated_error(e):
                    logger.warning(f"Model '{model}' is deprecated or unavailable: {e}")
                    continue
                else:
                    raise

        raise DeprecatedModelError(
            f"All models are unavailable. Last error: {last_error}. "
            f"Tried models: {models_to_try}. "
            f"Please check Hugging Face status or try a different provider."
        )

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

        response, used_model = self._try_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
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

        stream, used_model = self._try_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        if used_model != self.model:
            logger.warning(
                f"Model '{self.model}' is deprecated. Using fallback model '{used_model}'. "
                f"Consider updating your configuration."
            )
            self.model = used_model

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
