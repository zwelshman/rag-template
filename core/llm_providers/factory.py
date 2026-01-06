"""
LLM Client Factory
Factory class for creating appropriate LLM clients.
"""

from typing import Optional, List, Generator
import logging
from .base import BaseLLMClient
from .openai_provider import OpenAIClient
from .anthropic_provider import AnthropicClient
from .huggingface_provider import HuggingFaceClient

logger = logging.getLogger("rag_app.llm.factory")


class LLMClient:
    """
    Unified LLM client using Hugging Face with open source models (optimized for latency).
    Factory class for creating appropriate LLM client based on provider.
    """

    PROVIDERS = {
        "huggingface": HuggingFaceClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }

    AVAILABLE_MODELS = {
        "huggingface": [
            "mistralai/Mistral-7B-Instruct-v0.3",  # Fast, good quality
            "meta-llama/Llama-3.2-3B-Instruct",  # Smallest, fastest
            "microsoft/Phi-3-mini-4k-instruct",  # Compact, efficient
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-sonnet-4-5-20241022",
        ],
    }

    def __init__(
        self,
        provider: str = "huggingface",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize unified LLM client.

        Args:
            provider: LLM provider ('huggingface', 'openai', or 'anthropic')
            api_key: API key for the provider
            model: Model name (uses provider default if not specified)
            **kwargs: Additional arguments passed to the provider client
        """
        logger.info(f"Creating LLMClient with provider: {provider}")
        provider = provider.lower()

        if provider not in self.PROVIDERS:
            logger.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.PROVIDERS.keys())}")

        client_class = self.PROVIDERS[provider]

        if model is None:
            model = self.AVAILABLE_MODELS[provider][0]
            logger.info(f"Using default model: {model}")

        self._client = client_class(api_key=api_key, model=model, **kwargs)
        self.provider = provider
        self.model = model
        logger.info(f"LLMClient ready: provider={provider}, model={model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from the LLM."""
        return self._client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        return self._client.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """Get list of available models for a provider."""
        return cls.AVAILABLE_MODELS.get(provider.lower(), [])
