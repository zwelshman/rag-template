"""
LLM Client Module
Provides a unified interface for interacting with various LLM providers.
Uses Anthropic Claude Sonnet 4.5 as the primary model.
"""

from typing import List, Dict, Any, Optional, Generator
from abc import ABC, abstractmethod
import os
import logging

# Configure logger for this module
logger = logging.getLogger("rag_app.llm")
logger.setLevel(logging.INFO)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini)
            base_url: Optional custom base URL for API
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model
        self._client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
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
        """Generate a streaming response from OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client using Sonnet 4.5."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20241022",
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-5-20241022)
        """
        logger.info("Initializing Anthropic client...")
        try:
            from anthropic import Anthropic
        except ImportError:
            logger.error("anthropic package not installed")
            raise ImportError("anthropic is required. Install with: pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("No Anthropic API key provided")
            raise ValueError("Anthropic API key is required")

        self.model = model
        self._client = Anthropic(api_key=self.api_key)
        logger.info(f"Anthropic client initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from Anthropic Claude Sonnet 4.5."""
        logger.info(f"Generating response (non-streaming)")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max tokens: {max_tokens}")
        logger.info(f"  Prompt length: {len(prompt)} chars")

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt
            logger.debug(f"  System prompt length: {len(system_prompt)} chars")

        # Temperature must be between 0 and 1 for Anthropic
        kwargs["temperature"] = min(max(temperature, 0), 1)

        logger.info("Sending request to Anthropic API...")
        response = self._client.messages.create(**kwargs)

        result = response.content[0].text
        logger.info(f"Response received: {len(result)} chars")
        logger.info(f"  Usage - Input tokens: {response.usage.input_tokens}")
        logger.info(f"  Usage - Output tokens: {response.usage.output_tokens}")

        return result

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from Anthropic Claude Sonnet 4.5."""
        logger.info(f"Generating response (streaming)")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max tokens: {max_tokens}")
        logger.info(f"  Prompt length: {len(prompt)} chars")

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt
            logger.debug(f"  System prompt length: {len(system_prompt)} chars")

        kwargs["temperature"] = min(max(temperature, 0), 1)

        logger.info("Starting streaming request to Anthropic API...")
        token_count = 0
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                token_count += 1
                yield text
        logger.info(f"Streaming complete: ~{token_count} tokens streamed")


class LLMClient:
    """
    Unified LLM client using Anthropic Claude Sonnet 4.5.
    Factory class for creating appropriate LLM client based on provider.
    """

    PROVIDERS = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }

    # Sonnet 4.5 is the only supported Anthropic model
    AVAILABLE_MODELS = {
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
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize unified LLM client.

        Args:
            provider: LLM provider ('openai' or 'anthropic')
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

        # Set default model if not provided
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
