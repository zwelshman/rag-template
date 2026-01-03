"""
Anthropic Claude Provider
Implementation for Anthropic's Claude models.
"""

from typing import Optional, Generator
import os
import logging
from .base import BaseLLMClient

logger = logging.getLogger("rag_app.llm.anthropic")


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
