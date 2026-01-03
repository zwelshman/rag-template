"""
Cost Tracking Utilities
Track and estimate costs for LLM API usage.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UsageRecord:
    """Record of API usage."""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float


class CostTracker:
    """
    Track costs across LLM API calls.
    Provides estimates based on current pricing.
    """

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        'anthropic': {
            'claude-sonnet-4-5-20241022': {
                'input': 3.00,   # $3.00 per 1M input tokens
                'output': 15.00  # $15.00 per 1M output tokens
            },
        },
        'openai': {
            'gpt-4o': {
                'input': 2.50,
                'output': 10.00
            },
            'gpt-4o-mini': {
                'input': 0.15,
                'output': 0.60
            },
        },
    }

    def __init__(self):
        """Initialize cost tracker."""
        self.records: List[UsageRecord] = []

    def add_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Add a usage record and calculate cost.

        Args:
            provider: LLM provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Get pricing
        pricing = self.PRICING.get(provider, {}).get(model, {'input': 0, 'output': 0})

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost

        # Record usage
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=total_cost,
        )
        self.records.append(record)

        return total_cost

    def get_total_cost(self) -> float:
        """Get total estimated cost across all records."""
        return sum(r.estimated_cost for r in self.records)

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total tokens used."""
        return {
            'input': sum(r.input_tokens for r in self.records),
            'output': sum(r.output_tokens for r in self.records),
            'total': sum(r.input_tokens + r.output_tokens for r in self.records),
        }

    def get_summary(self) -> Dict:
        """Get a summary of usage and costs."""
        tokens = self.get_total_tokens()
        return {
            'total_requests': len(self.records),
            'total_cost': self.get_total_cost(),
            'total_input_tokens': tokens['input'],
            'total_output_tokens': tokens['output'],
            'total_tokens': tokens['total'],
        }

    def clear(self):
        """Clear all usage records."""
        self.records = []
