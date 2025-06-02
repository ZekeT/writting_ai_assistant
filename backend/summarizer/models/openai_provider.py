# models/openai_provider.py
"""OpenAI-specific implementation."""

import os
from typing import Dict, Any, Optional
import logging
from openai import OpenAI

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 2000):
        """Initialize OpenAI provider."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at creating structured market summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )

            content = response.choices[0].message.content
            return content.strip() if content is not None else ""

        except Exception as e:
            self.logger.error(f"Error generating with OpenAI: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': 'OpenAI',
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
