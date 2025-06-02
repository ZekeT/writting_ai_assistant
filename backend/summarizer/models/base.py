# models/base.py
"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for language model providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass
