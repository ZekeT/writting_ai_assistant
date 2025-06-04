"""
Model interfaces for the advanced summarizer module.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    """Abstract base class for LLM model interfaces"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate text from messages"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name"""
        pass


class OpenAIInterface(ModelInterface):
    """Interface for OpenAI models"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI interface.
        
        Args:
            model_name: Name of the model
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            api_key: Optional API key
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Generated text
        """
        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Generate response
        response = self.llm.invoke(lc_messages)
        
        return response.content
    
    def get_name(self) -> str:
        """Get model name"""
        return self.model_name


class GoogleInterface(ModelInterface):
    """Interface for Google Gemini models"""
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Google interface.
        
        Args:
            model_name: Name of the model
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            api_key: Optional API key
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            google_api_key=self.api_key
        )
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Generated text
        """
        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Generate response
        response = self.llm.invoke(lc_messages)
        
        return response.content
    
    def get_name(self) -> str:
        """Get model name"""
        return self.model_name


def create_model_interface(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    api_key: Optional[str] = None
) -> ModelInterface:
    """
    Create a model interface based on provider.
    
    Args:
        provider: Provider name ('openai' or 'google')
        model_name: Optional model name (uses default if None)
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        api_key: Optional API key
        
    Returns:
        Model interface
    """
    if provider.lower() == "openai":
        model = model_name or "gpt-4-turbo"
        return OpenAIInterface(model, temperature, max_tokens, api_key)
    elif provider.lower() == "google":
        model = model_name or "gemini-pro"
        return GoogleInterface(model, temperature, max_tokens, api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
