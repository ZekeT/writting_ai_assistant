import os
import json
from typing import Optional, Dict, Any, List, Tuple, Literal
from datetime import datetime
from enum import Enum
from langchain_core.messages import HumanMessage, AIMessage

from backend.langgraph_memory import LangGraphMemoryManager


class LLMProvider(str, Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMInterface:
    """
    Interface for LLM operations with LangGraph memory integration.
    Handles draft generation, chat interactions, and memory management.
    Supports multiple LLM providers (OpenAI and Google Gemini).
    """

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM interface.

        Args:
            provider: LLM provider to use ('openai' or 'gemini')
            api_key: Optional API key for the LLM service
        """
        # Determine provider from input or environment
        self.provider = self._get_provider(provider)

        # Set up API keys based on provider
        self.api_keys = self._setup_api_keys(api_key)

        # Initialize LangGraph memory manager
        self.memory_manager = LangGraphMemoryManager(
            provider=self.provider,
            api_key=self.api_keys.get(self.provider)
        )

    def _get_provider(self, provider: Optional[str]) -> str:
        """
        Determine which LLM provider to use.

        Args:
            provider: Optional provider name

        Returns:
            Provider name as string
        """
        if provider:
            # Use specified provider if valid
            if provider.lower() in [LLMProvider.OPENAI, LLMProvider.GEMINI]:
                return provider.lower()

        # Check environment variables for provider preference
        env_provider = os.environ.get("LLM_PROVIDER", "").lower()
        if env_provider in [LLMProvider.OPENAI, LLMProvider.GEMINI]:
            return env_provider

        # Check which API keys are available
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return LLMProvider.GEMINI

        # Default to OpenAI
        return LLMProvider.OPENAI

    def _setup_api_keys(self, api_key: Optional[str]) -> Dict[str, str]:
        """
        Set up API keys for all providers.

        Args:
            api_key: Optional API key passed directly

        Returns:
            Dictionary of API keys by provider
        """
        api_keys = {}

        # OpenAI API key
        openai_key = api_key if self.provider == LLMProvider.OPENAI else None
        api_keys[LLMProvider.OPENAI] = openai_key or os.environ.get(
            "OPENAI_API_KEY", "")

        # Google Gemini API key
        gemini_key = api_key if self.provider == LLMProvider.GEMINI else None
        api_keys[LLMProvider.GEMINI] = (
            gemini_key or
            os.environ.get("GOOGLE_API_KEY", "") or
            os.environ.get("GEMINI_API_KEY", "")
        )

        return api_keys

    def generate_draft(self, publication_type: str, draft_date: str, user_prompt: Optional[str] = None):
        """
        Generate a draft with LangGraph memory.

        Args:
            publication_type: Type of publication
            draft_date: Date of the draft
            user_prompt: Optional user prompt to guide generation

        Returns:
            Tuple of (draft_content, serialized_memory)
        """
        # Create new memory context
        memory_state = self.memory_manager.create_new_memory(
            publication_type, draft_date)

        # If user provided a prompt, add it to memory
        if user_prompt:
            memory_state = self.memory_manager.add_message(
                memory_state,
                HumanMessage(content=user_prompt)
            )

        # Generate draft content
        draft_content, updated_memory = self.memory_manager.generate_draft_content(
            memory_state,
            publication_type,
            draft_date
        )

        # Add provider info to metadata
        updated_memory = self.memory_manager.update_metadata(
            updated_memory,
            "llm_provider",
            self.provider
        )

        # Serialize memory for storage
        serialized_memory = self.memory_manager.serialize_memory(
            updated_memory)

        return draft_content, serialized_memory

    def process_chat(self, user_input: str, memory_json: Optional[str] = None):
        """
        Process a chat message using LangGraph memory.

        Args:
            user_input: User's chat message
            memory_json: Optional serialized memory state

        Returns:
            Tuple of (ai_response, serialized_memory)
        """
        # Deserialize memory or create new if none exists
        memory_state = self.memory_manager.deserialize_memory(memory_json)

        # Check if we need to switch providers based on memory metadata
        if memory_state.get("metadata", {}).get("llm_provider"):
            stored_provider = memory_state["metadata"]["llm_provider"]
            if stored_provider != self.provider and stored_provider in [LLMProvider.OPENAI, LLMProvider.GEMINI]:
                # If memory was created with a different provider, use that one for consistency
                self.provider = stored_provider
                self.memory_manager.provider = stored_provider
                self.memory_manager.api_key = self.api_keys.get(
                    stored_provider)

        # Run the graph with user input
        updated_memory = self.memory_manager.run_graph_with_memory(
            memory_state, user_input)

        # Get the last AI message as the response
        messages = self.memory_manager.get_messages(updated_memory)
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

        if ai_messages:
            ai_response = ai_messages[-1].content
        else:
            ai_response = "I'm sorry, I couldn't generate a response. Please try again."
            # Add fallback response to memory
            updated_memory = self.memory_manager.add_message(
                updated_memory,
                AIMessage(content=ai_response)
            )

        # Serialize updated memory for storage
        serialized_memory = self.memory_manager.serialize_memory(
            updated_memory)

        return ai_response, serialized_memory

    def chat_response(self, user_input: str, memory_context: Optional[Dict[str, Any]] = None):
        """
        Process a chat message and return a response.
        This is a wrapper around process_chat that handles memory serialization.

        Args:
            user_input: User's chat message
            memory_context: Optional memory context (can be serialized or deserialized)

        Returns:
            Tuple of (ai_response, updated_memory_context)
        """
        # Convert memory_context to JSON string if it's a dict
        memory_json = None
        if memory_context:
            if isinstance(memory_context, dict):
                memory_json = json.dumps(memory_context)
            elif isinstance(memory_context, str):
                memory_json = memory_context

        # Process the chat
        response, updated_memory_json = self.process_chat(
            user_input, memory_json)

        # Convert memory JSON back to dict if needed
        updated_memory = json.loads(
            updated_memory_json) if updated_memory_json else {}

        return response, updated_memory

    def get_memory_summary(self, memory_json: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the memory state.

        Args:
            memory_json: Optional serialized memory state

        Returns:
            Dictionary with memory summary
        """
        # Deserialize memory
        memory_state = self.memory_manager.deserialize_memory(memory_json)

        # Extract metadata
        metadata = memory_state.get("metadata", {})

        # Count messages by type
        messages = memory_state.get("messages", [])
        message_counts = {
            "human": sum(1 for msg in messages if msg.get("type") == "HumanMessage"),
            "ai": sum(1 for msg in messages if msg.get("type") == "AIMessage"),
            "system": sum(1 for msg in messages if msg.get("type") == "SystemMessage"),
            "total": len(messages)
        }

        # Create summary
        summary = {
            "publication_type": metadata.get("publication_type", "Unknown"),
            "draft_date": metadata.get("draft_date", "Unknown"),
            "creation_timestamp": metadata.get("creation_timestamp"),
            "last_updated": metadata.get("last_updated"),
            "llm_provider": metadata.get("llm_provider", self.provider),
            "message_counts": message_counts,
            "has_draft": metadata.get("draft_generated", False),
            "topics": metadata.get("topics", []),
            "recommendations": metadata.get("recommendations", [])
        }

        return summary

    def update_memory_metadata(self, memory_json: str, updates: Dict[str, Any]) -> str:
        """
        Update metadata in memory state.

        Args:
            memory_json: Serialized memory state
            updates: Dictionary of metadata updates

        Returns:
            Updated serialized memory
        """
        # Deserialize memory
        memory_state = self.memory_manager.deserialize_memory(memory_json)

        # Apply updates
        for key, value in updates.items():
            memory_state = self.memory_manager.update_metadata(
                memory_state, key, value)

        # Serialize updated memory
        return self.memory_manager.serialize_memory(memory_state)

    def get_available_providers(self) -> Dict[str, bool]:
        """
        Get available LLM providers based on API keys.

        Returns:
            Dictionary of provider availability
        """
        return {
            LLMProvider.OPENAI: bool(self.api_keys.get(LLMProvider.OPENAI)),
            LLMProvider.GEMINI: bool(self.api_keys.get(LLMProvider.GEMINI))
        }

    def set_provider(self, provider: str) -> bool:
        """
        Set the active LLM provider.

        Args:
            provider: Provider name ('openai' or 'gemini')

        Returns:
            Success status
        """
        if provider.lower() not in [LLMProvider.OPENAI, LLMProvider.GEMINI]:
            return False

        if not self.api_keys.get(provider.lower()):
            return False

        self.provider = provider.lower()
        self.memory_manager.provider = self.provider
        self.memory_manager.api_key = self.api_keys.get(self.provider)
        return True
