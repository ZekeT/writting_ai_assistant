# LangGraph Integration for Writing Assistant with Multi-Provider Support

This document provides a comprehensive guide to the LangGraph integration in the Writing Assistant application, with support for multiple LLM providers (OpenAI and Google Gemini). It explains how LangGraph memory is used to enhance draft creation and AI chat functionality, and how you can extend or modify the implementation for your specific needs.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [LLM Provider Support](#llm-provider-support)
4. [LangGraph Memory Structure](#langgraph-memory-structure)
5. [Key Components](#key-components)
6. [How It Works](#how-it-works)
7. [Setup and Installation](#setup-and-installation)
8. [Usage Examples](#usage-examples)
9. [Extending the Implementation](#extending-the-implementation)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

## Overview

The Writing Assistant app uses LangGraph to maintain memory context for each draft, enabling more coherent and contextually aware interactions. This integration allows the AI assistant to:

- Remember previous conversations about a specific draft
- Access tools and retrieve information during chat
- Generate draft content based on accumulated context
- Maintain a persistent memory state across user sessions
- Switch between different LLM providers (OpenAI and Google Gemini)

LangGraph is a library for building stateful, multi-actor applications with LLMs. It provides a framework for creating directed graphs where nodes can be LLM calls, tool executions, or custom functions, with edges defining the flow between these components.

## Architecture

The integration follows a layered architecture:

1. **Storage Layer**: SQLite database with a `memory_context` field in the drafts table
2. **Memory Management Layer**: `LangGraphMemoryManager` class for serialization/deserialization
3. **LLM Interface Layer**: `LLMInterface` class for draft generation and chat processing
4. **Application Layer**: Frontend components and draft operations

This separation of concerns allows for easy maintenance and extension of the functionality.

## LLM Provider Support

The application supports two LLM providers:

### OpenAI

- Default provider if no other is specified
- Requires an OpenAI API key
- Set via `OPENAI_API_KEY` environment variable

### Google Gemini

- Alternative provider for draft generation and chat
- Requires a Google API key
- Set via `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable

### Provider Selection

The provider can be selected in several ways:

1. **Environment Variable**: Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini`
2. **API Key Availability**: If only one provider's API key is available, it will be used automatically
3. **Explicit Selection**: Pass the provider name to the LLMInterface constructor
4. **Per-Draft Consistency**: Each draft remembers which provider was used to create it and will use the same provider for chat interactions

## LangGraph Memory Structure

Each draft's memory is stored as a JSON structure with the following components:

```json
{
  "messages": [
    {
      "type": "SystemMessage",
      "content": "You are an expert investment writer...",
      "additional_kwargs": {},
      "timestamp": "2025-05-26T04:20:00.000Z"
    },
    {
      "type": "HumanMessage",
      "content": "Please help me draft content...",
      "additional_kwargs": {},
      "timestamp": "2025-05-26T04:20:10.000Z"
    },
    {
      "type": "AIMessage",
      "content": "I'll help you create a comprehensive investment draft...",
      "additional_kwargs": {},
      "timestamp": "2025-05-26T04:20:20.000Z"
    }
  ],
  "metadata": {
    "publication_type": "Daily Wealth Wire",
    "draft_date": "2025-05-26",
    "creation_timestamp": "2025-05-26T04:20:00.000Z",
    "last_updated": "2025-05-26T04:20:30.000Z",
    "llm_provider": "openai",
    "market_data": {},
    "topics": ["Technology", "Healthcare"],
    "recommendations": ["AAPL", "MSFT", "JNJ"]
  },
  "graph_state": {
    "last_result": {
      "tools_output": "get_market_data: Current price: $185.92, Change: 0.75%, Volume: 32500000"
    }
  }
}
```

- **messages**: Chronological history of the conversation, including system, human, and AI messages
- **metadata**: Draft-specific information and tracked entities, including the LLM provider used
- **graph_state**: State information from the LangGraph execution

## Key Components

### 1. LangGraphMemoryManager

Located in `backend/langgraph_memory.py`, this class handles:

- Creating new memory contexts
- Serializing/deserializing memory state
- Managing message history
- Updating metadata
- Running the LangGraph for chat interactions
- Generating draft content
- Selecting the appropriate LLM model based on provider

### 2. LLMInterface

Located in `backend/llm_interface.py`, this class provides:

- A simplified interface for draft generation
- Chat processing with memory context
- Memory summary generation
- Metadata updates
- Provider selection and API key management
- Provider consistency across interactions

### 3. Draft Operations

Located in `frontend/draft_operations.py`, these functions:

- Create new drafts with LangGraph memory
- Process chat interactions using the memory context
- Update the database with new memory states
- Provide memory summaries
- Support provider selection for draft creation and chat

## How It Works

### Provider Selection Flow

1. When initializing `LLMInterface`, the provider is determined in this order:
   - Explicitly provided provider parameter
   - `LLM_PROVIDER` environment variable
   - Available API keys (if only one provider has a key)
   - Default to OpenAI

2. For chat interactions with existing drafts:
   - The provider stored in the draft's memory context is used
   - This ensures consistency in the conversation style and capabilities

### Draft Creation Flow

1. User clicks "Add New Draft" and enters a title
2. `create_new_draft()` is called in `draft_operations.py`
3. `LLMInterface` is initialized with the selected provider
4. `generate_draft()` creates a new memory context and generates content
5. The draft and its memory context (including provider info) are saved to the database

### Chat Interaction Flow

1. User selects a draft and enters a message in the chat
2. `process_ai_chat()` is called in `draft_operations.py`
3. The draft's memory context is retrieved from the database
4. `LLMInterface.process_chat()` deserializes the memory and processes the message
5. The provider from the memory context is used for consistency
6. The LangGraph is run with the user's input
7. The response and updated memory are returned
8. The draft's memory context is updated in the database

## Setup and Installation

### Prerequisites

- Python 3.8+
- Streamlit
- LangChain and LangGraph libraries
- SQLite
- OpenAI API key and/or Google API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/writing_assistant_app.git
   cd writing_assistant_app
   ```

2. Install dependencies:
   ```bash
   pip install streamlit langchain langchain-core langchain-community langgraph google-generativeai langchain-google-genai
   ```

3. Set up the database:
   ```bash
   python setup_database.py --sample-data
   ```

4. Set up API keys:
   ```bash
   # For OpenAI
   export OPENAI_API_KEY=your_openai_api_key_here
   
   # For Google Gemini
   export GOOGLE_API_KEY=your_google_api_key_here
   # OR
   export GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional: Set default provider
   export LLM_PROVIDER=openai  # or gemini
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage Examples

### Creating a New Draft with OpenAI

```python
from backend.llm_interface import LLMInterface, LLMProvider

# Initialize LLM interface with OpenAI
llm = LLMInterface(provider=LLMProvider.OPENAI)

# Generate draft content with memory
content, memory_json = llm.generate_draft(
    "Daily Wealth Wire",
    "2025-05-26",
    "Create a draft about technology stocks"
)

# Save to database
draft_id = db.create_draft(
    "Tech Stock Analysis", 
    publication_id, 
    "2025-05-26", 
    content=content,
    memory_context=memory_json
)
```

### Creating a New Draft with Google Gemini

```python
from backend.llm_interface import LLMInterface, LLMProvider

# Initialize LLM interface with Gemini
llm = LLMInterface(provider=LLMProvider.GEMINI)

# Generate draft content with memory
content, memory_json = llm.generate_draft(
    "Weekly Investment Ideas",
    "2025-05-26",
    "Create a draft about renewable energy investments"
)

# Save to database
draft_id = db.create_draft(
    "Renewable Energy Focus", 
    publication_id, 
    "2025-05-26", 
    content=content,
    memory_context=memory_json
)
```

### Processing Chat with Memory

```python
from backend.llm_interface import LLMInterface

# Initialize LLM interface (provider will be determined from memory)
llm = LLMInterface()

# Get memory context from database
draft = db.get_draft_by_id(draft_id)
memory_context = draft.get('memory_context')

# Process chat with memory
response, updated_memory = llm.process_chat(
    "What are the top tech stocks to watch?",
    memory_context
)

# Update memory in database
db.update_draft(draft_id, memory_context=updated_memory)
```

### Checking Available Providers

```python
from backend.llm_interface import LLMInterface

# Initialize LLM interface
llm = LLMInterface()

# Check which providers are available
available_providers = llm.get_available_providers()
print(available_providers)
# Output: {'openai': True, 'gemini': True}

# Switch provider if available
if available_providers['gemini']:
    llm.set_provider('gemini')
```

## Extending the Implementation

### Adding a New Provider

To add support for a new LLM provider:

1. Update the `LLMProvider` enum in `llm_interface.py`:

```python
class LLMProvider(str, Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"  # New provider
```

2. Update the `_setup_api_keys` method to handle the new provider's API key:

```python
def _setup_api_keys(self, api_key: Optional[str]) -> Dict[str, str]:
    # Existing code...
    
    # Anthropic API key
    anthropic_key = api_key if self.provider == LLMProvider.ANTHROPIC else None
    api_keys[LLMProvider.ANTHROPIC] = (
        anthropic_key or 
        os.environ.get("ANTHROPIC_API_KEY", "")
    )
    
    return api_keys
```

3. Update the `_get_llm_model` method in `LangGraphMemoryManager`:

```python
def _get_llm_model(self):
    if self.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=self.api_key
        )
    elif self.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0.7,
            anthropic_api_key=self.api_key
        )
    else:  # Default to OpenAI
        return ChatOpenAI(
            temperature=0.7,
            api_key=self.api_key
        )
```

### Adding New Tools

To add new tools to the LangGraph:

1. Define the tool in `LangGraphMemoryManager.create_investment_graph()`:

```python
@tool
def get_stock_sentiment(ticker: str) -> str:
    """Get sentiment analysis for a specific stock."""
    # Implementation here
    return f"Sentiment for {ticker}: Positive (0.75)"
```

2. Add the tool to the tools list:

```python
tools = [get_market_data, get_sector_performance, get_economic_indicators, get_stock_sentiment]
```

### Customizing the Graph

To modify the graph structure:

1. Update the `create_investment_graph()` method in `LangGraphMemoryManager`
2. Add new nodes, edges, or conditional logic as needed

Example of adding a new node:

```python
def analyze_sentiment(state):
    """Analyze sentiment in the conversation."""
    # Implementation here
    return {"sentiment_score": 0.75}

# Add the node
workflow.add_node("analyze_sentiment", analyze_sentiment)

# Add edges
workflow.add_edge("call_model", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "call_tools")
```

## Troubleshooting

### Common Issues

#### API Key Issues

**Symptom**: Error messages about invalid API keys or authentication failures.

**Solution**: 
- Verify that the correct API keys are set in environment variables
- Check that the keys have the necessary permissions
- For Gemini, ensure you're using the correct key format (API keys start with "AI")

#### Provider Selection Issues

**Symptom**: Wrong provider being used or provider switching unexpectedly.

**Solution**:
- Check the memory context to see which provider is stored
- Verify environment variables are set correctly
- Use explicit provider selection when creating drafts

#### Memory Not Persisting

**Symptom**: Chat history or context is lost between sessions.

**Solution**: 
- Check that memory serialization is working correctly
- Verify that the database is updating the `memory_context` field
- Ensure the JSON structure is valid

#### LangGraph Errors

**Symptom**: Errors when running the graph or tool execution failures.

**Solution**:
- Check tool implementations for errors
- Verify that the graph structure is valid
- Check for missing dependencies

### Debugging Tips

1. Enable verbose logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Inspect memory state:
   ```python
   memory_state = memory_manager.deserialize_memory(memory_json)
   print(json.dumps(memory_state, indent=2))
   ```

3. Test providers individually:
   ```python
   # Test OpenAI
   llm = LLMInterface(provider="openai")
   print(llm.provider)  # Should print "openai"
   
   # Test Gemini
   llm = LLMInterface(provider="gemini")
   print(llm.provider)  # Should print "gemini"
   ```

## Advanced Topics

### Provider-Specific Customization

You can customize the behavior of each provider:

```python
def _get_llm_model(self):
    if self.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.5,  # Lower temperature for Gemini
            google_api_key=self.api_key
        )
    else:  # OpenAI
        return ChatOpenAI(
            model="gpt-4",  # Specify model version
            temperature=0.7,
            api_key=self.api_key
        )
```

### Provider Fallback Mechanism

Implement a fallback mechanism when a provider fails:

```python
def process_chat(self, user_input: str, memory_json: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    # Existing code...
    
    try:
        # Try with primary provider
        result = self.memory_manager.run_graph_with_memory(memory_state, user_input)
    except Exception as e:
        # If primary provider fails, try fallback
        original_provider = self.provider
        fallback_provider = "openai" if original_provider == "gemini" else "gemini"
        
        if self.api_keys.get(fallback_provider):
            print(f"Primary provider {original_provider} failed, falling back to {fallback_provider}")
            self.provider = fallback_provider
            self.memory_manager.provider = fallback_provider
            self.memory_manager.api_key = self.api_keys.get(fallback_provider)
            
            # Try with fallback provider
            result = self.memory_manager.run_graph_with_memory(memory_state, user_input)
            
            # Reset to original provider
            self.provider = original_provider
            self.memory_manager.provider = original_provider
            self.memory_manager.api_key = self.api_keys.get(original_provider)
        else:
            # No fallback available, re-raise the exception
            raise
```

### Memory Summarization

For long-running conversations, implement memory summarization:

```python
def summarize_memory(memory_state):
    """Summarize memory to prevent it from growing too large."""
    messages = memory_state.get("messages", [])
    
    if len(messages) > 20:
        # Extract key messages (first 5 and last 10)
        key_messages = messages[:5] + messages[-10:]
        
        # Create a summary message
        summary = {
            "type": "SystemMessage",
            "content": f"Conversation summarized. {len(messages) - 15} messages omitted.",
            "additional_kwargs": {"is_summary": True},
            "timestamp": datetime.now().isoformat()
        }
        
        # Replace messages with summarized version
        memory_state["messages"] = key_messages[:5] + [summary] + key_messages[5:]
    
    return memory_state
```

## Conclusion

This LangGraph integration provides a powerful foundation for building context-aware AI assistants for investment writing, with the flexibility to choose between OpenAI and Google Gemini as LLM providers. By maintaining memory state per draft and supporting multiple providers, the application can provide more coherent and helpful interactions over time while giving users choice in which AI technology to use.

The modular architecture allows for easy extension and customization, whether you need to add new providers, tools, modify the graph structure, or integrate with external systems.

For further learning, explore the [LangGraph documentation](https://python.langchain.com/docs/langgraph), [LangChain guides](https://python.langchain.com/docs/get_started/introduction), and the documentation for [OpenAI](https://platform.openai.com/docs/api-reference) and [Google Gemini](https://ai.google.dev/docs) APIs.
