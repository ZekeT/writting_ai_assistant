import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Import both OpenAI and Gemini models
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


class LangGraphMemoryManager:
    """
    Manages LangGraph memory for investment writing assistant drafts.
    Handles serialization, deserialization, and state management.
    Supports multiple LLM providers (OpenAI and Google Gemini).
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the LangGraph memory manager.

        Args:
            provider: LLM provider to use ('openai' or 'gemini')
            api_key: Optional API key for the LLM service
        """
        self.provider = provider.lower()
        self.api_key = api_key

    def create_new_memory(self, publication_type: str, draft_date: str) -> Dict[str, Any]:
        """
        Create a new memory context for a draft.

        Args:
            publication_type: Type of publication (e.g., "Daily Wealth Wire")
            draft_date: Date of the draft

        Returns:
            Dict containing serialized memory state
        """
        # Create initial memory state
        memory_state = {
            "messages": [
                self._serialize_message(SystemMessage(content=f"You are an expert investment writer for {publication_type}. "
                                                      f"You provide insightful analysis and clear recommendations. "
                                                      f"Today's date is {draft_date}.")),
                self._serialize_message(HumanMessage(
                    content=f"Please help me draft content for {publication_type} on {draft_date}.")),
                self._serialize_message(AIMessage(content="I'll help you create a comprehensive investment draft. "
                                                  "What specific topics or market sectors would you like to focus on?"))
            ],
            "metadata": {
                "publication_type": publication_type,
                "draft_date": draft_date,
                "creation_timestamp": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "llm_provider": self.provider,
                "market_data": {},
                "topics": [],
                "recommendations": []
            },
            "graph_state": {}
        }

        return memory_state

    def serialize_memory(self, memory_state: Dict[str, Any]) -> str:
        """
        Serialize memory state to JSON string for storage.

        Args:
            memory_state: Memory state dictionary

        Returns:
            JSON string representation
        """
        # Update last_updated timestamp
        if "metadata" in memory_state:
            memory_state["metadata"]["last_updated"] = datetime.now().isoformat()

        return json.dumps(memory_state)

    def deserialize_memory(self, memory_json: Optional[str]) -> Dict[str, Any]:
        """
        Deserialize memory state from JSON string.

        Args:
            memory_json: JSON string representation of memory state

        Returns:
            Memory state dictionary or empty state if None
        """
        if not memory_json:
            return {"messages": [], "metadata": {}, "graph_state": {}}

        try:
            return json.loads(memory_json)
        except json.JSONDecodeError:
            # Return empty state if JSON is invalid
            return {"messages": [], "metadata": {}, "graph_state": {}}

    def _serialize_message(self, message) -> Dict[str, Any]:
        """
        Serialize a LangChain message to dictionary.

        Args:
            message: LangChain message object

        Returns:
            Dictionary representation of message
        """
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "timestamp": datetime.now().isoformat()
        }

    def _deserialize_message(self, message_dict: Dict[str, Any]):
        """
        Deserialize a dictionary to a LangChain message.

        Args:
            message_dict: Dictionary representation of message

        Returns:
            LangChain message object
        """
        message_type = message_dict["type"]
        content = message_dict["content"]
        additional_kwargs = message_dict.get("additional_kwargs", {})

        if message_type == "HumanMessage":
            return HumanMessage(content=content, additional_kwargs=additional_kwargs)
        elif message_type == "AIMessage":
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif message_type == "SystemMessage":
            return SystemMessage(content=content, additional_kwargs=additional_kwargs)
        else:
            # Default to HumanMessage if type is unknown
            return HumanMessage(content=content, additional_kwargs=additional_kwargs)

    def get_messages(self, memory_state: Dict[str, Any]) -> List[Any]:
        """
        Get LangChain message objects from memory state.

        Args:
            memory_state: Memory state dictionary

        Returns:
            List of LangChain message objects
        """
        messages = []
        for msg_dict in memory_state.get("messages", []):
            messages.append(self._deserialize_message(msg_dict))
        return messages

    def add_message(self, memory_state: Dict[str, Any], message) -> Dict[str, Any]:
        """
        Add a message to memory state.

        Args:
            memory_state: Memory state dictionary
            message: LangChain message object

        Returns:
            Updated memory state
        """
        memory_state["messages"].append(self._serialize_message(message))
        memory_state["metadata"]["last_updated"] = datetime.now().isoformat()
        return memory_state

    def update_metadata(self, memory_state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """
        Update metadata in memory state.

        Args:
            memory_state: Memory state dictionary
            key: Metadata key
            value: Metadata value

        Returns:
            Updated memory state
        """
        memory_state["metadata"][key] = value
        memory_state["metadata"]["last_updated"] = datetime.now().isoformat()
        return memory_state

    def _get_llm_model(self):
        """
        Get the appropriate LLM model based on provider setting.

        Returns:
            LangChain chat model
        """
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
                google_api_key=self.api_key
            )
        else:  # Default to OpenAI
            return ChatOpenAI(
                temperature=0.7,
                api_key=self.api_key
            )

    def create_investment_graph(self):
        """
        Create a LangGraph for investment analysis.
        This is a simplified example for demonstration purposes.

        Returns:
            LangGraph StateGraph
        """
        # Define tools
        @tool
        def get_market_data(ticker: str) -> str:
            """Get market data for a specific ticker symbol."""
            # This would normally fetch real data
            sample_data = {
                "AAPL": {"price": 185.92, "change": 0.75, "volume": 32500000},
                "MSFT": {"price": 420.45, "change": 1.2, "volume": 28700000},
                "GOOGL": {"price": 175.33, "change": -0.5, "volume": 18900000},
                "AMZN": {"price": 182.75, "change": 0.3, "volume": 22100000},
                "TSLA": {"price": 215.65, "change": -1.8, "volume": 35600000}
            }

            if ticker in sample_data:
                data = sample_data[ticker]
                return f"Current price: ${data['price']}, Change: {data['change']}%, Volume: {data['volume']}"
            else:
                return f"No data available for {ticker}"

        @tool
        def get_sector_performance() -> str:
            """Get performance data for market sectors."""
            # This would normally fetch real data
            return """
            Sector Performance (Last 7 Days):
            - Technology: +2.3%
            - Healthcare: +1.1%
            - Financials: -0.5%
            - Energy: -1.8%
            - Consumer Discretionary: +0.7%
            - Utilities: +0.2%
            """

        @tool
        def get_economic_indicators() -> str:
            """Get current economic indicators."""
            # This would normally fetch real data
            return """
            Economic Indicators:
            - Inflation Rate: 2.4%
            - Unemployment: 3.7%
            - GDP Growth: 2.1%
            - Fed Funds Rate: 5.25-5.50%
            - 10-Year Treasury Yield: 4.32%
            """

        # Define tools list
        tools = [get_market_data, get_sector_performance,
                 get_economic_indicators]

        # Define state
        class GraphState(Dict):
            """State for the investment analysis graph."""
            messages: List
            tools_output: Optional[str] = None

        # Define nodes
        def call_model(state):
            """Call the LLM to generate a response."""
            # Get the appropriate model based on provider
            model = self._get_llm_model()
            messages = state["messages"]
            response = model.invoke(messages)
            return {"messages": messages + [response]}

        def call_tools(state):
            """Execute tools based on the AI's response."""
            messages = state["messages"]
            last_message = messages[-1]

            # Check if the message has tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tools_output = []
                for tool_call in last_message.tool_calls:
                    # Find the matching tool
                    for tool_func in tools:
                        if tool_func.name == tool_call.name:
                            # Execute the tool with arguments
                            try:
                                tool_args = tool_call.args
                                if isinstance(tool_args, str):
                                    # Try to parse as JSON if it's a string
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except json.JSONDecodeError:
                                        pass

                                # Call the tool with arguments
                                if isinstance(tool_args, dict):
                                    tool_output = tool_func(**tool_args)
                                else:
                                    tool_output = tool_func(tool_args)

                                tools_output.append(
                                    f"{tool_call.name}: {tool_output}")
                            except Exception as e:
                                tools_output.append(
                                    f"{tool_call.name}: Error: {str(e)}")
                            break

                return {"tools_output": "\n".join(tools_output)}

            return {"tools_output": None}

        def should_continue(state):
            """Determine if we should continue the conversation."""
            if state.get("tools_output"):
                return "continue"
            return END

        def add_tool_output_to_messages(state):
            """Add tool output as a message."""
            tools_output = state["tools_output"]
            if tools_output:
                message = HumanMessage(content=f"Tool output:\n{tools_output}")
                return {"messages": state["messages"] + [message]}
            return {}

        # Build the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("call_model", call_model)
        workflow.add_node("call_tools", call_tools)
        workflow.add_node("add_tool_output", add_tool_output_to_messages)

        # Add edges
        workflow.add_edge("call_model", "call_tools")
        workflow.add_conditional_edges(
            "call_tools",
            should_continue,
            {
                "continue": "add_tool_output",
                END: END
            }
        )
        workflow.add_edge("add_tool_output", "call_model")

        # Set entry point
        workflow.set_entry_point("call_model")

        return workflow.compile()

    def run_graph_with_memory(self, memory_state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        Run the LangGraph with the current memory state.

        Args:
            memory_state: Memory state dictionary
            user_input: User's input message

        Returns:
            Updated memory state
        """
        # Add user input to messages
        memory_state = self.add_message(
            memory_state, HumanMessage(content=user_input))

        # Get messages for the graph
        messages = self.get_messages(memory_state)

        # Create and run the graph
        graph = self.create_investment_graph()

        # Initialize graph state
        graph_state = {"messages": messages}

        # Run the graph
        try:
            result = graph.invoke(graph_state)

            # Update memory with new messages
            for message in result["messages"]:
                if message not in messages:  # Only add new messages
                    memory_state = self.add_message(memory_state, message)

            # Store graph state
            memory_state["graph_state"] = {
                "last_result": {
                    "tools_output": result.get("tools_output")
                }
            }

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error running graph: {str(e)}"
            memory_state = self.add_message(
                memory_state, AIMessage(content=error_message))

        return memory_state

    def generate_draft_content(self, memory_state: Dict[str, Any], publication_type: str, draft_date: str) -> tuple:
        """
        Generate draft content based on memory state.

        Args:
            memory_state: Memory state dictionary
            publication_type: Type of publication
            draft_date: Date of the draft

        Returns:
            Tuple of (draft_content, updated_memory_state)
        """
        # Create a system message for draft generation
        system_message = SystemMessage(content=f"""
        You are an expert investment writer for {publication_type}.
        Generate a complete draft for {draft_date} based on our conversation history.
        Include market analysis, sector insights, and specific investment recommendations.
        Format the content with Markdown headings and bullet points for readability.
        """)

        # Add system message to memory
        memory_state = self.add_message(memory_state, system_message)

        # Add a specific request for the draft
        draft_request = HumanMessage(
            content=f"Please generate the complete {publication_type} draft for {draft_date}.")
        memory_state = self.add_message(memory_state, draft_request)

        # Use the appropriate model based on provider
        model = self._get_llm_model()
        messages = self.get_messages(memory_state)

        try:
            # Generate the draft
            response = model.invoke(messages)

            # Add the response to memory
            memory_state = self.add_message(memory_state, response)

            # Extract the draft content
            draft_content = response.content

            # Update metadata
            memory_state = self.update_metadata(
                memory_state, "draft_generated", True)
            memory_state = self.update_metadata(
                memory_state, "draft_timestamp", datetime.now().isoformat())

            return draft_content, memory_state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error generating draft: {str(e)}"
            fallback_content = f"""
            # {publication_type} - {draft_date}
            
            ## Market Overview
            
            This is a placeholder draft. The actual draft generation encountered an error.
            
            Error details: {error_message}
            
            ## Investment Recommendations
            
            Please try again or contact support if the issue persists.
            """

            # Add error message to memory
            memory_state = self.add_message(
                memory_state, AIMessage(content=error_message))

            return fallback_content, memory_state
