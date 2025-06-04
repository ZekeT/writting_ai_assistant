"""
Graph nodes and workflow for the advanced summarizer module.
"""

import logging
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

# Configure logging
logger = logging.getLogger(__name__)

class GraphState(dict):
    """State for the summarization graph"""
    def __init__(
        self, 
        articles=None, 
        prompt=None, 
        formatted_articles=None, 
        summary=None,
        intermediate_summaries=None
    ):
        self.articles = articles or []
        self.prompt = prompt or ""
        self.formatted_articles = formatted_articles or ""
        self.summary = summary
        self.intermediate_summaries = intermediate_summaries or []
        super().__init__(
            articles=self.articles,
            prompt=self.prompt,
            formatted_articles=self.formatted_articles,
            summary=self.summary,
            intermediate_summaries=self.intermediate_summaries
        )

def format_articles_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format articles for the prompt.
    
    Args:
        state: Graph state
        
    Returns:
        Updated state with formatted articles
    """
    from ..core.prompt_builder import format_article_for_prompt
    
    articles_text = "# INPUT ARTICLES:\n\n"
    
    for i, article in enumerate(state["articles"]):
        articles_text += format_article_for_prompt(article, i)
    
    return {"formatted_articles": articles_text}

def generate_summary_node(state: Dict[str, Any], model_interface) -> Dict[str, Any]:
    """
    Generate summary using the LLM.
    
    Args:
        state: Graph state
        model_interface: Model interface for generation
        
    Returns:
        Updated state with summary
    """
    # Combine prompt with formatted articles
    full_prompt = state["prompt"] + state["formatted_articles"]
    
    # Create messages for the LLM
    messages = [
        {"role": "system", "content": "You are an expert financial news analyst creating a digest."},
        {"role": "user", "content": full_prompt}
    ]
    
    # Generate summary
    summary = model_interface.generate(messages)
    
    return {"summary": summary}

def hierarchical_summarize_node(state: Dict[str, Any], model_interface) -> Dict[str, Any]:
    """
    Perform hierarchical summarization for large article sets.
    
    Args:
        state: Graph state
        model_interface: Model interface for generation
        
    Returns:
        Updated state with summary
    """
    from ..core.prompt_builder import build_hierarchical_prompt
    
    # First, summarize each article individually
    individual_summaries = []
    
    for i, article in enumerate(state["articles"]):
        # Create individual article prompt
        individual_prompt = build_hierarchical_prompt("individual")
        
        # Create messages for the LLM
        messages = [
            {"role": "system", "content": "You are an expert financial news analyst."},
            {"role": "user", "content": f"{individual_prompt}\n\n{article['content']}"}
        ]
        
        # Generate individual summary
        summary = model_interface.generate(messages)
        individual_summaries.append(summary)
        
        logger.info(f"Generated individual summary for article {i+1}/{len(state['articles'])}")
    
    # Then, combine the individual summaries
    combined_prompt = build_hierarchical_prompt("combined")
    combined_content = "\n\n---\n\n".join(individual_summaries)
    
    # Create messages for the LLM
    messages = [
        {"role": "system", "content": "You are an expert financial news analyst."},
        {"role": "user", "content": f"{combined_prompt}\n\n{combined_content}"}
    ]
    
    # Generate combined summary
    final_summary = model_interface.generate(messages)
    
    return {
        "summary": final_summary,
        "intermediate_summaries": individual_summaries
    }

def build_summarization_graph(model_interface, use_hierarchical: bool = False) -> StateGraph:
    """
    Build the LangGraph for summarization.
    
    Args:
        model_interface: Model interface for generation
        use_hierarchical: Whether to use hierarchical summarization
        
    Returns:
        Compiled graph
    """
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("format_articles", lambda state: format_articles_node(state))
    
    if use_hierarchical:
        workflow.add_node(
            "generate_summary", 
            lambda state: hierarchical_summarize_node(state, model_interface)
        )
    else:
        workflow.add_node(
            "generate_summary", 
            lambda state: generate_summary_node(state, model_interface)
        )
    
    # Add edges
    workflow.add_edge("format_articles", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    # Set entry point
    workflow.set_entry_point("format_articles")
    
    # Compile the graph
    return workflow.compile()
