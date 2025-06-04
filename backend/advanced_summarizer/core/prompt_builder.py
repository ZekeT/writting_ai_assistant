"""
Prompt building functionality for the advanced summarizer module.
"""

import re
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def extract_article_metadata(article_text: str) -> Dict[str, str]:
    """Extract metadata from article text"""
    # Extract title (first heading)
    title_match = re.search(r'^# (.+)$', article_text, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled Article"
    
    # Extract date if present
    date_match = re.search(r'Date: (\d{4}-\d{2}-\d{2})', article_text)
    date = date_match.group(1) if date_match else "Unknown Date"
    
    # Extract source if present
    source_match = re.search(r'Source: (.+?)$', article_text, re.MULTILINE)
    source = source_match.group(1) if source_match else "Unknown Source"
    
    return {
        "title": title,
        "date": date,
        "source": source
    }

def format_article_for_prompt(article: Dict[str, str], index: int) -> str:
    """Format a single article for inclusion in the prompt"""
    metadata = extract_article_metadata(article["content"])
    
    return f"""
ARTICLE {index + 1}: {metadata['title']}
SOURCE: {metadata['source']}
DATE: {metadata['date']}
CONTENT:
{article['content']}
---
"""

def format_example_for_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """Format a training example for inclusion in the prompt"""
    input_text = "# INPUT ARTICLES:\n\n"
    
    for i, article in enumerate(example["input_articles"]):
        input_text += format_article_for_prompt(article, i)
    
    output_text = f"# SUMMARY DIGEST:\n\n{example['expected_output']}"
    
    return {
        "input": input_text,
        "output": output_text
    }

def build_base_system_prompt() -> str:
    """Build the base system prompt for the summarizer"""
    return """You are an expert financial news analyst and writer. Your task is to analyze multiple news articles and create a concise, informative digest that captures the key information, trends, and insights across all articles.

Follow these guidelines:
1. Identify the most important facts, events, and trends across all articles
2. Highlight consensus views and note significant disagreements between sources
3. Include relevant data points, statistics, and quotes that support key points
4. Maintain a professional, objective tone appropriate for financial analysis
5. Organize the digest with clear sections and a logical flow
6. Begin with a title and executive summary of the main points
7. End with key takeaways or implications for investors

I'll provide you with several examples of input articles and their corresponding digests. Learn from these examples to understand the style, format, and level of detail expected.
"""

def build_transformation_prompt(training_examples: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """
    Build the transformation prompt from training examples.
    
    Args:
        training_examples: List of training examples
        num_examples: Number of examples to include in the prompt
        
    Returns:
        The transformation prompt
    """
    if not training_examples:
        raise ValueError("No training examples provided")
    
    # Use at most the specified number of examples
    examples_to_use = training_examples[:num_examples]
    
    # Format system prompt
    system_prompt = build_base_system_prompt()

    # Format examples
    formatted_examples = [format_example_for_prompt(ex) for ex in examples_to_use]
    
    # Build the complete transformation prompt
    transformation_prompt = f"{system_prompt}\n\n"
    
    for i, example in enumerate(formatted_examples):
        transformation_prompt += f"EXAMPLE {i+1}:\n\n"
        transformation_prompt += f"{example['input']}\n\n"
        transformation_prompt += f"{example['output']}\n\n"
        transformation_prompt += "---\n\n"
    
    transformation_prompt += "Now, analyze the following articles and create a similar digest:\n\n"
    
    logger.info(f"Built transformation prompt with {num_examples} examples")
    
    return transformation_prompt

def build_hierarchical_prompt(summary_type: str = "individual") -> str:
    """
    Build a prompt for hierarchical summarization.
    
    Args:
        summary_type: Type of summary ("individual" or "combined")
        
    Returns:
        The hierarchical prompt
    """
    if summary_type == "individual":
        return """You are an expert financial news analyst. Summarize the following article, capturing the key facts, figures, and insights. Focus on information that would be relevant for a financial digest.

Extract:
1. Main topic and key events
2. Important data points and statistics
3. Expert opinions and market implications
4. Unique insights not commonly found in other articles

Keep the summary concise but comprehensive, maintaining the professional tone of financial analysis.

ARTICLE:
"""
    elif summary_type == "combined":
        return """You are an expert financial news analyst. Create a comprehensive digest that synthesizes the following article summaries into a cohesive analysis.

Focus on:
1. Identifying common themes and trends across the summaries
2. Highlighting consensus views and noting significant disagreements
3. Presenting a complete picture of the financial situation or event
4. Providing key takeaways and implications for investors

Maintain a professional, objective tone appropriate for financial analysis. Organize the digest with clear sections and a logical flow.

ARTICLE SUMMARIES:
"""
    else:
        raise ValueError(f"Unsupported summary type: {summary_type}")

def save_transformation_prompt(prompt: str, output_file: str):
    """
    Save the transformation prompt to a file.
    
    Args:
        prompt: Transformation prompt
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    logger.info(f"Saved transformation prompt to {output_file}")

def load_transformation_prompt(input_file: str) -> str:
    """
    Load a transformation prompt from a file.
    
    Args:
        input_file: Path to input file
        
    Returns:
        The transformation prompt
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    logger.info(f"Loaded transformation prompt from {input_file}")
    
    return prompt
