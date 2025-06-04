"""
Data loading functionality for the advanced summarizer module.
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_training_data(train_dir: str) -> List[Dict[str, Any]]:
    """
    Load training data from directory structure.
    
    Args:
        train_dir: Path to directory containing training folders
        
    Returns:
        List of training examples
    """
    logger.info(f"Loading training data from {train_dir}")
    training_examples = []
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(train_dir, subdir)
        
        # Find all training articles (prefixed with "train_")
        train_files = glob.glob(os.path.join(subdir_path, "train_*.md"))
        
        # Find the expected output (prefixed with "test_")
        test_files = glob.glob(os.path.join(subdir_path, "test_*.md"))
        
        if not train_files or not test_files:
            logger.warning(f"Skipping {subdir}: Missing train or test files")
            continue
        
        # Load training articles
        input_articles = []
        for train_file in train_files:
            with open(train_file, 'r', encoding='utf-8') as f:
                article_text = f.read()
                article_name = os.path.basename(train_file)
                input_articles.append({
                    "filename": article_name,
                    "content": article_text
                })
        
        # Load expected output
        with open(test_files[0], 'r', encoding='utf-8') as f:
            expected_output = f.read()
        
        # Create training example
        training_examples.append({
            "folder": subdir,
            "input_articles": input_articles,
            "expected_output": expected_output
        })
        
        logger.info(f"Loaded example from {subdir}: {len(input_articles)} articles")
    
    logger.info(f"Loaded {len(training_examples)} training examples")
    
    return training_examples

def load_new_articles(articles_dir: str) -> List[Dict[str, str]]:
    """
    Load new articles from a directory.
    
    Args:
        articles_dir: Path to directory containing articles
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Find all markdown files
    article_files = glob.glob(os.path.join(articles_dir, "*.md"))
    
    for article_file in article_files:
        with open(article_file, 'r', encoding='utf-8') as f:
            article_text = f.read()
            article_name = os.path.basename(article_file)
            articles.append({
                "filename": article_name,
                "content": article_text
            })
    
    logger.info(f"Loaded {len(articles)} new articles from {articles_dir}")
    
    return articles

def split_articles_for_context_window(articles: List[Dict[str, str]], max_tokens_per_chunk: int = 3000) -> List[List[Dict[str, str]]]:
    """
    Split articles into chunks that fit within context window limits.
    
    Args:
        articles: List of article dictionaries
        max_tokens_per_chunk: Maximum tokens per chunk
        
    Returns:
        List of article chunks
    """
    # Simple estimation: ~1.3 tokens per word
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for article in articles:
        article_tokens = estimate_tokens(article["content"])
        
        # If this article alone exceeds the limit, it needs special handling
        if article_tokens > max_tokens_per_chunk:
            # If we have accumulated articles, add them as a chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            # Add this large article as its own chunk (will be handled by hierarchical summarization)
            chunks.append([article])
            continue
        
        # If adding this article would exceed the limit, start a new chunk
        if current_tokens + article_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        # Add article to current chunk
        current_chunk.append(article)
        current_tokens += article_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Split {len(articles)} articles into {len(chunks)} chunks")
    
    return chunks

def save_summary(summary: str, output_file: str):
    """
    Save a summary to a file.
    
    Args:
        summary: Summary text
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Saved summary to {output_file}")
