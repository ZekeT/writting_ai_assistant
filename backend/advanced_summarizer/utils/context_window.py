"""
Context window management strategies for the advanced summarizer module.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

class ContextWindowManager:
    """
    Manager for handling context window limitations.
    """
    
    def __init__(self, max_tokens_per_chunk: int = 3000):
        """
        Initialize the context window manager.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)
    
    def chunk_articles(self, articles: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """
        Split articles into chunks that fit within context window limits.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of article chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for article in articles:
            article_tokens = self.estimate_tokens(article["content"])
            
            # If this article alone exceeds the limit, it needs special handling
            if article_tokens > self.max_tokens_per_chunk:
                # If we have accumulated articles, add them as a chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Add this large article as its own chunk (will be handled by hierarchical summarization)
                chunks.append([article])
                continue
            
            # If adding this article would exceed the limit, start a new chunk
            if current_tokens + article_tokens > self.max_tokens_per_chunk and current_chunk:
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
    
    def select_representative_examples(self, examples: List[Dict[str, Any]], num_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Select the most representative examples for few-shot learning.
        
        Args:
            examples: List of training examples
            num_examples: Number of examples to select
            
        Returns:
            Selected examples
        """
        if len(examples) <= num_examples:
            return examples
        
        # Extract text from examples
        texts = []
        for example in examples:
            # Combine input articles and expected output
            example_text = ""
            for article in example["input_articles"]:
                example_text += article["content"] + " "
            example_text += example["expected_output"]
            texts.append(example_text)
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Calculate average similarity for each example
        avg_similarities = np.mean(similarities, axis=1)
        
        # Select examples with highest average similarity (most representative)
        selected_indices = np.argsort(-avg_similarities)[:num_examples]
        
        # Return selected examples
        selected_examples = [examples[i] for i in selected_indices]
        
        logger.info(f"Selected {num_examples} representative examples from {len(examples)}")
        
        return selected_examples
    
    def select_relevant_examples(self, examples: List[Dict[str, Any]], articles: List[Dict[str, str]], num_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Select examples most relevant to the current articles.
        
        Args:
            examples: List of training examples
            articles: List of articles to summarize
            num_examples: Number of examples to select
            
        Returns:
            Selected examples
        """
        if len(examples) <= num_examples:
            return examples
        
        # Extract text from articles
        article_text = ""
        for article in articles:
            article_text += article["content"] + " "
        
        # Extract text from examples
        example_texts = []
        for example in examples:
            # Combine input articles
            example_text = ""
            for article in example["input_articles"]:
                example_text += article["content"] + " "
            example_texts.append(example_text)
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        example_vectors = vectorizer.fit_transform(example_texts)
        article_vector = vectorizer.transform([article_text])
        
        # Calculate similarities
        similarities = cosine_similarity(article_vector, example_vectors)[0]
        
        # Select examples with highest similarity to current articles
        selected_indices = np.argsort(-similarities)[:num_examples]
        
        # Return selected examples
        selected_examples = [examples[i] for i in selected_indices]
        
        logger.info(f"Selected {num_examples} relevant examples for current articles")
        
        return selected_examples
