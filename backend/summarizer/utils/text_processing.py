# utils/text_processing.py
"""Text extraction and formatting utilities."""

import re
from typing import List, Dict, Any
from pathlib import Path


class TextProcessor:
    """Utilities for text processing and formatting."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\-.,!?%$()[\]{}:;\'"]', '', text)

        return text.strip()

    @staticmethod
    def extract_bullet_points(text: str) -> List[str]:
        """Extract bullet points from text."""
        bullet_pattern = r'^[\s]*[-*•]\s+(.+)
        bullets = []
        
        for line in text.split('\n'):
            match = re.match(bullet_pattern, line)
            if match:
                bullets.append(match.group(1).strip())
        
        return bullets
    
    @staticmethod
    def format_as_bullets(points: List[str]) -> str:
        """Format list of points as bullet points."""
        return '\n'.join(f'- {point}' for point in points)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text to maximum length while preserving word boundaries."""
        if len(text <= max_length:
            return text
        
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can find a space reasonably close
            return truncated[:last_space] + '...'
        
        return truncated + '...'
    
    @staticmethod
    def extract_financial_entities(text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text."""
        entities = {
            'companies': [],
            'indices': [],
            'percentages': [],
            'currencies': []
        }
        
        # Company patterns (simplified)
        company_pattern = r'\b[A-Z][a-zA-Z&\s]+(Inc|Corp|Ltd|Co|Group|Bank)\b'
        entities['companies'] = re.findall(company_pattern, text)
        
        # Index patterns
        index_pattern = r'\b(S&P 500|Nasdaq|Dow Jones|FTSE|DAX|CAC|Nikkei|Hang Seng)\b'
        entities['indices'] = re.findall(index_pattern, text, re.IGNORECASE)
        
        # Percentage patterns
        percentage_pattern = r'\b\d+\.?\d*%\b'
        entities['percentages'] = re.findall(percentage_pattern, text)
        
        # Currency patterns
        currency_pattern = r'\$\d+\.?\d*[BMK]?'
        entities['currencies'] = re.findall(currency_pattern, text)
        
        return entities