#!/usr/bin/env python3
"""
Market Close Summary Generator with Few-Shot Learning

This script helps you create few-shot examples from existing market summaries and news articles,
then applies this learning to generate summaries for new articles.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

# You'll need to install these packages:
# pip install openai langchain

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class MarketSummaryGenerator:
    """
    A class to generate market close summaries using few-shot learning from existing examples.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the generator with API key and model.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (default: gpt-4)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.model = model
        self.llm = ChatOpenAI(temperature=0.3, model=model, api_key=self.api_key)
        self.few_shot_examples = []
        
    def load_examples_from_files(self, 
                                 articles_dir: str, 
                                 summaries_dir: str, 
                                 date_pattern: str = r'(\d{4}-\d{2}-\d{2})') -> None:
        """
        Load examples from directories containing articles and summaries.
        
        Args:
            articles_dir: Directory containing article files
            summaries_dir: Directory containing summary files
            date_pattern: Regex pattern to extract dates from filenames
        """
        article_files = os.listdir(articles_dir)
        summary_files = os.listdir(summaries_dir)
        
        # Create a mapping of dates to files
        article_dates = {}
        for filename in article_files:
            match = re.search(date_pattern, filename)
            if match:
                date = match.group(1)
                article_dates[date] = os.path.join(articles_dir, filename)
        
        summary_dates = {}
        for filename in summary_files:
            match = re.search(date_pattern, filename)
            if match:
                date = match.group(1)
                summary_dates[date] = os.path.join(summaries_dir, filename)
        
        # Find matching dates
        common_dates = set(article_dates.keys()) & set(summary_dates.keys())
        
        # Load the examples
        for date in common_dates:
            with open(article_dates[date], 'r', encoding='utf-8') as f:
                article_text = f.read()
            
            with open(summary_dates[date], 'r', encoding='utf-8') as f:
                summary_text = f.read()
            
            self.few_shot_examples.append({
                'date': date,
                'article': article_text,
                'summary': summary_text
            })
        
        print(f"Loaded {len(self.few_shot_examples)} examples")
    
    def load_examples_from_dict(self, examples: List[Dict[str, str]]) -> None:
        """
        Load examples from a list of dictionaries.
        
        Args:
            examples: List of dictionaries with 'date', 'article', and 'summary' keys
        """
        self.few_shot_examples = examples
        print(f"Loaded {len(self.few_shot_examples)} examples")
    
    def create_few_shot_prompt(self, num_examples: int = 3) -> str:
        """
        Create a few-shot prompt from the loaded examples.
        
        Args:
            num_examples: Number of examples to include in the prompt
            
        Returns:
            Formatted few-shot prompt
        """
        if not self.few_shot_examples:
            raise ValueError("No examples loaded. Call load_examples_* first.")
        
        # Use the most recent examples if we have more than requested
        examples_to_use = self.few_shot_examples[-num_examples:] if len(self.few_shot_examples) > num_examples else self.few_shot_examples
        
        prompt = "I'll provide you with news articles about the US stock market, and I'd like you to generate a concise market close summary in the same style as these examples:\n\n"
        
        for i, example in enumerate(examples_to_use, 1):
            prompt += f"Example {i} - Date: {example['date']}\n"
            prompt += f"Articles:\n{example['article'][:1000]}...\n\n"  # Truncate long articles
            prompt += f"Summary:\n{example['summary']}\n\n"
            prompt += "-" * 80 + "\n\n"
        
        prompt += "Now, please generate a similar summary for the following articles:\n\n"
        return prompt
    
    def generate_summary(self, new_articles: str, date: Optional[str] = None) -> str:
        """
        Generate a summary for new articles using few-shot learning.
        
        Args:
            new_articles: Text of the new articles to summarize
            date: Date for the new articles (optional)
            
        Returns:
            Generated summary
        """
        few_shot_prompt = self.create_few_shot_prompt()
        
        date_str = f"Date: {date}\n" if date else ""
        full_prompt = f"{few_shot_prompt}{date_str}Articles:\n{new_articles}\n\nSummary:"
        
        messages = [
            SystemMessage(content="You are a financial analyst who writes concise, informative market close summaries."),
            HumanMessage(content=full_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def save_examples_to_json(self, output_file: str) -> None:
        """
        Save the loaded examples to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.few_shot_examples, f, indent=2)
        
        print(f"Saved {len(self.few_shot_examples)} examples to {output_file}")
    
    def load_examples_from_json(self, input_file: str) -> None:
        """
        Load examples from a JSON file.
        
        Args:
            input_file: Path to the input JSON file
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            self.few_shot_examples = json.load(f)
        
        print(f"Loaded {len(self.few_shot_examples)} examples from {input_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate market close summaries using few-shot learning")
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create examples parser
    create_parser = subparsers.add_parser("create", help="Create few-shot examples from files")
    create_parser.add_argument("--articles-dir", required=True, help="Directory containing article files")
    create_parser.add_argument("--summaries-dir", required=True, help="Directory containing summary files")
    create_parser.add_argument("--output", required=True, help="Output JSON file for examples")
    create_parser.add_argument("--date-pattern", default=r'(\d{4}-\d{2}-\d{2})', help="Regex pattern to extract dates from filenames")
    
    # Generate summary parser
    generate_parser = subparsers.add_parser("generate", help="Generate a summary for new articles")
    generate_parser.add_argument("--examples", required=True, help="JSON file containing examples")
    generate_parser.add_argument("--articles", required=True, help="File containing new articles to summarize")
    generate_parser.add_argument("--output", required=True, help="Output file for the generated summary")
    generate_parser.add_argument("--date", help="Date for the new articles")
    generate_parser.add_argument("--model", default="gpt-4", help="Model to use (default: gpt-4)")
    
    args = parser.parse_args()
    
    if args.command == "create":
        generator = MarketSummaryGenerator()
        generator.load_examples_from_files(args.articles_dir, args.summaries_dir, args.date_pattern)
        generator.save_examples_to_json(args.output)
    
    elif args.command == "generate":
        generator = MarketSummaryGenerator(model=args.model)
        generator.load_examples_from_json(args.examples)
        
        with open(args.articles, 'r', encoding='utf-8') as f:
            new_articles = f.read()
        
        summary = generator.generate_summary(new_articles, args.date)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Generated summary saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
