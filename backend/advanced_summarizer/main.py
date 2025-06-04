"""
Main module for the advanced summarizer.
"""

import os
import argparse
import logging
import json
from typing import List, Dict, Any, Optional

from .core.data_loader import load_training_data, load_new_articles, save_summary
from .core.prompt_builder import build_transformation_prompt, save_transformation_prompt, load_transformation_prompt
from .models.base import create_model_interface
from .graph.workflow import build_summarization_graph, GraphState
from .evaluation.evaluator import SummaryEvaluator
from .utils.context_window import ContextWindowManager
from .core.progressive_training import ProgressiveTrainer, DynamicExampleSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSummarizer:
    """
    Advanced summarizer with evaluation, context window management, and progressive training.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        provider: str = "openai",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        api_key: Optional[str] = None
    ):
        """
        Initialize the advanced summarizer.
        
        Args:
            model_name: Name of the LLM model to use
            provider: LLM provider ('openai' or 'google')
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            api_key: Optional API key (otherwise uses environment variables)
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # Initialize components
        self.model_interface = create_model_interface(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        self.context_window_manager = ContextWindowManager(max_tokens_per_chunk=3000)
        self.evaluator = SummaryEvaluator(self.model_interface)
        self.example_selector = DynamicExampleSelector(self.context_window_manager)
        self.trainer = ProgressiveTrainer(
            self.model_interface,
            self.evaluator,
            self.context_window_manager
        )
        
        # Storage for training examples and learned prompt
        self.training_examples = []
        self.transformation_prompt = None
        self.graph = None
    
    def load_training_data(self, train_dir: str) -> List[Dict[str, Any]]:
        """
        Load training data from directory structure.
        
        Args:
            train_dir: Path to directory containing training folders
            
        Returns:
            List of training examples
        """
        self.training_examples = load_training_data(train_dir)
        return self.training_examples
    
    def build_transformation_prompt(self, num_examples: int = 3) -> str:
        """
        Build the transformation prompt from training examples.
        
        Args:
            num_examples: Number of examples to include in the prompt
            
        Returns:
            The transformation prompt
        """
        if not self.training_examples:
            raise ValueError("No training examples loaded. Call load_training_data first.")
        
        # Select representative examples
        selected_examples = self.context_window_manager.select_representative_examples(
            self.training_examples, num_examples=num_examples
        )
        
        # Build prompt
        self.transformation_prompt = build_transformation_prompt(selected_examples, num_examples)
        
        return self.transformation_prompt
    
    def train_progressively(
        self, 
        validation_examples: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Train the summarizer progressively, improving the prompt based on evaluation.
        
        Args:
            validation_examples: Optional list of validation examples (uses subset of training if None)
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Training results
        """
        if not self.training_examples:
            raise ValueError("No training examples loaded. Call load_training_data first.")
        
        # Update trainer settings
        self.trainer.max_iterations = max_iterations
        
        # Train progressively
        results = self.trainer.train_iteratively(
            self.training_examples,
            validation_examples,
            self.transformation_prompt
        )
        
        # Update transformation prompt
        self.transformation_prompt = results["best_prompt"]
        
        return results
    
    def summarize(self, articles: List[Dict[str, str]], use_dynamic_examples: bool = True) -> str:
        """
        Summarize a set of articles using the learned transformation.
        
        Args:
            articles: List of article dictionaries with 'filename' and 'content' keys
            use_dynamic_examples: Whether to use dynamic example selection
            
        Returns:
            Generated summary digest
        """
        if not self.transformation_prompt and not use_dynamic_examples:
            raise ValueError("Transformation prompt not built. Call build_transformation_prompt first.")
        
        # Check if articles exceed context window
        chunks = self.context_window_manager.chunk_articles(articles)
        use_hierarchical = len(chunks) > 1
        
        # If using dynamic examples, build a custom prompt for these articles
        if use_dynamic_examples and self.training_examples:
            prompt = self.example_selector.build_dynamic_prompt(
                self.training_examples, articles
            )
        else:
            prompt = self.transformation_prompt
        
        # Build graph
        graph = build_summarization_graph(self.model_interface, use_hierarchical=use_hierarchical)
        
        if use_hierarchical:
            # Process each chunk and combine
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} articles")
                
                # Run graph on this chunk
                result = graph.invoke({
                    "articles": chunk,
                    "prompt": prompt
                })
                
                # Store chunk summary
                chunk_summaries.append(result["summary"])
            
            # If we have multiple chunk summaries, combine them
            if len(chunk_summaries) > 1:
                # Create pseudo-articles from chunk summaries
                summary_articles = [
                    {"filename": f"chunk_{i}.md", "content": summary}
                    for i, summary in enumerate(chunk_summaries)
                ]
                
                # Run graph again to combine
                result = graph.invoke({
                    "articles": summary_articles,
                    "prompt": prompt
                })
                
                final_summary = result["summary"]
            else:
                final_summary = chunk_summaries[0]
        else:
            # Process all articles at once
            result = graph.invoke({
                "articles": articles,
                "prompt": prompt
            })
            
            final_summary = result["summary"]
        
        return final_summary
    
    def evaluate_on_test_examples(self, test_examples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate the summarizer on test examples.
        
        Args:
            test_examples: Optional list of test examples (uses training examples if None)
            
        Returns:
            Evaluation results
        """
        if test_examples is None:
            # Use a subset of training examples as test examples
            test_examples = self.training_examples[-2:] if len(self.training_examples) > 2 else self.training_examples
        
        generated_summaries = []
        
        for example in test_examples:
            # Generate summary
            generated_summary = self.summarize(example["input_articles"])
            generated_summaries.append(generated_summary)
        
        # Evaluate summaries
        evaluation_results = self.evaluator.evaluate_batch(test_examples, generated_summaries)
        
        # Calculate average score
        avg_score = sum(r["metrics"]["overall_score"] for r in evaluation_results) / len(evaluation_results)
        
        return {
            "num_examples": len(evaluation_results),
            "average_score": avg_score,
            "results": evaluation_results
        }
    
    def save_transformation_prompt(self, output_file: str):
        """
        Save the transformation prompt to a file.
        
        Args:
            output_file: Path to output file
        """
        if not self.transformation_prompt:
            raise ValueError("Transformation prompt not built. Call build_transformation_prompt first.")
        
        save_transformation_prompt(self.transformation_prompt, output_file)
    
    def load_transformation_prompt(self, input_file: str):
        """
        Load a transformation prompt from a file.
        
        Args:
            input_file: Path to input file
        """
        self.transformation_prompt = load_transformation_prompt(input_file)
    
    def load_new_articles(self, articles_dir: str) -> List[Dict[str, str]]:
        """
        Load new articles from a directory.
        
        Args:
            articles_dir: Path to directory containing articles
            
        Returns:
            List of article dictionaries
        """
        return load_new_articles(articles_dir)
    
    def save_summary(self, summary: str, output_file: str):
        """
        Save a summary to a file.
        
        Args:
            summary: Summary text
            output_file: Path to output file
        """
        save_summary(summary, output_file)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Advanced News Summarization with LangGraph")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train on examples and build transformation prompt")
    mode_group.add_argument("--progressive-train", action="store_true", help="Train progressively with evaluation feedback")
    mode_group.add_argument("--summarize", action="store_true", help="Summarize new articles using existing prompt")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate on test examples")
    
    # Common arguments
    parser.add_argument("--model", default="gpt-4-turbo", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "google"], help="LLM provider")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for generation")
    
    # Training arguments
    parser.add_argument("--train_dir", help="Directory containing training examples")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to include in prompt")
    parser.add_argument("--prompt_output", help="File to save transformation prompt")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum iterations for progressive training")
    
    # Summarization arguments
    parser.add_argument("--prompt_input", help="File containing transformation prompt")
    parser.add_argument("--articles_dir", help="Directory containing new articles to summarize")
    parser.add_argument("--output_file", help="File to save generated summary")
    parser.add_argument("--dynamic_examples", action="store_true", help="Use dynamic example selection")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = AdvancedSummarizer(
        model_name=args.model,
        provider=args.provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    if args.train or args.progressive_train:
        # Training mode
        if not args.train_dir:
            parser.error("--train_dir is required for training mode")
        if not args.prompt_output:
            parser.error("--prompt_output is required for training mode")
        
        # Load training data
        summarizer.load_training_data(args.train_dir)
        
        if args.progressive_train:
            # Progressive training
            results = summarizer.train_progressively(max_iterations=args.max_iterations)
            
            # Save training history
            history_file = os.path.splitext(args.prompt_output)[0] + "_history.json"
            summarizer.trainer.save_training_history(history_file)
            
            logger.info(f"Progressive training completed with best score: {results['best_score']:.3f}")
        else:
            # Basic training
            summarizer.build_transformation_prompt(num_examples=args.num_examples)
        
        # Save transformation prompt
        summarizer.save_transformation_prompt(args.prompt_output)
        
        # Optional evaluation
        if args.evaluate:
            results = summarizer.evaluate_on_test_examples()
            print(json.dumps(results, indent=2))
    
    elif args.summarize:
        # Summarization mode
        if not args.articles_dir:
            parser.error("--articles_dir is required for summarization mode")
        if not args.output_file:
            parser.error("--output_file is required for summarization mode")
        
        # Load transformation prompt if not using dynamic examples
        if not args.dynamic_examples:
            if not args.prompt_input:
                parser.error("--prompt_input is required when not using dynamic examples")
            summarizer.load_transformation_prompt(args.prompt_input)
        elif args.train_dir:
            # Load training data for dynamic example selection
            summarizer.load_training_data(args.train_dir)
        
        # Load new articles
        articles = summarizer.load_new_articles(args.articles_dir)
        
        # Generate summary
        summary = summarizer.summarize(articles, use_dynamic_examples=args.dynamic_examples)
        
        # Save summary
        summarizer.save_summary(summary, args.output_file)
    
    elif args.evaluate:
        # Evaluation mode
        if not args.train_dir:
            parser.error("--train_dir is required for evaluation mode")
        
        # Load training data
        test_examples = summarizer.load_training_data(args.train_dir)
        
        # Load transformation prompt if not using dynamic examples
        if not args.dynamic_examples:
            if not args.prompt_input:
                parser.error("--prompt_input is required when not using dynamic examples")
            summarizer.load_transformation_prompt(args.prompt_input)
        
        # Evaluate
        results = summarizer.evaluate_on_test_examples(test_examples)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
