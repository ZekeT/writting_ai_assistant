"""
Progressive training and dynamic example selection for the advanced summarizer module.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class ProgressiveTrainer:
    """
    Progressive trainer for iterative improvement of summarization.
    """
    
    def __init__(
        self, 
        model_interface, 
        evaluator, 
        context_window_manager,
        max_iterations: int = 3,
        improvement_threshold: float = 0.05
    ):
        """
        Initialize the progressive trainer.
        
        Args:
            model_interface: Model interface for generation
            evaluator: Summary evaluator
            context_window_manager: Context window manager
            max_iterations: Maximum number of improvement iterations
            improvement_threshold: Minimum improvement threshold to continue iterations
        """
        self.model_interface = model_interface
        self.evaluator = evaluator
        self.context_window_manager = context_window_manager
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        
        # Training history
        self.training_history = []
    
    def train_iteratively(
        self, 
        training_examples: List[Dict[str, Any]], 
        validation_examples: Optional[List[Dict[str, Any]]] = None,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the summarizer iteratively, improving the prompt based on evaluation.
        
        Args:
            training_examples: List of training examples
            validation_examples: Optional list of validation examples (uses subset of training if None)
            initial_prompt: Optional initial transformation prompt
            
        Returns:
            Training results
        """
        from ..core.prompt_builder import build_transformation_prompt
        
        # If no validation examples provided, use a subset of training examples
        if validation_examples is None:
            # Use 20% of training examples for validation
            val_size = max(1, int(len(training_examples) * 0.2))
            validation_examples = training_examples[-val_size:]
            training_examples = training_examples[:-val_size]
        
        # Select representative training examples
        selected_examples = self.context_window_manager.select_representative_examples(
            training_examples, num_examples=3
        )
        
        # Build initial transformation prompt
        if initial_prompt:
            current_prompt = initial_prompt
        else:
            current_prompt = build_transformation_prompt(selected_examples)
        
        best_prompt = current_prompt
        best_score = 0.0
        
        # Iterative improvement
        for iteration in range(self.max_iterations):
            logger.info(f"Starting training iteration {iteration+1}/{self.max_iterations}")
            
            # Evaluate current prompt on validation examples
            evaluation_results, avg_score = self._evaluate_prompt(current_prompt, validation_examples)
            
            # Record results
            iteration_result = {
                "iteration": iteration,
                "prompt": current_prompt,
                "evaluation_results": evaluation_results,
                "average_score": avg_score
            }
            self.training_history.append(iteration_result)
            
            # Update best prompt if improved
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = current_prompt
            
            # Stop if this is the last iteration
            if iteration == self.max_iterations - 1:
                break
            
            # Generate improved prompt
            improved_prompt = self.evaluator.generate_prompt_improvement(
                current_prompt, evaluation_results
            )
            
            # Evaluate improvement
            _, improved_score = self._evaluate_prompt(improved_prompt, validation_examples)
            
            # If improvement is significant, use the improved prompt
            if improved_score > avg_score + self.improvement_threshold:
                current_prompt = improved_prompt
                logger.info(f"Prompt improved: score increased from {avg_score:.3f} to {improved_score:.3f}")
            else:
                logger.info(f"No significant improvement: {improved_score:.3f} vs {avg_score:.3f}")
                # Early stopping if no significant improvement
                break
        
        logger.info(f"Training completed after {len(self.training_history)} iterations")
        logger.info(f"Best average score: {best_score:.3f}")
        
        return {
            "best_prompt": best_prompt,
            "best_score": best_score,
            "training_history": self.training_history
        }
    
    def _evaluate_prompt(
        self, 
        prompt: str, 
        validation_examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Evaluate a prompt on validation examples.
        
        Args:
            prompt: Transformation prompt
            validation_examples: List of validation examples
            
        Returns:
            Tuple of (evaluation results, average score)
        """
        from ..graph.workflow import build_summarization_graph, GraphState
        
        evaluation_results = []
        total_score = 0.0
        
        for example in validation_examples:
            # Build graph
            graph = build_summarization_graph(self.model_interface)
            
            # Run graph
            result = graph.invoke({
                "articles": example["input_articles"],
                "prompt": prompt
            })
            
            # Get generated summary
            generated_summary = result["summary"]
            
            # Evaluate summary
            evaluation = self.evaluator.evaluate_summary(
                generated_summary, example["expected_output"]
            )
            
            # Store result
            evaluation_result = {
                "folder": example.get("folder", "unknown"),
                "metrics": evaluation["metrics"],
                "feedback": evaluation["feedback"]
            }
            evaluation_results.append(evaluation_result)
            
            # Add to total score
            total_score += evaluation["metrics"]["overall_score"]
        
        # Calculate average score
        avg_score = total_score / len(validation_examples) if validation_examples else 0.0
        
        return evaluation_results, avg_score
    
    def save_training_history(self, output_file: str):
        """
        Save training history to a file.
        
        Args:
            output_file: Path to output file
        """
        # Create a simplified version of history for saving
        simplified_history = []
        
        for item in self.training_history:
            simplified_item = {
                "iteration": item["iteration"],
                "average_score": item["average_score"],
                "evaluation_summary": {
                    "num_examples": len(item["evaluation_results"]),
                    "scores": [r["metrics"]["overall_score"] for r in item["evaluation_results"]]
                }
            }
            simplified_history.append(simplified_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_history, f, indent=2)
        
        logger.info(f"Saved training history to {output_file}")
    
    def load_training_history(self, input_file: str):
        """
        Load training history from a file.
        
        Args:
            input_file: Path to input file
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            self.training_history = json.load(f)
        
        logger.info(f"Loaded training history from {input_file}")


class DynamicExampleSelector:
    """
    Dynamic example selector for adaptive few-shot learning.
    """
    
    def __init__(self, context_window_manager):
        """
        Initialize the dynamic example selector.
        
        Args:
            context_window_manager: Context window manager
        """
        self.context_window_manager = context_window_manager
    
    def select_examples_for_articles(
        self, 
        training_examples: List[Dict[str, Any]], 
        articles: List[Dict[str, str]],
        num_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select the most relevant examples for the given articles.
        
        Args:
            training_examples: List of training examples
            articles: List of articles to summarize
            num_examples: Number of examples to select
            
        Returns:
            Selected examples
        """
        return self.context_window_manager.select_relevant_examples(
            training_examples, articles, num_examples
        )
    
    def build_dynamic_prompt(
        self, 
        training_examples: List[Dict[str, Any]], 
        articles: List[Dict[str, str]],
        num_examples: int = 3,
        base_prompt: Optional[str] = None
    ) -> str:
        """
        Build a dynamic prompt with examples selected based on the articles.
        
        Args:
            training_examples: List of training examples
            articles: List of articles to summarize
            num_examples: Number of examples to select
            base_prompt: Optional base prompt to use
            
        Returns:
            Dynamic prompt
        """
        from ..core.prompt_builder import build_transformation_prompt, build_base_system_prompt
        
        # Select relevant examples
        selected_examples = self.select_examples_for_articles(
            training_examples, articles, num_examples
        )
        
        # Build prompt
        if base_prompt:
            # Use provided base prompt
            prompt = base_prompt
        else:
            # Build from scratch
            prompt = build_transformation_prompt(selected_examples)
        
        return prompt
