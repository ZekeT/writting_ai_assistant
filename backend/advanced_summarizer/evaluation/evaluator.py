"""
Evaluation functionality for the advanced summarizer module.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import re
from difflib import SequenceMatcher

# Configure logging
logger = logging.getLogger(__name__)

class SummaryEvaluator:
    """
    Evaluator for summary quality and feedback generation.
    """
    
    def __init__(self, model_interface):
        """
        Initialize the evaluator.
        
        Args:
            model_interface: Model interface for evaluation
        """
        self.model_interface = model_interface
    
    def evaluate_summary(self, generated_summary: str, expected_summary: str) -> Dict[str, Any]:
        """
        Evaluate a generated summary against an expected summary.
        
        Args:
            generated_summary: Generated summary
            expected_summary: Expected summary
            
        Returns:
            Evaluation metrics
        """
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(generated_summary, expected_summary)
        
        # Get detailed feedback
        feedback = self._generate_detailed_feedback(generated_summary, expected_summary)
        
        return {
            "metrics": metrics,
            "feedback": feedback
        }
    
    def _calculate_basic_metrics(self, generated_summary: str, expected_summary: str) -> Dict[str, float]:
        """
        Calculate basic evaluation metrics.
        
        Args:
            generated_summary: Generated summary
            expected_summary: Expected summary
            
        Returns:
            Metrics dictionary
        """
        # Length comparison
        gen_length = len(generated_summary.split())
        exp_length = len(expected_summary.split())
        length_ratio = gen_length / max(1, exp_length)
        
        # Similarity using sequence matcher
        similarity = SequenceMatcher(None, generated_summary, expected_summary).ratio()
        
        # Key phrase coverage
        key_phrases = self._extract_key_phrases(expected_summary)
        covered_phrases = sum(1 for phrase in key_phrases if phrase.lower() in generated_summary.lower())
        phrase_coverage = covered_phrases / max(1, len(key_phrases)) if key_phrases else 0
        
        return {
            "length_ratio": length_ratio,
            "similarity": similarity,
            "phrase_coverage": phrase_coverage,
            "overall_score": (similarity * 0.5) + (phrase_coverage * 0.5)
        }
    
    def _extract_key_phrases(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract key phrases from text.
        
        Args:
            text: Input text
            min_length: Minimum phrase length in words
            
        Returns:
            List of key phrases
        """
        # Extract sentences
        sentences = re.split(r'[.!?]', text)
        
        # Extract potential key phrases (nouns, numbers, proper nouns)
        phrases = []
        for sentence in sentences:
            # Simple heuristic: look for capitalized phrases, numbers, and quoted text
            # This is a simplified approach; in a real implementation, use NLP libraries
            
            # Find quoted text
            quoted = re.findall(r'"([^"]*)"', sentence)
            phrases.extend([q for q in quoted if len(q.split()) >= min_length])
            
            # Find capitalized phrases
            capitalized = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)', sentence)
            phrases.extend([c for c in capitalized if len(c.split()) >= min_length])
            
            # Find phrases with numbers
            with_numbers = re.findall(r'(\d+(?:\.\d+)?%?(?:\s+[a-zA-Z]+){2,})', sentence)
            phrases.extend([n for n in with_numbers if len(n.split()) >= min_length])
        
        return list(set(phrases))
    
    def _generate_detailed_feedback(self, generated_summary: str, expected_summary: str) -> Dict[str, Any]:
        """
        Generate detailed feedback using the LLM.
        
        Args:
            generated_summary: Generated summary
            expected_summary: Expected summary
            
        Returns:
            Detailed feedback
        """
        prompt = f"""You are an expert evaluator of financial news summaries. Compare the following generated summary with the expected summary and provide detailed feedback.

Focus on:
1. Content coverage - What important information is missing or incorrectly stated?
2. Structure and organization - How does the structure compare to the expected summary?
3. Style and tone - Is the tone appropriate for financial analysis?
4. Specific improvements - What concrete changes would make the generated summary better?

Generated Summary:
{generated_summary}

Expected Summary:
{expected_summary}

Provide your evaluation in JSON format with these keys:
- strengths: List of strengths in the generated summary
- weaknesses: List of weaknesses or missing elements
- improvement_suggestions: Specific suggestions for improvement
- overall_assessment: Brief overall assessment
"""

        messages = [
            {"role": "system", "content": "You are an expert evaluator of financial news summaries."},
            {"role": "user", "content": prompt}
        ]
        
        # Generate feedback
        feedback_text = self.model_interface.generate(messages)
        
        # Extract JSON from the response
        try:
            # Find JSON block in the response
            json_match = re.search(r'\{[\s\S]*\}', feedback_text)
            if json_match:
                feedback_json = json.loads(json_match.group(0))
            else:
                # If no JSON block found, try to parse the entire response
                feedback_json = json.loads(feedback_text)
        except json.JSONDecodeError:
            # If parsing fails, create a structured response from the text
            feedback_json = {
                "strengths": ["Unable to parse structured feedback"],
                "weaknesses": ["Unable to parse structured feedback"],
                "improvement_suggestions": ["Unable to parse structured feedback"],
                "overall_assessment": feedback_text[:200] + "..."
            }
        
        return feedback_json
    
    def generate_prompt_improvement(self, transformation_prompt: str, evaluations: List[Dict[str, Any]]) -> str:
        """
        Generate an improved transformation prompt based on evaluations.
        
        Args:
            transformation_prompt: Current transformation prompt
            evaluations: List of evaluation results
            
        Returns:
            Improved transformation prompt
        """
        # Aggregate feedback from evaluations
        strengths = []
        weaknesses = []
        suggestions = []
        
        for eval_result in evaluations:
            if "feedback" in eval_result:
                feedback = eval_result["feedback"]
                strengths.extend(feedback.get("strengths", []))
                weaknesses.extend(feedback.get("weaknesses", []))
                suggestions.extend(feedback.get("improvement_suggestions", []))
        
        # Deduplicate feedback
        strengths = list(set(strengths))[:5]  # Limit to top 5
        weaknesses = list(set(weaknesses))[:5]
        suggestions = list(set(suggestions))[:5]
        
        # Create prompt for improvement
        improvement_prompt = f"""You are an expert at creating effective prompts for language models. I have a transformation prompt that needs improvement based on evaluation feedback.

Current Transformation Prompt:
{transformation_prompt}

Evaluation Feedback:
Strengths:
{chr(10).join(f"- {s}" for s in strengths)}

Weaknesses:
{chr(10).join(f"- {w}" for w in weaknesses)}

Improvement Suggestions:
{chr(10).join(f"- {s}" for s in suggestions)}

Please create an improved version of the transformation prompt that:
1. Maintains the overall structure and examples
2. Addresses the weaknesses identified in the evaluation
3. Incorporates the improvement suggestions
4. Enhances clarity and specificity of instructions

Return only the improved transformation prompt without any additional explanation.
"""

        messages = [
            {"role": "system", "content": "You are an expert at creating effective prompts for language models."},
            {"role": "user", "content": improvement_prompt}
        ]
        
        # Generate improved prompt
        improved_prompt = self.model_interface.generate(messages)
        
        logger.info("Generated improved transformation prompt")
        
        return improved_prompt
    
    def evaluate_batch(self, test_examples: List[Dict[str, Any]], generated_summaries: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of generated summaries.
        
        Args:
            test_examples: List of test examples
            generated_summaries: List of generated summaries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, (example, summary) in enumerate(zip(test_examples, generated_summaries)):
            expected_summary = example["expected_output"]
            
            # Evaluate summary
            evaluation = self.evaluate_summary(summary, expected_summary)
            
            # Store result
            results.append({
                "folder": example.get("folder", f"example_{i}"),
                "metrics": evaluation["metrics"],
                "feedback": evaluation["feedback"]
            })
            
            logger.info(f"Evaluated summary for {example.get('folder', f'example_{i}')}: "
                       f"score={evaluation['metrics']['overall_score']:.2f}")
        
        return results
