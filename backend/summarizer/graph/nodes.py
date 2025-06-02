# graph/nodes.py
"""Individual node implementations for the LangGraph workflow."""

from typing import Dict, Any, List
import logging
from ..core.data_loader import TrainingSet
from ..core.prompt_builder import PromptBuilder
from ..core.evaluator import SummaryEvaluator
from ..models.base import BaseLLMProvider


class SummarizerNode:
    """Financial analyst summarizer agent node."""

    def __init__(self, llm_provider: BaseLLMProvider, prompt_builder: PromptBuilder):
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary based on current prompt template."""
        training_set = state['current_training_set']

        # Build prompt with current template
        prompt = self.prompt_builder.build_prompt(training_set.articles)

        # Generate summary
        try:
            generated_summary = self.llm_provider.generate(prompt)

            state.update({
                'generated_summary': generated_summary,
                'current_prompt': prompt,
                'summarizer_success': True
            })

            self.logger.info(f"Generated summary for {training_set.date}")

        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            state.update({
                'generated_summary': '',
                'summarizer_success': False,
                'error': str(e)
            })

        return state


class EvaluatorNode:
    """Editor/evaluator agent node."""

    def __init__(self, evaluator: SummaryEvaluator):
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generated summary and provide feedback."""
        if not state.get('summarizer_success', False):
            return state

        training_set = state['current_training_set']
        generated_summary = state['generated_summary']

        try:
            # Evaluate the summary
            evaluation = self.evaluator.evaluate_summary(
                generated_summary,
                training_set.expected_summary,
                training_set.articles
            )

            state.update({
                'evaluation_score': evaluation,
                'feedback': evaluation.detailed_feedback,
                'evaluator_success': True
            })

            self.logger.info(
                f"Evaluated summary for {training_set.date}, score: {evaluation.overall_score:.3f}")

        except Exception as e:
            self.logger.error(f"Error evaluating summary: {e}")
            state.update({
                'evaluator_success': False,
                'error': str(e)
            })

        return state


class RefinementNode:
    """Prompt refinement node."""

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Refine prompt based on evaluation feedback."""
        if not state.get('evaluator_success', False):
            return state

        feedback = state['feedback']
        evaluation_score = state['evaluation_score']

        # Only refine if score is below threshold
        if evaluation_score.overall_score >= 0.8:
            state['needs_refinement'] = False
            self.logger.info("Score acceptable, no refinement needed")
        else:
            # Refine the prompt
            self.prompt_builder.refine_prompt(feedback)
            state['needs_refinement'] = True
            state['prompt_version'] = self.prompt_builder.current_template.version
            self.logger.info(
                f"Refined prompt to version {state['prompt_version']}")

        return state


class MetaLearningNode:
    """Meta-learning node for cross-set learning."""

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply meta-learning across all training sets."""
        all_results = state.get('all_set_results', [])

        if not all_results:
            return state

        # Analyze patterns across all sets
        avg_scores = {}
        common_issues = {}

        for result in all_results:
            evaluation = result.get('evaluation_score')
            if not evaluation:
                continue

            # Collect scores
            for metric in ['content_coverage', 'accuracy', 'structure_score', 'style_consistency']:
                if metric not in avg_scores:
                    avg_scores[metric] = []
                avg_scores[metric].append(getattr(evaluation, metric))

            # Collect common feedback patterns
            feedback = result.get('feedback', {})
            for key, value in feedback.items():
                if key not in common_issues:
                    common_issues[key] = 0
                if value:
                    common_issues[key] += 1

        # Calculate average scores
        for metric in avg_scores:
            avg_scores[metric] = sum(
                avg_scores[metric]) / len(avg_scores[metric])

        # Apply meta-learning refinements
        meta_feedback = self._generate_meta_feedback(
            avg_scores, common_issues, len(all_results))

        if meta_feedback:
            self.prompt_builder.refine_prompt(meta_feedback)
            self.logger.info("Applied meta-learning refinements")

        state.update({
            'meta_learning_complete': True,
            'average_scores': avg_scores,
            'meta_feedback': meta_feedback
        })

        return state

    def _generate_meta_feedback(self, avg_scores: Dict[str, float],
                                common_issues: Dict[str, int],
                                total_sets: int) -> Dict[str, Any]:
        """Generate meta-feedback based on patterns across all sets."""
        meta_feedback = {}
        threshold = total_sets * 0.6  # Issues occurring in 60%+ of sets

        # Identify persistent issues
        if common_issues.get('missing_context', 0) >= threshold:
            meta_feedback['missing_context'] = True

        if common_issues.get('tone_issues', 0) >= threshold:
            meta_feedback['tone_issues'] = True

        if common_issues.get('accuracy_issues', 0) >= threshold:
            meta_feedback['accuracy_issues'] = True

        # Section-specific meta-feedback
        for section in ['us', 'europe', 'asia']:
            section_issues = 0
            feedback_key = f'{section}_feedback'

            if common_issues.get(feedback_key, 0) >= threshold:
                section_issues += 1

            if section_issues > 0:
                meta_feedback[feedback_key] = {
                    'persistent_issues': True,
                    'needs_stronger_guidance': True
                }

        # Format issues
        format_issue_count = sum(1 for key in common_issues.keys()
                                 if 'format' in key and common_issues[key] >= threshold)

        if format_issue_count > 0:
            meta_feedback['format_issues'] = ['structure', 'bullet_count']

        return meta_feedback
