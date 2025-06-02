# graph/workflow.py
"""Graph construction and execution for the learning workflow."""

from typing import Dict, Any, List, Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .nodes import SummarizerNode, EvaluatorNode, RefinementNode, MetaLearningNode
from ..core.data_loader import DataLoader, TrainingSet
from ..core.prompt_builder import PromptBuilder
from ..core.evaluator import SummaryEvaluator
from ..models.base import BaseLLMProvider


class LearningWorkflow:
    """Main workflow orchestrator for the learning process."""

    def __init__(self, llm_provider: BaseLLMProvider, data_dir: str = "data", max_iterations: int = 5):
        self.llm_provider = llm_provider
        self.data_loader = DataLoader(data_dir)
        self.prompt_builder = PromptBuilder()
        self.evaluator = SummaryEvaluator()
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)

        # Initialize nodes
        self.summarizer_node = SummarizerNode(
            llm_provider, self.prompt_builder)
        self.evaluator_node = EvaluatorNode(self.evaluator)
        self.refinement_node = RefinementNode(self.prompt_builder)
        self.meta_learning_node = MetaLearningNode(self.prompt_builder)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Define state schema
        def state_schema(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "current_training_set": state.get("current_training_set"),
                "generated_summary": state.get("generated_summary", ""),
                "current_prompt": state.get("current_prompt", ""),
                "evaluation_score": state.get("evaluation_score"),
                "feedback": state.get("feedback", {}),
                "iteration_count": state.get("iteration_count", 0),
                "needs_refinement": state.get("needs_refinement", True),
                "prompt_version": state.get("prompt_version", 1),
                "set_complete": state.get("set_complete", False),
                "all_set_results": state.get("all_set_results", []),
                "meta_learning_complete": state.get("meta_learning_complete", False),
                "summarizer_success": state.get("summarizer_success", False),
                "evaluator_success": state.get("evaluator_success", False),
                "error": state.get("error"),
                "average_scores": state.get("average_scores", {}),
                "meta_feedback": state.get("meta_feedback", {})
            }

        # Create graph
        workflow = StateGraph(state_schema)

        # Add nodes
        workflow.add_node("summarizer", self.summarizer_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("refinement", self.refinement_node)
        workflow.add_node("meta_learning", self.meta_learning_node)

        # Define conditional logic
        def should_continue_iteration(state: Dict[str, Any]) -> str:
            """Decide whether to continue iterating or move to next step."""
            if not state.get('evaluator_success', False):
                return END

            if state['iteration_count'] >= self.max_iterations:
                return "complete_set"

            if not state.get('needs_refinement', True):
                return "complete_set"

            return "summarizer"

        def should_apply_meta_learning(state: Dict[str, Any]) -> str:
            """Decide whether to apply meta-learning."""
            all_results = state.get('all_set_results', [])
            if len(all_results) >= 3:  # Apply meta-learning after 3+ sets
                return "meta_learning"
            return END

        # Add edges
        workflow.set_entry_point("summarizer")
        workflow.add_edge("summarizer", "evaluator")
        workflow.add_edge("evaluator", "refinement")
        workflow.add_conditional_edges(
            "refinement",
            should_continue_iteration,
            {
                "summarizer": "summarizer",
                "complete_set": "check_meta_learning",
                END: END
            }
        )

        # Add meta-learning logic
        workflow.add_node("check_meta_learning", lambda state: state)
        workflow.add_conditional_edges(
            "check_meta_learning",
            should_apply_meta_learning,
            {
                "meta_learning": "meta_learning",
                END: END
            }
        )
        workflow.add_edge("meta_learning", END)

        return workflow.compile(checkpointer=MemorySaver())

    def run_learning_process(self) -> Dict[str, Any]:
        """Run the complete learning process across all training sets."""
        # Load all training sets
        training_sets = self.data_loader.load_all_training_sets()

        if not training_sets:
            raise ValueError("No training sets found in data directory")

        self.logger.info(f"Loaded {len(training_sets)} training sets")

        all_results = []

        # Process each training set
        for i, training_set in enumerate(training_sets):
            self.logger.info(
                f"Processing training set {i+1}/{len(training_sets)}: {training_set.date}")

            # Run single set learning
            result = self._run_single_set_learning(training_set, i)
            all_results.append(result)

            # Apply incremental meta-learning every few sets
            if (i + 1) % 3 == 0:
                meta_state = {
                    'all_set_results': all_results,
                    'meta_learning_complete': False
                }
                meta_result = self.graph.invoke(
                    meta_state, {"configurable": {"thread_id": f"meta_{i}"}})
                self.logger.info(f"Applied meta-learning after {i+1} sets")

        # Final meta-learning pass
        final_meta_state = {
            'all_set_results': all_results,
            'meta_learning_complete': False
        }
        final_meta_result = self.graph.invoke(
            final_meta_state, {"configurable": {"thread_id": "final_meta"}})

        # Compile final results
        final_results = {
            'individual_results': all_results,
            'meta_learning_result': final_meta_result,
            'final_prompt_template': self.prompt_builder.get_best_prompt_template(),
            'average_scores': self._calculate_final_averages(all_results),
            'learning_summary': self._generate_learning_summary(all_results)
        }

        self.logger.info("Learning process complete")
        return final_results

    def _run_single_set_learning(self, training_set: TrainingSet, set_index: int) -> Dict[str, Any]:
        """Run learning process for a single training set."""
        initial_state = {
            'current_training_set': training_set,
            'iteration_count': 0,
            'set_complete': False
        }

        # Execute graph for this training set
        thread_id = f"set_{set_index}_{training_set.date}"
        result = self.graph.invoke(
            initial_state, {"configurable": {"thread_id": thread_id}})

        # Update iteration count
        def increment_iteration(state):
            state['iteration_count'] += 1
            return state

        # Run iterations
        for iteration in range(self.max_iterations):
            if result.get('set_complete', False) or not result.get('needs_refinement', True):
                break

            result['iteration_count'] = iteration + 1
            result = self.graph.invoke(
                result, {"configurable": {"thread_id": thread_id}})

        return result

    def _calculate_final_averages(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average scores across all training sets."""
        metrics = ['content_coverage', 'accuracy',
                   'structure_score', 'style_consistency', 'overall_score']
        averages = {}

        for metric in metrics:
            scores = []
            for result in all_results:
                evaluation = result.get('evaluation_score')
                if evaluation:
                    scores.append(getattr(evaluation, metric))

            if scores:
                averages[metric] = sum(scores) / len(scores)
            else:
                averages[metric] = 0.0

        return averages

    def _generate_learning_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the learning process."""
        total_iterations = sum(result.get('iteration_count', 0)
                               for result in all_results)
        successful_sets = sum(
            1 for result in all_results if result.get('evaluator_success', False))

        final_scores = self._calculate_final_averages(all_results)

        return {
            'total_training_sets': len(all_results),
            'successful_sets': successful_sets,
            'total_iterations': total_iterations,
            'average_iterations_per_set': total_iterations / len(all_results) if all_results else 0,
            'final_prompt_version': self.prompt_builder.current_template.version,
            'final_average_scores': final_scores,
            'improvement_achieved': final_scores.get('overall_score', 0) > 0.7
        }

    def get_production_prompt(self) -> str:
        """Export the learned prompt for production use."""
        return self.prompt_builder.export_prompt_for_production()
