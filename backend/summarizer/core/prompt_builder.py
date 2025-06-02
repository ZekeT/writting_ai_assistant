# core/prompt_builder.py
"""Logic for constructing and refining transformation prompts."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import logging


@dataclass
class PromptTemplate:
    """Represents a summarization prompt template."""
    base_instruction: str
    us_guidance: str
    europe_guidance: str
    asia_guidance: str
    formatting_rules: str
    examples: List[str]
    version: int = 1


class PromptBuilder:
    """Handles construction and refinement of summarization prompts."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_template = self._get_initial_template()
        self.refinement_history = []

    def _get_initial_template(self) -> PromptTemplate:
        """Get the initial base prompt template."""
        return PromptTemplate(
            base_instruction="""You are a financial analyst tasked with creating a structured market wrap summary from multiple news articles. 
            Analyze the provided news articles and create a summary following the exact format specified.""",

            us_guidance="""For the US section, focus on:
            - Major stock market movements (S&P 500, Nasdaq, Dow Jones)
            - Federal Reserve policy and interest rate news
            - Major corporate earnings and announcements
            - Economic indicators and data releases
            - Significant sector-specific developments""",

            europe_guidance="""For the Europe section, focus on:
            - European stock market performance (FTSE, DAX, CAC)
            - ECB policy decisions and monetary policy
            - Major European corporate news
            - Economic data from major European economies
            - Brexit-related developments if relevant""",

            asia_guidance="""For the Asia section, focus on:
            - Asian market performance (Nikkei, Hang Seng, Shanghai Composite)
            - Central bank policies in major Asian economies
            - Major corporate news from Asia
            - Trade-related developments
            - Economic indicators from key Asian markets""",

            formatting_rules="""
            CRITICAL FORMATTING REQUIREMENTS:
            1. Use exactly this structure:
               # Market Wrap
               ## US
               [5-7 bullet points]
               ## Europe
               [3-5 bullet points]
               ## Asia
               [3-5 bullet points]
            
            2. Each bullet point should be concise but informative
            3. Focus on after-market-close events and their implications
            4. Maintain professional, objective tone
            5. Include specific numbers, percentages, and company names when relevant""",

            examples=[]
        )

    def build_prompt(self, articles: List[Any]) -> str:
        """Build the complete summarization prompt with articles."""
        template = self.current_template

        # Combine articles text
        articles_text = "\n\n".join([
            f"**Article {i+1}: {article.title}**\n{article.content}"
            for i, article in enumerate(articles)
        ])

        prompt = f"""{template.base_instruction}

{template.us_guidance}

{template.europe_guidance}

{template.asia_guidance}

{template.formatting_rules}

NEWS ARTICLES TO SUMMARIZE:
{articles_text}

Please provide the market wrap summary following the exact format specified above."""

        return prompt

    def refine_prompt(self, feedback: Dict[str, Any]) -> None:
        """Refine the current prompt based on evaluator feedback."""
        template = self.current_template

        # Store current version in history
        self.refinement_history.append({
            'version': template.version,
            'template': template,
            'feedback': feedback
        })

        # Create refined template
        refined_template = PromptTemplate(
            base_instruction=self._refine_base_instruction(
                template.base_instruction, feedback),
            us_guidance=self._refine_section_guidance(
                template.us_guidance, feedback.get('us_feedback', {})),
            europe_guidance=self._refine_section_guidance(
                template.europe_guidance, feedback.get('europe_feedback', {})),
            asia_guidance=self._refine_section_guidance(
                template.asia_guidance, feedback.get('asia_feedback', {})),
            formatting_rules=self._refine_formatting_rules(
                template.formatting_rules, feedback),
            examples=template.examples.copy(),
            version=template.version + 1
        )

        self.current_template = refined_template
        self.logger.info(
            f"Refined prompt to version {refined_template.version}")

    def _refine_base_instruction(self, current: str, feedback: Dict[str, Any]) -> str:
        """Refine base instruction based on feedback."""
        improvements = []

        if feedback.get('missing_context'):
            improvements.append(
                "Pay special attention to market context and implications of events.")

        if feedback.get('tone_issues'):
            improvements.append(
                "Maintain a professional, analytical tone throughout.")

        if feedback.get('accuracy_issues'):
            improvements.append(
                "Ensure all facts and figures are accurately represented from the source articles.")

        if improvements:
            return current + "\n\nADDITIONAL FOCUS AREAS:\n" + "\n".join(f"- {imp}" for imp in improvements)

        return current

    def _refine_section_guidance(self, current: str, section_feedback: Dict[str, Any]) -> str:
        """Refine section-specific guidance based on feedback."""
        if not section_feedback:
            return current

        improvements = []

        if section_feedback.get('missing_topics'):
            topics = section_feedback['missing_topics']
            improvements.append(f"Ensure coverage of: {', '.join(topics)}")

        if section_feedback.get('insufficient_detail'):
            improvements.append(
                "Provide more specific details and quantitative information")

        if section_feedback.get('irrelevant_content'):
            improvements.append(
                "Focus only on the most market-relevant information")

        if improvements:
            return current + "\n\nIMPROVEMENTS:\n" + "\n".join(f"- {imp}" for imp in improvements)

        return current

    def _refine_formatting_rules(self, current: str, feedback: Dict[str, Any]) -> str:
        """Refine formatting rules based on feedback."""
        if feedback.get('format_issues'):
            format_additions = []

            if 'bullet_count' in feedback['format_issues']:
                format_additions.append(
                    "Strictly adhere to bullet point count requirements (US: 5-7, Europe: 3-5, Asia: 3-5)")

            if 'structure' in feedback['format_issues']:
                format_additions.append(
                    "Maintain exact heading structure and markdown formatting")

            if format_additions:
                return current + "\n\nCRITICAL FORMAT FIXES:\n" + "\n".join(f"- {add}" for add in format_additions)

        return current

    def get_best_prompt_template(self) -> PromptTemplate:
        """Get the current best prompt template."""
        return self.current_template

    def export_prompt_for_production(self) -> str:
        """Export the final prompt template for production use."""
        template = self.current_template

        return f"""# Financial News Summarization Prompt (Version {template.version})

## Base Instruction
{template.base_instruction}

## US Market Guidance
{template.us_guidance}

## Europe Market Guidance
{template.europe_guidance}

## Asia Market Guidance
{template.asia_guidance}

## Formatting Rules
{template.formatting_rules}

## Usage
To use this prompt, replace [NEWS_ARTICLES] with the actual news articles to be summarized.

```
[BASE_INSTRUCTION]
[SECTION_GUIDANCE]
[FORMATTING_RULES]

NEWS ARTICLES TO SUMMARIZE:
[NEWS_ARTICLES]

Please provide the market wrap summary following the exact format specified above.
```
"""
