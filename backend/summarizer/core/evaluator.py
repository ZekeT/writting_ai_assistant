# core/evaluator.py
"""Evaluation metrics and testing functions."""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvaluationScore:
    """Represents evaluation scores for a summary."""
    content_coverage: float  # 0-1
    accuracy: float         # 0-1
    structure_score: float  # 0-1
    style_consistency: float  # 0-1
    overall_score: float    # 0-1
    detailed_feedback: Dict[str, Any]


class SummaryEvaluator:
    """Evaluates generated summaries against expected outputs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def evaluate_summary(self, generated: str, expected: Any, source_articles: List[Any]) -> EvaluationScore:
        """Comprehensive evaluation of generated summary."""

        # Parse both summaries
        gen_sections = self._parse_summary_sections(generated)
        exp_sections = {
            'us': expected.us_points,
            'europe': expected.europe_points,
            'asia': expected.asia_points
        }

        # Calculate individual scores
        content_score = self._evaluate_content_coverage(
            gen_sections, exp_sections)
        accuracy_score = self._evaluate_accuracy(generated, source_articles)
        structure_score = self._evaluate_structure(
            generated, expected.raw_content)
        style_score = self._evaluate_style_consistency(
            generated, expected.raw_content)

        # Calculate overall score
        overall_score = (
            content_score * 0.3 +
            accuracy_score * 0.25 +
            structure_score * 0.25 +
            style_score * 0.2
        )

        # Generate detailed feedback
        feedback = self._generate_detailed_feedback(
            gen_sections, exp_sections, generated, expected, source_articles
        )

        return EvaluationScore(
            content_coverage=float(content_score),
            accuracy=float(accuracy_score),
            structure_score=float(structure_score),
            style_consistency=float(style_score),
            overall_score=float(overall_score),
            detailed_feedback=feedback
        )

    def _parse_summary_sections(self, summary: str) -> Dict[str, List[str]]:
        """Parse summary into sections and bullet points."""
        sections = {'us': [], 'europe': [], 'asia': []}

        current_section = None
        lines = summary.split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith('## US'):
                current_section = 'us'
            elif line.startswith('## Europe'):
                current_section = 'europe'
            elif line.startswith('## Asia'):
                current_section = 'asia'
            elif line.startswith('- ') or line.startswith('* '):
                if current_section:
                    sections[current_section].append(line[2:].strip())

        return sections

    def _evaluate_content_coverage(self, generated: Dict[str, List[str]],
                                   expected: Dict[str, List[str]]):
        """Evaluate how well the generated content covers expected topics."""
        section_scores = []

        for section in ['us', 'europe', 'asia']:
            gen_points = generated.get(section, [])
            exp_points = expected.get(section, [])

            if not exp_points:
                section_scores.append(1.0)
                continue

            if not gen_points:
                section_scores.append(0.0)
                continue

            # Calculate semantic similarity between point sets
            gen_text = ' '.join(gen_points)
            exp_text = ' '.join(exp_points)

            try:
                vectors = self.vectorizer.fit_transform([gen_text, exp_text])
                similarity = cosine_similarity(
                    vectors[0:1], vectors[1:2])[0][0]
                section_scores.append(max(0, similarity))
            except:
                # Fallback to simple keyword overlap
                gen_words = set(gen_text.lower().split())
                exp_words = set(exp_text.lower().split())
                overlap = len(gen_words & exp_words) / len(gen_words |
                                                           exp_words) if gen_words | exp_words else 0
                section_scores.append(overlap)

        return np.mean(section_scores)

    def _evaluate_accuracy(self, generated: str, source_articles: List[Any]) -> float:
        """Evaluate factual accuracy based on source articles."""
        # Extract numbers and company names from generated summary
        gen_numbers = re.findall(r'\b\d+\.?\d*%?\b', generated)
        gen_companies = re.findall(r'\b[A-Z][a-zA-Z&\s]+(?:Inc|Corp|Ltd|Co|Group|Bank|Motors|Electric|Systems|Technologies|Financial|Capital|Holdings|Partners|Associates|Solutions|Services|International|Global|Ventures|Investment|Management|Trust|Insurance|Energy|Oil|Gas|Pharma|Biotech|Media|Entertainment|Communications|Telecom|Retail|Foods|Beverages|Airlines|Airways|Rail|Shipping|Real Estate|REIT|ETF|Fund|Index|Exchange|Trading|Securities|Brokerage|Advisory|Consulting|Software|Technology|Tech|Data|Cloud|Cyber|Digital|Online|Platform|Network|Mobile|Wireless|Internet|Web|Social|Gaming|Streaming|Publishing|News|Broadcasting|Cable|Satellite|Utilities|Power|Water|Waste|Construction|Materials|Steel|Aluminum|Copper|Gold|Silver|Mining|Drilling|Exploration|Production|Refining|Chemical|Industrial|Manufacturing|Automotive|Aerospace|Defense|Healthcare|Medical|Devices|Equipment|Instruments|Laboratory|Research|Development|Pharmaceutical|Biotechnology|Genetic|Therapy|Diagnostic|Hospital|Clinic|Health|Wellness|Fitness|Sports|Apparel|Fashion|Luxury|Beauty|Personal Care|Consumer|Retail|Grocery|Supermarket|Department|Specialty|Discount|Warehouse|Club|Restaurant|Fast Food|Casual Dining|Hotel|Resort|Lodging|Travel|Tourism|Cruise|Casino|Gaming|Lottery|Education|University|College|School|Training|Publishing|Books|Magazines|Newspapers|Printing|Packaging|Paper|Forestry|Agriculture|Farming|Livestock|Fishing|Food Processing|Beverages|Alcohol|Tobacco|Textiles|Clothing|Footwear|Furniture|Home|Garden|Hardware|Tools|Appliances|Electronics|Computers|Semiconductors|Software|Internet|Telecommunications|Media|Entertainment|Games|Toys|Sports|Recreation|Fitness|Beauty|Personal Care)\\b', generated)

        # Check if key facts from articles appear in summary
        article_text = ' '.join(
            [article.content for article in source_articles])

        accuracy_indicators = 0
        total_checks = 0

        # Check number consistency
        for number in gen_numbers[:5]:  # Check first 5 numbers
            if number in article_text:
                accuracy_indicators += 1
            total_checks += 1

        # Check company name consistency
        for company in gen_companies[:5]:  # Check first 5 companies
            if company in article_text:
                accuracy_indicators += 1
            total_checks += 1

        if total_checks == 0:
            return 0.8  # Default moderate score if no specific facts to check

        return accuracy_indicators / total_checks

    def _evaluate_structure(self, generated: str, expected: str) -> float:
        """Evaluate structural compliance with expected format."""
        score = 0.0
        checks = 0

        # Check for main heading
        if "# Market Wrap" in generated:
            score += 1
        checks += 1

        # Check for section headings
        required_sections = ["## US", "## Europe", "## Asia"]
        for section in required_sections:
            if section in generated:
                score += 1
            checks += 1

        # Check bullet point counts
        gen_sections = self._parse_summary_sections(generated)

        # US should have 5-7 points
        us_count = len(gen_sections.get('us', []))
        if 5 <= us_count <= 7:
            score += 1
        checks += 1

        # Europe should have 3-5 points
        europe_count = len(gen_sections.get('europe', []))
        if 3 <= europe_count <= 5:
            score += 1
        checks += 1

        # Asia should have 3-5 points
        asia_count = len(gen_sections.get('asia', []))
        if 3 <= asia_count <= 5:
            score += 1
        checks += 1

        return score / checks if checks > 0 else 0.0

    def _evaluate_style_consistency(self, generated: str, expected: str) -> float:
        """Evaluate style consistency with expected output."""
        # Simple heuristics for style evaluation
        score = 0.0
        checks = 0

        # Check for professional tone (presence of financial terms)
        financial_terms = ['market', 'stock', 'trading', 'index', 'earnings', 'revenue', 'profit', 'loss', 'shares', 'dividend', 'yield', 'volatility', 'analyst', 'forecast', 'guidance', 'quarter',
                           'fiscal', 'economic', 'growth', 'inflation', 'rate', 'policy', 'federal', 'central bank', 'treasury', 'bond', 'equity', 'commodity', 'currency', 'exchange', 'sector', 'industry']

        gen_lower = generated.lower()
        financial_count = sum(
            1 for term in financial_terms if term in gen_lower)

        if financial_count >= 5:
            score += 1
        checks += 1

        # Check for appropriate length
        gen_length = len(generated.split())
        exp_length = len(expected.split())

        length_ratio = min(gen_length, exp_length) / \
            max(gen_length, exp_length)
        if length_ratio >= 0.7:  # Within 30% of expected length
            score += 1
        checks += 1

        # Check for bullet point formatting
        bullet_count = len(re.findall(r'^- ', generated, re.MULTILINE))
        if bullet_count >= 8:  # Should have at least 8 bullet points total
            score += 1
        checks += 1

        return score / checks if checks > 0 else 0.0

    def _generate_detailed_feedback(self, gen_sections: Dict[str, List[str]],
                                    exp_sections: Dict[str, List[str]],
                                    generated: str, expected: Any,
                                    source_articles: List[Any]) -> Dict[str, Any]:
        """Generate detailed feedback for prompt refinement."""
        feedback = {
            'us_feedback': {},
            'europe_feedback': {},
            'asia_feedback': {},
            'format_issues': [],
            'missing_context': False,
            'tone_issues': False,
            'accuracy_issues': False
        }

        # Section-specific feedback
        for section in ['us', 'europe', 'asia']:
            section_feedback = {}

            gen_points = gen_sections.get(section, [])
            exp_points = exp_sections.get(section, [])

            # Check coverage
            if len(gen_points) < len(exp_points):
                section_feedback['insufficient_detail'] = True

            # Check for missing topics (simplified)
            exp_topics = set()
            for point in exp_points:
                words = point.lower().split()
                exp_topics.update([w for w in words if len(w) > 3])

            gen_topics = set()
            for point in gen_points:
                words = point.lower().split()
                gen_topics.update([w for w in words if len(w) > 3])

            missing_topics = exp_topics - gen_topics
            if len(missing_topics) > len(exp_topics) * 0.3:  # More than 30% missing
                section_feedback['missing_topics'] = list(missing_topics)[:5]

            feedback[f'{section}_feedback'] = section_feedback

        # Format issues
        if not re.search(r'# Market Wrap', generated):
            feedback['format_issues'].append('missing_main_heading')

        us_count = len(gen_sections.get('us', []))
        if not (5 <= us_count <= 7):
            feedback['format_issues'].append('bullet_count')

        europe_count = len(gen_sections.get('europe', []))
        if not (3 <= europe_count <= 5):
            feedback['format_issues'].append('bullet_count')

        asia_count = len(gen_sections.get('asia', []))
        if not (3 <= asia_count <= 5):
            feedback['format_issues'].append('bullet_count')

        # Context and tone issues
        if len(generated.split()) < 100:
            feedback['missing_context'] = True

        if not any(term in generated.lower() for term in ['market', 'trading', 'stocks', 'index']):
            feedback['tone_issues'] = True

        return feedback
