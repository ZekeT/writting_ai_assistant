# core/data_loader.py
"""Data loading and processing functionality."""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class NewsArticle:
    """Represents a single news article."""
    title: str
    content: str
    filename: str


@dataclass
class ExpectedSummary:
    """Represents an expected summary with structured format."""
    us_points: List[str]
    europe_points: List[str]
    asia_points: List[str]
    raw_content: str
    filename: str


@dataclass
class TrainingSet:
    """Represents a complete training set with articles and expected summary."""
    date: str
    articles: List[NewsArticle]
    expected_summary: ExpectedSummary


class DataLoader:
    """Handles loading and processing of training data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_all_training_sets(self) -> List[TrainingSet]:
        """Load all training sets from the data directory."""
        training_sets = []

        # Find all date directories
        date_dirs = [d for d in self.data_dir.iterdir()
                     if d.is_dir() and d.name.startswith('dww_')]

        for date_dir in sorted(date_dirs):
            try:
                training_set = self._load_training_set(date_dir)
                training_sets.append(training_set)
                self.logger.info(
                    f"Loaded training set for {training_set.date}")
            except Exception as e:
                self.logger.error(f"Error loading {date_dir}: {e}")

        return training_sets

    def _load_training_set(self, date_dir: Path) -> TrainingSet:
        """Load a single training set from a date directory."""
        date = date_dir.name.replace('dww_', '')

        # Load articles
        articles = []
        article_files = [f for f in date_dir.glob('in_*.md')]

        for article_file in sorted(article_files):
            article = self._load_article(article_file)
            articles.append(article)

        # Load expected summary
        summary_files = list(date_dir.glob('out_*.md'))
        if not summary_files:
            raise ValueError(f"No expected summary found in {date_dir}")

        expected_summary = self._load_expected_summary(summary_files[0])

        return TrainingSet(
            date=date,
            articles=articles,
            expected_summary=expected_summary
        )

    def _load_article(self, file_path: Path) -> NewsArticle:
        """Load a single news article from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title (first line after # if present)
        lines = content.split('\n')
        title = ""
        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip()[2:]
                break

        if not title:
            title = file_path.stem

        return NewsArticle(
            title=title,
            content=content,
            filename=file_path.name
        )

    def _load_expected_summary(self, file_path: Path) -> ExpectedSummary:
        """Load and parse expected summary from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse structured summary
        us_points = self._extract_section_points(content, "US")
        europe_points = self._extract_section_points(content, "Europe")
        asia_points = self._extract_section_points(content, "Asia")

        return ExpectedSummary(
            us_points=us_points,
            europe_points=europe_points,
            asia_points=asia_points,
            raw_content=content,
            filename=file_path.name
        )

    def _extract_section_points(self, content: str, section: str) -> List[str]:
        """Extract bullet points from a specific section."""
        pattern = rf"## {section}\s*\n(.*?)(?=\n## |\n#|\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return []

        section_content = match.group(1).strip()

        # Extract bullet points
        points = []
        for line in section_content.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                points.append(line[2:].strip())

        return points
