# utils/file_io.py
"""File operations utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging


class FileManager:
    """Handles file I/O operations for the system."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)

    def save_json(self, data: Dict[str, Any], filename: str) -> None:
        """Save data as JSON file."""
        file_path = self.base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Saved JSON to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {e}")
            raise

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load data from JSON file."""
        file_path = self.base_dir / filename

        if not file_path.exists():
            self.logger.warning(f"JSON file not found: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded JSON from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def save_text(self, text: str, filename: str) -> None:
        """Save text to file."""
        file_path = self.base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Saved text to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving text to {file_path}: {e}")
            raise

    def load_text(self, filename: str) -> Optional[str]:
        """Load text from file."""
        file_path = self.base_dir / filename

        if not file_path.exists():
            self.logger.warning(f"Text file not found: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.logger.info(f"Loaded text from {file_path}")
            return text
        except Exception as e:
            self.logger.error(f"Error loading text from {file_path}: {e}")
            return None

    def save_model(self, model: Any, filename: str) -> None:
        """Save model using pickle."""
        file_path = self.base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved model to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model to {file_path}: {e}")
            raise

    def load_model(self, filename: str) -> Optional[Any]:
        """Load model using pickle."""
        file_path = self.base_dir / filename

        if not file_path.exists():
            self.logger.warning(f"Model file not found: {file_path}")
            return None

        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"Loaded model from {file_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {file_path}: {e}")
            return None

    def ensure_directory(self, directory: str) -> Path:
        """Ensure directory exists and return path."""
        dir_path = self.base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
