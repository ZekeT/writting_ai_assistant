# Advanced News Summarizer

A modular, advanced news summarization system that uses few-shot learning with LangGraph to generate high-quality summaries from multiple news articles.

## Features

- **Modular Architecture**: Clean separation of concerns for maintainability and extensibility
- **Evaluator Agent**: Provides feedback and suggestions to improve summarization quality
- **Context Window Management**: Handles large sets of articles through chunking and hierarchical summarization
- **Dynamic Example Selection**: Intelligently selects the most relevant few-shot examples for each new set of articles
- **Progressive Training**: Iteratively improves the transformation prompt based on evaluation feedback
- **Multiple Provider Support**: Works with both OpenAI (GPT-4) and Google Gemini models
- **Comprehensive Evaluation**: Includes metrics and detailed feedback for summary quality assessment

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd advanced_summarizer

# Install dependencies
pip install langchain langchain-core langchain-openai langchain-google-genai langgraph scikit-learn numpy
```

## Directory Structure

```
advanced_summarizer/
├── core/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and management
│   ├── prompt_builder.py     # Transformation prompt construction
│   └── progressive_training.py # Progressive training and dynamic example selection
├── models/
│   ├── __init__.py
│   └── base.py               # Model interfaces for different providers
├── graph/
│   ├── __init__.py
│   └── workflow.py           # LangGraph workflow definition
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py          # Summary evaluation and feedback
├── utils/
│   ├── __init__.py
│   └── context_window.py     # Context window management strategies
└── main.py                   # Main entry point and CLI
```

## Training Data Structure

Your training data should be organized as follows:

```
training_data/
├── example1/
│   ├── train_article1.md
│   ├── train_article2.md
│   ├── ...
│   └── test_expected_summary.md
├── example2/
│   ├── train_article1.md
│   ├── train_article2.md
│   ├── ...
│   └── test_expected_summary.md
└── ...
```

- Each subfolder contains a set of training articles (prefixed with `train_`) and an expected summary (prefixed with `test_`).
- The module will learn from these examples to generate summaries for new articles.

## Usage

### Basic Training

Train the model on your examples and generate a transformation prompt:

```bash
python -m advanced_summarizer.main --train \
    --train_dir /path/to/training_data \
    --prompt_output transformation_prompt.txt \
    --num_examples 3 \
    --model gpt-4-turbo \
    --provider openai
```

### Progressive Training

Train the model progressively with evaluation feedback:

```bash
python -m advanced_summarizer.main --progressive-train \
    --train_dir /path/to/training_data \
    --prompt_output transformation_prompt.txt \
    --max_iterations 3 \
    --model gpt-4-turbo \
    --provider openai
```

### Summarization

Apply the learned transformation to new articles:

```bash
python -m advanced_summarizer.main --summarize \
    --prompt_input transformation_prompt.txt \
    --articles_dir /path/to/new_articles \
    --output_file summary_digest.md \
    --model gpt-4-turbo \
    --provider openai
```

### Dynamic Example Selection

Use dynamic example selection for better results:

```bash
python -m advanced_summarizer.main --summarize \
    --train_dir /path/to/training_data \
    --articles_dir /path/to/new_articles \
    --output_file summary_digest.md \
    --dynamic_examples \
    --model gpt-4-turbo \
    --provider openai
```

### Evaluation

Evaluate the transformation prompt on test examples:

```bash
python -m advanced_summarizer.main --evaluate \
    --train_dir /path/to/training_data \
    --prompt_input transformation_prompt.txt \
    --model gpt-4-turbo \
    --provider openai
```

## API Usage

You can also use the module programmatically in your Python code:

```python
from advanced_summarizer.main import AdvancedSummarizer

# Initialize the summarizer
summarizer = AdvancedSummarizer(
    model_name="gpt-4-turbo",
    provider="openai",
    temperature=0.2
)

# Load training data
summarizer.load_training_data("/path/to/training_data")

# Option 1: Basic training
summarizer.build_transformation_prompt(num_examples=3)
summarizer.save_transformation_prompt("transformation_prompt.txt")

# Option 2: Progressive training
results = summarizer.train_progressively(max_iterations=3)
summarizer.save_transformation_prompt("improved_prompt.txt")

# Summarize new articles
articles = summarizer.load_new_articles("/path/to/new_articles")
summary = summarizer.summarize(articles, use_dynamic_examples=True)
summarizer.save_summary(summary, "summary_digest.md")

# Evaluate performance
evaluation = summarizer.evaluate_on_test_examples()
print(f"Average score: {evaluation['average_score']:.3f}")
```

## Environment Variables

Set these environment variables for API access:

- For OpenAI: `OPENAI_API_KEY`
- For Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Advanced Configuration

The module supports various configuration options:

- `--model`: Model name (e.g., "gpt-4-turbo", "gemini-pro")
- `--provider`: LLM provider ("openai" or "google")
- `--temperature`: Temperature for generation (default: 0.2)
- `--max_tokens`: Maximum tokens for generation (default: 4000)
- `--num_examples`: Number of examples to include in prompt (default: 3)
- `--max_iterations`: Maximum iterations for progressive training (default: 3)

## Integration with Writing Assistant

To integrate this module with your writing assistant app:

1. Copy the `advanced_summarizer` directory to your project
2. Import and use the `AdvancedSummarizer` class in your application
3. Call the appropriate methods based on your workflow

Example integration:

```python
from advanced_summarizer.main import AdvancedSummarizer

def generate_article_digest(articles_folder, prompt_file=None, use_dynamic=True):
    summarizer = AdvancedSummarizer()
    
    if not use_dynamic and prompt_file:
        summarizer.load_transformation_prompt(prompt_file)
    elif use_dynamic:
        summarizer.load_training_data("/path/to/training_data")
    
    articles = summarizer.load_new_articles(articles_folder)
    return summarizer.summarize(articles, use_dynamic_examples=use_dynamic)
```

## Architecture Overview

The advanced summarizer is built with a modular architecture:

1. **Core Module**:
   - `data_loader.py`: Handles loading and processing of training data and articles
   - `prompt_builder.py`: Constructs transformation prompts from examples
   - `progressive_training.py`: Implements progressive training and dynamic example selection

2. **Models Module**:
   - `base.py`: Provides interfaces for different LLM providers (OpenAI, Google)

3. **Graph Module**:
   - `workflow.py`: Defines the LangGraph workflow for summarization

4. **Evaluation Module**:
   - `evaluator.py`: Implements summary evaluation and feedback generation

5. **Utils Module**:
   - `context_window.py`: Manages context window limitations through chunking and selection strategies

6. **Main Module**:
   - `main.py`: Integrates all components and provides CLI and API interfaces

## Key Workflows

### Basic Summarization Workflow:
1. Load articles
2. Format articles for the prompt
3. Generate summary using the LLM
4. Return the summary

### Hierarchical Summarization Workflow:
1. Split articles into chunks
2. Summarize each chunk individually
3. Combine the chunk summaries into a final summary

### Progressive Training Workflow:
1. Evaluate current prompt on validation examples
2. Generate improved prompt based on evaluation feedback
3. Evaluate the improved prompt
4. If significant improvement, use the improved prompt
5. Repeat for specified number of iterations

### Dynamic Example Selection Workflow:
1. Calculate similarity between new articles and training examples
2. Select the most relevant examples
3. Build a custom prompt using the selected examples
4. Use this prompt for summarization
