# Market Close Summary Generator - Usage Guide

This guide explains how to use and customize the provided code templates for generating market close summaries using few-shot learning.

## Overview

The provided scripts help you:
1. Load existing market articles and their corresponding summaries as few-shot examples
2. Create a structured prompt with these examples
3. Generate new summaries for fresh articles in the same style

## Setup and Installation

### For OpenAI Version
```bash
pip install openai langchain
export OPENAI_API_KEY=your_openai_api_key_here
```

### For Google Gemini Version
```bash
pip install google-generativeai langchain-google-genai
export GOOGLE_API_KEY=your_google_api_key_here
```

## File Structure

For the scripts to work with your data, organize your files as follows:

```
data/
├── articles/
│   ├── market_news_2025-05-20.txt
│   ├── market_news_2025-05-21.txt
│   └── ...
└── summaries/
    ├── market_summary_2025-05-20.txt
    ├── market_summary_2025-05-21.txt
    └── ...
```

The scripts will match articles and summaries by extracting dates from filenames.

## Customizing for Your Data

### 1. File Naming Pattern

By default, the scripts look for dates in the format `YYYY-MM-DD` in your filenames. If your files use a different naming convention, modify the `date_pattern` parameter:

```python
# Default pattern (YYYY-MM-DD)
date_pattern = r'(\d{4}-\d{2}-\d{2})'

# For MM-DD-YYYY format
date_pattern = r'(\d{2}-\d{2}-\d{4})'

# For filenames like "market_20250527.txt"
date_pattern = r'market_(\d{8})\.txt'
```

### 2. Adjusting the Prompt Format

You can customize how the few-shot examples are formatted in the prompt by modifying the `create_few_shot_prompt` method:

```python
def create_few_shot_prompt(self, num_examples: int = 3) -> str:
    # ...existing code...
    
    # Customize the introduction
    prompt = "As a financial analyst, create a market close summary following these examples:\n\n"
    
    for i, example in enumerate(examples_to_use, 1):
        # Customize how each example is presented
        prompt += f"EXAMPLE {i} ({example['date']}):\n"
        prompt += f"SOURCE ARTICLES:\n{example['article'][:800]}...\n\n"  # Show less of each article
        prompt += f"MARKET SUMMARY:\n{example['summary']}\n\n"
        prompt += "=" * 50 + "\n\n"  # Different separator
    
    # Customize the instruction for the new articles
    prompt += "Based on these examples, write a market close summary for today's articles:\n\n"
    return prompt
```

### 3. Changing the Number of Examples

You can adjust how many examples are included in each prompt:

```python
# Use more examples for complex summaries
summary = generator.generate_summary(new_articles, num_examples=5)

# Use fewer examples if your summaries are simple
summary = generator.generate_summary(new_articles, num_examples=2)
```

### 4. Model Selection

#### For OpenAI:
```python
# Use GPT-4 (default)
generator = MarketSummaryGenerator(model="gpt-4")

# Use GPT-3.5 Turbo for faster, cheaper results
generator = MarketSummaryGenerator(model="gpt-3.5-turbo")

# Use GPT-4 Turbo with more context
generator = MarketSummaryGenerator(model="gpt-4-turbo")
```

#### For Gemini:
```python
# Use Gemini Pro (default)
generator = MarketSummaryGeneratorGemini(model="gemini-pro")

# Use Gemini Ultra for more advanced capabilities
generator = MarketSummaryGeneratorGemini(model="gemini-ultra")
```

### 5. Handling Long Articles

If your articles are very long, you might need to truncate or summarize them first:

```python
def preprocess_article(article_text, max_length=4000):
    """Truncate or summarize long articles"""
    if len(article_text) <= max_length:
        return article_text
    
    # Simple truncation
    return article_text[:max_length] + "..."
    
    # Or use a summarization approach instead
    # return summarize_text(article_text)  # Implement your own summarization
```

Then use this in your workflow:
```python
with open(args.articles, 'r', encoding='utf-8') as f:
    new_articles = f.read()

new_articles = preprocess_article(new_articles)
summary = generator.generate_summary(new_articles, args.date)
```

## Command Line Usage Examples

### Creating Examples Database

```bash
# OpenAI version
python market_summary_few_shot.py create \
  --articles-dir data/articles \
  --summaries-dir data/summaries \
  --output examples.json

# With custom date pattern
python market_summary_few_shot.py create \
  --articles-dir data/articles \
  --summaries-dir data/summaries \
  --output examples.json \
  --date-pattern "market_(\d{8})\.txt"
```

### Generating New Summaries

```bash
# OpenAI version
python market_summary_few_shot.py generate \
  --examples examples.json \
  --articles data/new_articles/market_news_2025-05-27.txt \
  --output data/new_summaries/market_summary_2025-05-27.txt \
  --date 2025-05-27 \
  --model gpt-4

# Gemini version (modify the script to add command-line functionality)
python market_summary_gemini.py \
  --examples examples.json \
  --articles data/new_articles/market_news_2025-05-27.txt \
  --output data/new_summaries/market_summary_2025-05-27.txt
```

## Advanced Customization

### 1. Adding Structure to Summaries

If you want to ensure your summaries follow a specific structure, modify the system message:

```python
system_message = """You are a financial analyst who writes market close summaries.
Your summaries should follow this structure:
1. Overall market performance (indices)
2. Key sector movements
3. Notable stock performers
4. Economic factors influencing the market
5. Brief outlook

Keep the summary concise but informative, around 250-300 words."""
```

### 2. Implementing a Custom Loader

If your data is stored in a database or has a unique format, implement a custom loader:

```python
def load_examples_from_database(self, connection_string, query):
    """Load examples from a database"""
    import sqlite3  # or another database library
    
    conn = sqlite3.connect(connection_string)
    cursor = conn.cursor()
    cursor.execute(query)
    
    for row in cursor.fetchall():
        date, article, summary = row
        self.few_shot_examples.append({
            'date': date,
            'article': article,
            'summary': summary
        })
    
    conn.close()
    print(f"Loaded {len(self.few_shot_examples)} examples from database")
```

### 3. Implementing Evaluation

Add functionality to evaluate the quality of generated summaries:

```python
def evaluate_summary(generated_summary, reference_summary):
    """Compare generated summary to a reference summary"""
    from rouge import Rouge
    
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    
    print(f"ROUGE-1: {scores[0]['rouge-1']['f']}")
    print(f"ROUGE-2: {scores[0]['rouge-2']['f']}")
    print(f"ROUGE-L: {scores[0]['rouge-l']['f']}")
    
    return scores
```

## Troubleshooting

1. **API Key Issues**: Ensure your API key is correctly set as an environment variable or passed directly.

2. **Context Length Errors**: If you get context length errors, reduce the number of examples or the length of articles.

3. **File Encoding Issues**: If you encounter encoding errors, specify the encoding when opening files:
   ```python
   with open(filename, 'r', encoding='utf-8') as f:
   ```

4. **Date Matching Problems**: If your files aren't being matched correctly, print the extracted dates to debug:
   ```python
   for filename in article_files:
       match = re.search(date_pattern, filename)
       if match:
           print(f"File: {filename}, Extracted date: {match.group(1)}")
       else:
           print(f"No date found in: {filename}")
   ```

## Next Steps

- Implement a web interface for easier interaction
- Add support for batch processing multiple days at once
- Integrate with news APIs to automatically fetch articles
- Implement a feedback loop to improve summaries over time
