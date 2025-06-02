import os
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage


class NewsDigestGenerator:
    def __init__(self, training_data_path, llm_model="claude-3-sonnet-20240229"):
        """
        Initialize the digest generator with training data and LLM configuration.

        Args:
            training_data_path: Path to directory containing training folders
            llm_model: LLM model identifier (default: Claude 3 Sonnet)
        """
        self.llm = ChatAnthropic(
            model=llm_model, temperature=0.2, max_tokens=4096)
        self.base_prompt = self._build_prompt_from_data(training_data_path)
        self.graph = self._build_graph()

    def _read_folder(self, folder_path):
        """
        Read articles and summary from a training folder.

        Returns:
            tuple: (list of article contents, summary content)
        """
        articles = []
        summary = None

        for file in os.listdir(folder_path):
            if file.endswith(".md"):
                with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    if file.startswith("train_"):
                        articles.append(content)
                    elif file.startswith("test_"):
                        summary = content

        if not articles or summary is None:
            raise ValueError(f"Invalid folder structure in {folder_path}")

        return articles, summary

    def _build_prompt_from_data(self, training_data_path):
        """
        Construct few-shot prompt from training examples.

        Returns:
            str: Formatted prompt with system message and examples
        """
        examples = []

        # Process all training folders
        for folder in os.listdir(training_data_path):
            folder_path = os.path.join(training_data_path, folder)
            if os.path.isdir(folder_path):
                try:
                    articles, summary = self._read_folder(folder_path)
                    examples.append((articles, summary))
                except Exception as e:
                    print(f"Skipping folder {folder}: {str(e)}")

        if not examples:
            raise RuntimeError("No valid training examples found")

        # Start with system message
        prompt = (
            "You are an expert news synthesizer. Your task is to analyze multiple articles "
            "about the same event and produce a comprehensive digest that:\n"
            "1. Identifies ALL key facts and perspectives\n"
            "2. Resolves contradictions between sources\n"
            "3. Maintains neutrality\n"
            "4. Highlights developments over time\n"
            "5. Omits redundant information\n\n"
            "Format guidelines:\n"
            "- Use clear section headings\n"
            "- Include chronological development\n"
            "- Attribute claims to sources\n"
            "- Preserve quantitative data\n\n"
            "Examples:\n"
        )

        # Add examples (using first 2 for few-shot learning)
        for i, (articles, summary) in enumerate(examples[:2]):
            prompt += f"\nEXAMPLE SET {i+1}:\n"
            prompt += f"ARTICLES ({len(articles)} sources):\n"

            for j, article in enumerate(articles):
                # Use first 150 chars as identifier
                truncated = article[:150].replace('\n', ' ') + "..."
                prompt += f"- Source {j+1}: {truncated}\n"

            prompt += f"\nEXPECTED DIGEST:\n{summary}\n"
            prompt += f"{'-'*40}\n"

        prompt += "\nNEW ARTICLE SET:\n"
        return prompt

    def _build_graph(self):
        """Build LangGraph workflow for summarization."""
        def summarization_node(state):
            """Node that performs the summarization using the LLM."""
            articles = state["articles"]

            # Format article list for prompt
            articles_formatted = "\n".join(
                f"ARTICLE {i+1}:\n{art[:2000]}...\n"
                for i, art in enumerate(articles)
            )

            full_prompt = self.base_prompt + articles_formatted

            # Call LLM with structured messages
            messages = [
                SystemMessage(content="You are a professional news analyst."),
                HumanMessage(content=full_prompt)
            ]

            result = self.llm.invoke(messages)
            return {"summary": result.content}

        # Define graph structure
        builder = StateGraph({"articles": None, "summary": None})
        builder.add_node("summarize", summarization_node)
        builder.set_entry_point("summarize")
        return builder.compile()

    def generate_digest(self, articles):
        """
        Generate summary digest from new articles.

        Args:
            articles: List of article contents (strings)

        Returns:
            str: Generated summary digest
        """
        if len(articles) < 5:
            print("Warning: Recommended minimum 5 articles for best results")

        state = {"articles": articles}
        result = self.graph.invoke(state)
        return result["summary"]

    def save_prompt(self, file_path):
        """Save the transformation prompt for external use."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.base_prompt)


# Usage Example
if __name__ == "__main__":
    # Initialize with training data
    generator = NewsDigestGenerator("/path/to/training_folders")

    # Save the transformation prompt
    generator.save_prompt("digest_prompt.txt")

    # Process new articles
    new_articles = [
        "Full text of article 1...",
        "Full text of article 2...",
        # ... 6-8 more articles
    ]

    digest = generator.generate_digest(new_articles)
    print("Generated Digest:")
    print(digest)
