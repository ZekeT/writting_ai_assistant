# main.py
"""Command-line interface and entry point."""

import click
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from graph.workflow import LearningWorkflow
from models.openai_provider import OpenAIProvider
from utils.file_io import FileManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('learning.log'),
        logging.StreamHandler()
    ]
)


@click.group()
def cli():
    """Financial News Summarization Learning System."""
    pass


@cli.command()
@click.option('--data-dir', default='data', help='Directory containing training data')
@click.option('--model', default='gpt-4', help='OpenAI model to use')
@click.option('--max-iterations', default=5, help='Maximum iterations per training set')
@click.option('--output-dir', default='output', help='Directory to save results')
@click.option('--temperature', default=0.3, help='Model temperature')
def learn(data_dir, model, max_iterations, output_dir, temperature):
    """Run the learning process to develop optimal summarization prompts."""

    logger = logging.getLogger(__name__)
    logger.info("Starting learning process...")

    # Validate data directory
    if not Path(data_dir).exists():
        raise click.ClickException(f"Data directory not found: {data_dir}")

    try:
        # Initialize components
        llm_provider = OpenAIProvider(model=model, temperature=temperature)
        workflow = LearningWorkflow(
            llm_provider=llm_provider,
            data_dir=data_dir,
            max_iterations=max_iterations
        )

        # Run learning process
        results = workflow.run_learning_process()

        # Save results
        file_manager = FileManager(output_dir)

        # Save detailed results
        file_manager.save_json(
            {
                'learning_summary': results['learning_summary'],
                'average_scores': results['average_scores'],
                'model_info': llm_provider.get_model_info()
            },
            'learning_results.json'
        )

        # Save production prompt
        production_prompt = workflow.get_production_prompt()
        file_manager.save_text(production_prompt, 'production_prompt.md')

        # Save prompt template for programmatic use
        template = results['final_prompt_template']
        file_manager.save_json(
            {
                'base_instruction': template.base_instruction,
                'us_guidance': template.us_guidance,
                'europe_guidance': template.europe_guidance,
                'asia_guidance': template.asia_guidance,
                'formatting_rules': template.formatting_rules,
                'version': template.version
            },
            'prompt_template.json'
        )

        # Print summary
        summary = results['learning_summary']
        click.echo(f"\n{'='*50}")
        click.echo("LEARNING COMPLETE")
        click.echo(f"{'='*50}")
        click.echo(
            f"Training sets processed: {summary['total_training_sets']}")
        click.echo(f"Successful sets: {summary['successful_sets']}")
        click.echo(f"Total iterations: {summary['total_iterations']}")
        click.echo(
            f"Average iterations per set: {summary['average_iterations_per_set']:.1f}")
        click.echo(f"Final prompt version: {summary['final_prompt_version']}")
        click.echo(
            f"Final average score: {summary['final_average_scores']['overall_score']:.3f}")
        click.echo(
            f"Improvement achieved: {'Yes' if summary['improvement_achieved'] else 'No'}")
        click.echo(f"\nResults saved to: {output_dir}/")
        click.echo(f"Production prompt: {output_dir}/production_prompt.md")

    except Exception as e:
        logger.error(f"Learning process failed: {e}")
        raise click.ClickException(f"Learning failed: {e}")


@cli.command()
@click.option('--prompt-file', default='output/prompt_template.json', help='Path to learned prompt template')
@click.option('--articles-dir', required=True, help='Directory containing news articles to summarize')
@click.option('--model', default='gpt-4', help='OpenAI model to use')
@click.option('--output-file', default='summary.md', help='Output file for generated summary')
def summarize(prompt_file, articles_dir, model, output_file):
    """Use learned prompt to summarize new news articles."""

    logger = logging.getLogger(__name__)

    # Load learned prompt template
    file_manager = FileManager()
    template_data = file_manager.load_json(prompt_file)

    if not template_data:
        raise click.ClickException(
            f"Could not load prompt template from {prompt_file}")

    # Load articles
    articles_path = Path(articles_dir)
    if not articles_path.exists():
        raise click.ClickException(
            f"Articles directory not found: {articles_dir}")

    article_files = list(articles_path.glob('*.md'))
    if not article_files:
        raise click.ClickException(
            f"No markdown files found in {articles_dir}")

    # Read articles
    articles_text = []
    for i, article_file in enumerate(sorted(article_files)):
        with open(article_file, 'r', encoding='utf-8') as f:
            content = f.read()
        articles_text.append(
            f"**Article {i+1}: {article_file.stem}**\n{content}")

    # Build prompt
    combined_articles = "\n\n".join(articles_text)

    prompt = f"""{template_data['base_instruction']}

{template_data['us_guidance']}

{template_data['europe_guidance']}

{template_data['asia_guidance']}

{template_data['formatting_rules']}

NEWS ARTICLES TO SUMMARIZE:
{combined_articles}

Please provide the market wrap summary following the exact format specified above."""

    # Generate summary
    try:
        llm_provider = OpenAIProvider(model=model)
        summary = llm_provider.generate(prompt)

        # Save summary
        file_manager.save_text(summary, output_file)

        click.echo(f"Summary generated and saved to: {output_file}")
        click.echo(f"\nPreview:")
        click.echo("-" * 40)
        click.echo(summary[:500] + "..." if len(summary) > 500 else summary)

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise click.ClickException(f"Summarization failed: {e}")


@cli.command()
@click.option('--data-dir', default='data', help='Directory containing training data')
def validate_data(data_dir):
    """Validate the structure of training data."""

    from core.data_loader import DataLoader

    try:
        loader = DataLoader(data_dir)
        training_sets = loader.load_all_training_sets()

        click.echo(f"Found {len(training_sets)} training sets:")

        for i, training_set in enumerate(training_sets, 1):
            click.echo(f"\n{i}. Training Set #{i}")
            click.echo(f"   Articles: {len(training_set.articles)}")
            click.echo(
                f"   Expected summary length: {len(str(training_set.expected_summary))} characters")

            # Validate each article
            for j, article in enumerate(training_set.articles, 1):
                if not article.title or not article.content:
                    click.echo(f"   ⚠️  Article {j} missing title or content")
                else:
                    click.echo(f"   ✓ Article {j}: {article.title[:50]}...")

            # Validate expected summary
            if not training_set.expected_summary:
                click.echo(f"   ⚠️  Missing expected summary")
            else:
                click.echo(
                    f"   ✓ Expected summary: {str(training_set.expected_summary)[:100]}...")

        click.echo(
            f"\n✓ Data validation complete. Found {len(training_sets)} valid training sets.")

    except Exception as e:
        click.echo(f"❌ Data validation failed: {e}")
        raise click.ClickException(f"Validation failed: {e}")


@cli.command()
@click.option('--output-dir', default='output', help='Directory to check for results')
def status(output_dir):
    """Check the status of learning results and available outputs."""

    output_path = Path(output_dir)

    if not output_path.exists():
        click.echo(f"❌ Output directory not found: {output_dir}")
        return

    click.echo(f"📁 Checking output directory: {output_dir}")
    click.echo("-" * 40)

    # Check for key files
    files_to_check = [
        ('learning_results.json', 'Learning results'),
        ('production_prompt.md', 'Production prompt'),
        ('prompt_template.json', 'Prompt template'),
        ('learning.log', 'Learning log')
    ]

    found_files = []
    missing_files = []

    for filename, description in files_to_check:
        file_path = output_path / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            found_files.append((filename, description, file_size))
        else:
            missing_files.append((filename, description))

    # Display found files
    if found_files:
        click.echo("✓ Found files:")
        for filename, description, size in found_files:
            click.echo(f"  • {description}: {filename} ({size:,} bytes)")

    # Display missing files
    if missing_files:
        click.echo("\n⚠️  Missing files:")
        for filename, description in missing_files:
            click.echo(f"  • {description}: {filename}")

    # Try to load and display learning results if available
    results_file = output_path / 'learning_results.json'
    if results_file.exists():
        try:
            file_manager = FileManager()
            results = file_manager.load_json(str(results_file))

            if results and 'learning_summary' in results:
                summary = results['learning_summary']
                click.echo(f"\n📊 Last Learning Run Summary:")
                click.echo(
                    f"  • Training sets processed: {summary.get('total_training_sets', 'N/A')}")
                click.echo(
                    f"  • Successful sets: {summary.get('successful_sets', 'N/A')}")
                click.echo(
                    f"  • Total iterations: {summary.get('total_iterations', 'N/A')}")
                click.echo(
                    f"  • Final score: {summary.get('final_average_scores', {}).get('overall_score', 'N/A')}")
                click.echo(
                    f"  • Improvement achieved: {summary.get('improvement_achieved', 'N/A')}")

        except Exception as e:
            click.echo(f"\n⚠️  Could not read learning results: {e}")

    click.echo(f"\n{'='*40}")
    if found_files:
        click.echo("✓ System appears to be working. Ready to learn or summarize!")
    else:
        click.echo("❌ No output files found. Run 'learn' command first.")


@cli.command()
@click.option('--config-file', default='.env', help='Environment configuration file')
def setup(config_file):
    """Setup and validate the environment configuration."""

    click.echo("🔧 Setting up Financial News Summarization System")
    click.echo("=" * 50)

    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        click.echo("❌ Missing required environment variables:")
        for var in missing_vars:
            click.echo(f"  • {var}")

        click.echo(f"\nPlease add these to your {config_file} file:")
        for var in missing_vars:
            if var == 'OPENAI_API_KEY':
                click.echo(f"{var}=your_openai_api_key_here")

        return

    # Test OpenAI connection
    try:
        click.echo("🧪 Testing OpenAI connection...")
        llm_provider = OpenAIProvider(model='gpt-3.5-turbo', temperature=0.1)
        test_response = llm_provider.generate(
            "Say 'Connection successful' if you can read this.")

        if "successful" in test_response.lower():
            click.echo("✓ OpenAI connection working")
        else:
            click.echo("⚠️  OpenAI connection may have issues")

    except Exception as e:
        click.echo(f"❌ OpenAI connection failed: {e}")
        return

    # Check directory structure
    click.echo("\n📁 Checking directory structure...")
    required_dirs = ['data', 'output']

    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            click.echo(f"✓ Created directory: {dir_name}/")
        else:
            click.echo(f"✓ Directory exists: {dir_name}/")

    click.echo("\n🎉 Setup complete! System is ready to use.")
    click.echo("\nNext steps:")
    click.echo("1. Add training data to the data/ directory")
    click.echo("2. Run 'python main.py validate-data' to check your data")
    click.echo("3. Run 'python main.py learn' to start the learning process")


if __name__ == '__main__':
    cli()
