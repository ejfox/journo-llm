"""
Journo LLM CLI - Command line interface for data prep and training.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from journo_llm.data import (
    WordPressConfig,
    fetch_articles,
    save_corpus,
    load_corpus,
    estimate_tokens,
)

app = typer.Typer(
    name="journo",
    help="Journo LLM - Train journalism-focused language models",
)
console = Console()


@app.command()
def fetch(
    output_dir: Path = typer.Option(
        Path("data/corpus"),
        "--output", "-o",
        help="Output directory for corpus",
    ),
    wordpress_url: Optional[str] = typer.Option(
        None,
        "--url",
        help="WordPress site URL (or set WORDPRESS_URL env var)",
    ),
    max_articles: Optional[int] = typer.Option(
        None,
        "--max", "-m",
        help="Maximum articles to fetch (default: all)",
    ),
    format: str = typer.Option(
        "jsonl",
        "--format", "-f",
        help="Output format: jsonl or txt",
    ),
):
    """
    Fetch articles from a WordPress site.

    Pulls all published articles and saves them in training-ready format.
    """
    config = WordPressConfig.from_env()

    if wordpress_url:
        config.url = wordpress_url

    if not config.url:
        console.print("[red]Error: WordPress URL required")
        console.print("Set WORDPRESS_URL env var or use --url flag")
        raise typer.Exit(1)

    console.print(f"[blue]Fetching from: {config.url}")

    articles = fetch_articles(config, max_articles=max_articles)
    stats = save_corpus(articles, output_dir, format=format)

    console.print(f"\n[green]Done! Corpus saved to {output_dir}")

    # Show stats table
    table = Table(title="Corpus Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Articles", f"{stats['total_articles']:,}")
    table.add_row("Words", f"{stats['total_words']:,}")
    table.add_row("Characters", f"{stats['total_chars']:,}")
    table.add_row("Est. Tokens", f"{int(stats['total_chars'] / 4):,}")

    console.print(table)


@app.command()
def stats(
    corpus_dir: Path = typer.Argument(
        Path("data/corpus"),
        help="Path to corpus directory",
    ),
):
    """Show statistics for a corpus."""
    import json

    stats_file = corpus_dir / "stats.json"
    if not stats_file.exists():
        console.print(f"[red]Stats file not found: {stats_file}")
        raise typer.Exit(1)

    with open(stats_file) as f:
        data = json.load(f)

    table = Table(title=f"Corpus: {corpus_dir}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Articles", f"{data['total_articles']:,}")
    table.add_row("Words", f"{data['total_words']:,}")
    table.add_row("Characters", f"{data['total_chars']:,}")
    table.add_row("Est. Tokens", f"{estimate_tokens(corpus_dir):,}")
    table.add_row("Categories", str(len(data.get("categories", {}))))
    table.add_row("Authors", str(len(data.get("authors", {}))))

    console.print(table)

    # Top categories
    if data.get("categories"):
        cat_table = Table(title="Top Categories")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green")

        sorted_cats = sorted(
            data["categories"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        for cat, count in sorted_cats:
            cat_table.add_row(cat, str(count))

        console.print(cat_table)


@app.command()
def prepare(
    corpus_dir: Path = typer.Argument(
        Path("data/corpus"),
        help="Path to corpus directory",
    ),
    output_dir: Path = typer.Option(
        Path("data/prepared"),
        "--output", "-o",
        help="Output directory for prepared data",
    ),
    train_split: float = typer.Option(
        0.95,
        "--train-split",
        help="Fraction of data for training",
    ),
):
    """
    Prepare corpus for training.

    Splits into train/eval sets and formats for HuggingFace datasets.
    """
    import json
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all articles
    articles = list(load_corpus(corpus_dir))
    random.shuffle(articles)

    split_idx = int(len(articles) * train_split)
    train_articles = articles[:split_idx]
    eval_articles = articles[split_idx:]

    console.print(f"Train: {len(train_articles)}, Eval: {len(eval_articles)}")

    # Save splits
    for name, data in [("train", train_articles), ("eval", eval_articles)]:
        filepath = output_dir / f"{name}.jsonl"
        with open(filepath, "w") as f:
            for article in data:
                f.write(json.dumps(article) + "\n")
        console.print(f"Saved {filepath}")

    console.print(f"[green]Prepared data saved to {output_dir}")


@app.command()
def upload(
    corpus_path: Path = typer.Argument(
        Path("data/corpus/corpus.jsonl"),
        help="Path to corpus file",
    ),
):
    """
    Upload corpus to Modal volume for training.

    Requires Modal CLI to be configured.
    """
    import subprocess

    console.print(f"[blue]Uploading {corpus_path} to Modal...")

    result = subprocess.run(
        [
            "modal", "run",
            "journo_llm/train_modal.py",
            "--action", "upload",
            "--corpus-path", str(corpus_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Upload complete!")
        console.print(result.stdout)
    else:
        console.print("[red]Upload failed:")
        console.print(result.stderr)
        raise typer.Exit(1)


@app.command()
def train(
    action: str = typer.Argument(
        "pretrain",
        help="Training action: pretrain, finetune",
    ),
):
    """
    Run training on Modal.

    Actions:
    - pretrain: Continue pretraining on journalism corpus
    - finetune: Instruction fine-tuning for tasks
    """
    import subprocess

    modal_action = "train" if action == "pretrain" else "finetune"

    console.print(f"[blue]Starting {action} on Modal...")

    result = subprocess.run(
        [
            "modal", "run",
            "journo_llm/train_modal.py",
            "--action", modal_action,
        ],
    )

    if result.returncode != 0:
        raise typer.Exit(1)


@app.command()
def create_instructions(
    corpus_dir: Path = typer.Argument(
        Path("data/corpus"),
        help="Path to corpus directory",
    ),
    output_file: Path = typer.Option(
        Path("data/instructions.jsonl"),
        "--output", "-o",
        help="Output file for instruction data",
    ),
    task: str = typer.Option(
        "summarize",
        "--task", "-t",
        help="Task type: summarize, newsworthiness, citations",
    ),
):
    """
    Create instruction fine-tuning data from corpus.

    This is a simple heuristic approach - for production,
    you'd want human-written instruction/response pairs.
    """
    import json

    articles = list(load_corpus(corpus_dir))
    instructions = []

    console.print(f"[blue]Creating {task} instructions from {len(articles)} articles...")

    for article in articles:
        if task == "summarize":
            # Use excerpt as summary target (if available)
            if article.get("excerpt") and len(article["excerpt"]) > 50:
                instructions.append({
                    "instruction": f"Summarize the following article:\n\n{article['content'][:2000]}",
                    "response": article["excerpt"],
                })

        elif task == "newsworthiness":
            # Simple heuristic based on categories
            categories = article.get("categories", [])
            score = 50  # baseline
            reasoning = []

            if any("breaking" in c.lower() for c in categories):
                score += 30
                reasoning.append("Breaking news - high timeliness")
            if any("government" in c.lower() or "politics" in c.lower() for c in categories):
                score += 15
                reasoning.append("Government/politics - high prominence")
            if article.get("word_count", 0) > 1000:
                score += 10
                reasoning.append("Long-form coverage suggests depth")

            instructions.append({
                "instruction": f"Evaluate the newsworthiness of this article:\n\n{article['title']}\n\n{article['content'][:1000]}",
                "response": f"Newsworthiness score: {min(score, 100)}/100\n\nReasoning:\n" + "\n".join(f"- {r}" for r in reasoning) if reasoning else "- Standard news coverage",
            })

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for inst in instructions:
            f.write(json.dumps(inst) + "\n")

    console.print(f"[green]Created {len(instructions)} instruction pairs")
    console.print(f"Saved to {output_file}")


if __name__ == "__main__":
    app()
