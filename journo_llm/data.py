"""
Data extraction and processing for Journo LLM.

Pulls journalism data from WordPress sites and prepares it for training.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

console = Console()


@dataclass
class Article:
    """A single journalism article."""
    id: int
    title: str
    content: str
    excerpt: str
    author: str | None
    date: str
    categories: list[str]
    tags: list[str]
    word_count: int
    url: str


@dataclass
class WordPressConfig:
    """WordPress connection configuration."""
    url: str
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls) -> "WordPressConfig":
        """Load config from environment variables."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            url=os.getenv("WORDPRESS_URL", ""),
            username=os.getenv("WORDPRESS_USER"),
            password=os.getenv("WORDPRESS_APP_PASSWORD"),
        )


def strip_html(html: str) -> str:
    """Clean HTML and extract plain text."""
    if not html:
        return ""

    # Remove script and style tags
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up whitespace
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def fetch_articles(
    config: WordPressConfig,
    per_page: int = 100,
    max_articles: int | None = None,
) -> Iterator[Article]:
    """
    Fetch all articles from a WordPress site.

    Yields Article objects as they're fetched.
    """
    if not config.url:
        raise ValueError("WordPress URL is required")

    api_base = f"{config.url.rstrip('/')}/wp-json/wp/v2"

    headers = {"Content-Type": "application/json"}
    if config.username and config.password:
        import base64
        credentials = base64.b64encode(
            f"{config.username}:{config.password}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials}"

    page = 1
    total_fetched = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles...", total=None)

        while True:
            params = {
                "page": page,
                "per_page": per_page,
                "_embed": "1",
                "orderby": "date",
                "order": "desc",
            }

            response = requests.get(
                f"{api_base}/posts",
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 400:
                # No more pages
                break

            response.raise_for_status()
            posts = response.json()

            if not posts:
                break

            for post in posts:
                # Extract author
                author = None
                if "_embedded" in post and "author" in post["_embedded"]:
                    authors = post["_embedded"]["author"]
                    if authors:
                        author = authors[0].get("name")

                # Extract categories
                categories = []
                if "_embedded" in post and "wp:term" in post["_embedded"]:
                    terms = post["_embedded"]["wp:term"]
                    if terms and len(terms) > 0:
                        categories = [t["name"] for t in terms[0]]

                # Extract tags
                tags = []
                if "_embedded" in post and "wp:term" in post["_embedded"]:
                    terms = post["_embedded"]["wp:term"]
                    if terms and len(terms) > 1:
                        tags = [t["name"] for t in terms[1]]

                # Clean content
                content = strip_html(post.get("content", {}).get("rendered", ""))
                excerpt = strip_html(post.get("excerpt", {}).get("rendered", ""))
                title = strip_html(post.get("title", {}).get("rendered", "Untitled"))

                article = Article(
                    id=post["id"],
                    title=title,
                    content=content,
                    excerpt=excerpt,
                    author=author,
                    date=post.get("date", ""),
                    categories=categories,
                    tags=tags,
                    word_count=len(content.split()),
                    url=post.get("link", ""),
                )

                yield article
                total_fetched += 1

                progress.update(task, description=f"Fetched {total_fetched} articles...")

                if max_articles and total_fetched >= max_articles:
                    return

            page += 1


def save_corpus(
    articles: Iterator[Article],
    output_dir: Path,
    format: str = "jsonl",
) -> dict:
    """
    Save articles to disk in training-ready format.

    Returns corpus statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_articles": 0,
        "total_words": 0,
        "total_chars": 0,
        "categories": {},
        "authors": {},
    }

    if format == "jsonl":
        output_file = output_dir / "corpus.jsonl"
        with open(output_file, "w") as f:
            for article in articles:
                # Write as JSON line
                record = {
                    "id": article.id,
                    "title": article.title,
                    "content": article.content,
                    "author": article.author,
                    "date": article.date,
                    "categories": article.categories,
                    "tags": article.tags,
                    "word_count": article.word_count,
                    "url": article.url,
                }
                f.write(json.dumps(record) + "\n")

                # Update stats
                stats["total_articles"] += 1
                stats["total_words"] += article.word_count
                stats["total_chars"] += len(article.content)

                for cat in article.categories:
                    stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

                if article.author:
                    stats["authors"][article.author] = stats["authors"].get(article.author, 0) + 1

    elif format == "txt":
        # Plain text format for pretraining
        output_file = output_dir / "corpus.txt"
        with open(output_file, "w") as f:
            for article in articles:
                # Format: title + content with article separator
                f.write(f"# {article.title}\n\n")
                f.write(article.content)
                f.write("\n\n<|endofarticle|>\n\n")

                stats["total_articles"] += 1
                stats["total_words"] += article.word_count
                stats["total_chars"] += len(article.content)

    # Save stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"[green]Saved {stats['total_articles']} articles to {output_dir}")
    console.print(f"[blue]Total words: {stats['total_words']:,}")
    console.print(f"[blue]Total chars: {stats['total_chars']:,}")

    return stats


def load_corpus(corpus_dir: Path) -> Iterator[dict]:
    """Load corpus from JSONL file."""
    corpus_file = Path(corpus_dir) / "corpus.jsonl"
    with open(corpus_file) as f:
        for line in f:
            yield json.loads(line)


def estimate_tokens(corpus_dir: Path, chars_per_token: float = 4.0) -> int:
    """Estimate token count for corpus."""
    stats_file = Path(corpus_dir) / "stats.json"
    with open(stats_file) as f:
        stats = json.load(f)
    return int(stats["total_chars"] / chars_per_token)
