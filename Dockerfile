# Journo LLM - Development and Data Prep Container
#
# This container handles:
# - Data extraction from WordPress
# - Corpus preparation
# - Local testing (CPU-only)
#
# For GPU training, use Modal (see train_modal.py)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Modal CLI
RUN pip install modal

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY journo_llm/ journo_llm/
COPY README.md .

# Install Python dependencies
RUN pip install -e ".[dev]"

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - show help
CMD ["journo", "--help"]
