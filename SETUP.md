# Journo LLM Setup Guide

Get from a fresh Mac to training your own journalism-focused LLM.

## Prerequisites

You need:
- macOS (tested on Apple Silicon)
- ~20GB free disk space
- A HuggingFace account (free)
- A Modal account (free tier available)

## Step 1: Install Homebrew (if you don't have it)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Step 2: Install Python 3.11+

```bash
brew install python@3.11
```

Verify:
```bash
python3 --version  # Should be 3.11+
```

## Step 3: Install uv (fast Python package manager)

```bash
brew install uv
```

Or with pip:
```bash
pip install uv
```

## Step 4: Clone and Install Journo LLM

```bash
# Clone the repo
git clone https://github.com/ejfox/journo-llm.git
cd journo-llm

# Create venv and install dependencies (uv does this automatically)
uv sync

# Or with pip:
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Step 5: Set Up HuggingFace

1. Create account at https://huggingface.co/join
2. Create token at https://huggingface.co/settings/tokens
3. Accept gpt-oss-20b license at https://huggingface.co/openai/gpt-oss-20b

```bash
# Install HF CLI
pip install huggingface_hub

# Login (paste your token when prompted)
huggingface-cli login
```

## Step 6: Set Up Modal

1. Create account at https://modal.com
2. Install and authenticate:

```bash
# Install Modal CLI (already in dependencies, but just in case)
pip install modal

# Authenticate (opens browser)
modal token new

# Verify
modal token show
```

3. Create a Modal secret for HuggingFace:
```bash
modal secret create huggingface HF_TOKEN=<your-hf-token>
```

## Step 7: Configure Environment

```bash
# Copy example env
cp .env.example .env

# Edit with your settings (WordPress URL is pre-filled for Times of San Diego)
# No WordPress auth needed - API is public
```

## Step 8: Test Everything

```bash
# Activate venv if not already
source .venv/bin/activate

# Test CLI
journo --help

# Test WordPress connection (fetches 10 articles)
journo fetch --max 10 --output data/test_corpus

# Check stats
journo stats data/test_corpus
```

You should see something like:
```
Corpus: data/test_corpus
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric       ┃ Value     ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Articles     │ 10        │
│ Words        │ ~5,000    │
│ Est. Tokens  │ ~1,500    │
└──────────────┴───────────┘
```

## Step 9: Fetch Full Corpus

```bash
# This takes a while - 81k articles
journo fetch --output data/tosd_corpus

# Or do it in batches for testing
journo fetch --max 1000 --output data/tosd_corpus_sample
```

## Step 10: Upload to Modal and Train

```bash
# Upload corpus to Modal volume
modal run journo_llm/train_modal.py --action upload --corpus-path data/tosd_corpus/corpus.jsonl

# Start continued pretraining (this costs money - ~$20-100 depending on epochs)
modal run journo_llm/train_modal.py --action train
```

## Optional: Local Development with Docker

If you prefer Docker:

```bash
# Start MLflow for experiment tracking
docker compose up -d mlflow

# Run data fetch in container
docker compose run journo fetch --output data/tosd_corpus

# View MLflow UI
open http://localhost:5000
```

## Cost Estimates

| Action | Approximate Cost |
|--------|-----------------|
| Data fetching | Free (local) |
| 1 epoch continued pretraining (A100) | $20-50 |
| Full training run (3 epochs) | $60-150 |
| Instruction fine-tuning | $10-30 |
| Inference testing | $0.50-2/hour |

Modal has a free tier that may cover initial testing.

## Troubleshooting

### "Modal not authenticated"
```bash
modal token new
```

### "HuggingFace model access denied"
Make sure you:
1. Accepted the model license on HuggingFace
2. Created the Modal secret: `modal secret create huggingface HF_TOKEN=<token>`

### "WordPress API timeout"
The API is rate-limited. The script handles this with retries, but if issues persist:
```bash
# Fetch in smaller batches
journo fetch --max 5000 --output data/batch1
```

### "Out of memory during training"
Reduce batch size in `train_modal.py`:
```python
train_continued_pretraining.remote(batch_size=2)
```

## Next Steps

After setup:
1. **Phase 1**: Fetch full corpus, analyze statistics
2. **Phase 2**: Run small training test (1% of data)
3. **Phase 3**: Full continued pretraining
4. See README.md for the full 7-phase roadmap

## Getting Help

- GitHub Issues: https://github.com/ejfox/journo-llm/issues
- Modal Docs: https://modal.com/docs
- HuggingFace Docs: https://huggingface.co/docs
