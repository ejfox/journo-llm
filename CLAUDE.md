# Claude Notes for Journo LLM

## Project Status (Jan 17, 2026)

### What's Done
- Full project scaffold with 17 design decisions documented in README
- 7-phase implementation roadmap defined
- Data extraction code working (pulls from WordPress REST API)
- Modal training scripts ready for gpt-oss-20b fine-tuning
- Docker + docker-compose setup
- SETUP.md with fresh Mac → training instructions
- Repo live at: https://github.com/ejfox/journo-llm

### In Progress
- Fetching Times of San Diego corpus (81,155 articles total)
- ~600 articles/minute, full fetch takes ~2-2.5 hours

## Quick Resume Commands

```bash
cd ~/code/journo-llm

# Check if fetch is still running
ps aux | grep journo

# Check fetch progress
wc -l data/tosd_corpus/corpus.jsonl
ls -lh data/tosd_corpus/

# If fetch stopped, resume:
WORDPRESS_URL=https://timesofsandiego.com .venv/bin/python -m journo_llm.cli fetch --output data/tosd_corpus

# View corpus stats
.venv/bin/python -m journo_llm.cli stats data/tosd_corpus
```

## Next Steps (Phase 1 continuation)

1. **Wait for fetch to complete** (~81k articles)
2. **Check stats**: `journo stats data/tosd_corpus`
3. **Prepare train/eval split**: `journo prepare data/tosd_corpus`
4. **Upload to Modal**: `modal run journo_llm/train_modal.py --action upload --corpus-path data/tosd_corpus/corpus.jsonl`

## Next Steps (Phase 2 - Training Infra)

1. Set up Modal auth: `modal token new`
2. Create HF secret: `modal secret create huggingface HF_TOKEN=<token>`
3. Test small training run (1% data)
4. Set up MLflow: `docker compose up -d mlflow`

## Key Files

| File | Purpose |
|------|---------|
| `journo_llm/data.py` | WordPress extraction, corpus handling |
| `journo_llm/train_modal.py` | Modal GPU training scripts |
| `journo_llm/cli.py` | CLI interface |
| `data/tosd_corpus/` | Times of San Diego corpus (gitignored) |

## Verified Assumptions

- ✅ Times of San Diego API is public (no auth needed)
- ✅ 81,155 articles available (~40M words, ~10M tokens)
- ✅ gpt-oss-20b exists on HuggingFace (Apache 2.0, fits 16GB)
- ✅ Modal CLI works (v0.64.93)
- ✅ All Python deps install cleanly

## Related Projects

- `~/client-code/newswell/newswell-studio` - Has WordPress integration code we borrowed patterns from
- `~/client-code/newswell/newswell-studio/app/utils/newsworthiness.ts` - Newsworthiness scoring interface (can plug our model into this later)

## Cost Estimates

| Action | Cost |
|--------|------|
| Full corpus fetch | Free (local) |
| 1 epoch continued pretraining (A100) | $20-50 |
| Full training (3 epochs) | $60-150 |
| Instruction fine-tuning | $10-30 |

## Design Decisions Summary

1. **Corpus**: Times of San Diego (81k articles) → federated multi-publisher
2. **Base model**: gpt-oss-20b (Apache 2.0, 16GB RAM)
3. **Training**: Modal.com (serverless GPUs)
4. **Values**: Truthfulness, harm reduction, accountability
5. **RLHF**: Argilla + real journalists
6. **License**: Open for journalism only
7. **Multi-publisher**: Federated learning (data stays local)

See README.md for full 17 decisions.
