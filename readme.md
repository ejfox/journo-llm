# Journo LLM

A hand-made LLM made from scratch by journalists and for journalists.

## Design Decisions

### 1. Corpus Strategy
- **Initial corpus**: Times of San Diego WordPress archive (**81,155 articles** - ~40M words, ~10M tokens)
- **Future expansion**: Archives from multiple news publishers
- **Rationale**: Start with what we have, build pipeline that scales

### 2. Model Size Strategy
- **Approach**: Start tiny, scale up
- **Phase 1**: < 500M params - prove the pipeline works on consumer/cheap hardware
- **Phase 2**: Scale to 1-3B once process is validated
- **Rationale**: Don't burn money until we know what we're doing

### 3. Training Infrastructure
- **Platform**: Modal.com
- **Why**: Serverless GPUs, pay-per-second, Python-native, good free tier
- **Workflow**: Develop locally, push training jobs to Modal
- **Fallback**: Can always move to Lambda/vast.ai for long runs if cost becomes issue

---

### 4. Tokenizer Approach
- **Method**: Train BPE (Byte Pair Encoding) from scratch on our corpus
- **Library**: `sentencepiece` or HuggingFace `tokenizers`
- **Vocab size**: 50,000 tokens
- **Rationale**: Vocabulary will reflect journalism language, not generic web text. 50k gives room for journalism-specific terms.

### 5. Model Architecture
- **Baseline**: Start with nanoGPT-style transformer (decoder-only, standard attention)
- **Experimental direction**: Overall structure experiments once baseline works
- **Reference**: [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- **Rationale**: Prove the pipeline with something known to work, then diverge

### 6. Target Tasks
- **Summaries**: Condense articles, documents, meeting notes
- **Citations**: Track sources, generate proper attribution, link claims to evidence
- **Newsworthiness**: Evaluate what makes something newsworthy, prioritize story angles
- **Rationale**: Focused scope = can build targeted fine-tuning data and evaluation

### 7. Core Values (Non-Negotiable)
These are baked in from day one, not bolted on after:

- **Truthfulness above all**: Never fabricate. Always cite. Acknowledge uncertainty. Say "I don't know" when you don't. This is journalism's first obligation.
- **Harm reduction**: Consider impact on sources, subjects, communities. "Minimize harm" from the SPJ code of ethics.
- **Accountability**: Explain reasoning. Be correctable. Transparent about limitations. Show your work.

**How these get operationalized:**
- Training data selection: Exclude fabricated/retracted content
- RLHF labeling: Reward hedging uncertainty, penalize confident bullshit
- Evaluation: Test for hallucination, source attribution, harm awareness

### 8. Alignment / RLHF Strategy
- **Method**: Human feedback from real journalists
- **Labeling tool**: Argilla (open source, self-hosted)
- **Process**:
  1. Generate outputs from base model
  2. Journalists rate/rank responses on truthfulness, harm, accountability
  3. Train reward model on preferences
  4. PPO to align base model to reward model
- **Rationale**: No shortcuts on values. Actual journalists, not mechanical turkers.

### 9. Data Ingestion
- **Initial source**: Times of San Diego via existing newswell-studio infrastructure
- **Existing code**: `~/client-code/newswell/newswell-studio/server/api/wordpress/articles.ts`
- **Already handles**: HTML stripping, content extraction, metadata (categories, tags, authors), word counts
- **Credentials**: Stored in newswell-studio `.env` (`WORDPRESS_URL`, `WORDPRESS_USER`, `WORDPRESS_APP_PASSWORD`)
- **Bonus**: Existing `newsworthiness.ts` has interface for plugging in ML algorithms - perfect for our model

### 11. Licensing
- **Approach**: Open for journalism only
- **Model**: Custom license allowing use only for journalism/news purposes
- **Weights**: Publicly available with usage restrictions
- **Rationale**: Want the work to benefit journalism, not be weaponized for misinfo or corporate PR

### 12. Multi-Publisher Strategy
- **Model**: Federated / bring-your-own-data
- **How it works**:
  1. Publishers run training locally on their own archives
  2. Share only model weight updates (not raw data)
  3. Central coordinator aggregates weight updates
- **Benefits**: Data never leaves publisher, no legal pooling issues, each org retains ownership
- **Inspiration**: Federated learning (like what Google uses for keyboard prediction)

### 13. Training Framework
- **Framework**: PyTorch + HuggingFace Transformers
- **Why**: Best ecosystem, most documentation, easiest debugging
- **Training library**: HuggingFace `transformers` + `accelerate` for distributed training

### 14. Context Length
- **Target**: 8K+ tokens
- **Rationale**: Journalism involves long articles, documents, background material
- **Trade-off**: Higher memory requirements, will need attention efficiency techniques

### 15. Experiment Tracking
- **Tool**: MLflow
- **Why**: Open source, self-hostable, aligns with data sovereignty ethos
- **Hosting**: Self-hosted alongside training infrastructure

### 16. Base Model
- **Choice**: OpenAI gpt-oss-20b (Apache 2.0)
- **Why**:
  - Runs on 16GB RAM (perfect for federated local training)
  - Apache 2.0 license (journalism-only restriction still applies to our fine-tune)
  - Released August 2025, state-of-the-art for size
- **Approach**: Continue pretraining on journalism corpus, then fine-tune for tasks, then RLHF

---

### 17. Evaluation Strategy
Two-pronged, balanced:

**Task-specific benchmarks** (automated, fast):
- Summarization: ROUGE scores against human summaries
- Citations: Precision/recall on source attribution
- Newsworthiness: Classification accuracy on labeled examples

**Human evaluation** (gold standard, slower):
- Blind A/B comparisons by working journalists
- Rubrics for truthfulness, harm awareness, usefulness
- Periodic eval rounds, not continuous

---

## Implementation Roadmap

### Phase 1: Data Pipeline
**Goal**: Get journalism data flowing into training-ready format

- [ ] Set up project structure (Python, pyproject.toml, Modal config)
- [ ] Create data extraction script using newswell-studio WordPress API
- [ ] Pull full Times of San Diego archive (~15k articles)
- [ ] Clean and deduplicate
- [ ] Format for continued pretraining (plain text with article boundaries)
- [ ] Compute corpus statistics (token count, vocabulary analysis)
- [ ] Store in HuggingFace datasets format

**Output**: `data/tosd_corpus/` with training-ready text files

### Phase 2: Training Infrastructure
**Goal**: Get gpt-oss-20b running on Modal with our data

- [ ] Set up Modal.com account and GPU quota
- [ ] Create Modal training script with HuggingFace Transformers
- [ ] Download and verify gpt-oss-20b weights
- [ ] Test inference on sample journalism prompts
- [ ] Set up MLflow tracking
- [ ] Run small-scale continued pretraining test (1% of data)
- [ ] Verify loss curves look reasonable

**Output**: Working training pipeline, baseline model checkpoint

### Phase 3: Continued Pretraining
**Goal**: Adapt gpt-oss-20b to journalism domain

- [ ] Full continued pretraining run on TOSD corpus
- [ ] Monitor training metrics (loss, perplexity)
- [ ] Save checkpoints every N steps
- [ ] Evaluate on held-out journalism text
- [ ] Compare before/after on journalism-specific prompts

**Output**: `journo-llm-base` checkpoint

### Phase 4: Instruction Fine-tuning
**Goal**: Teach the model specific journalism tasks

- [ ] Create instruction datasets:
  - Summarization pairs (article → summary)
  - Citation extraction (text → sources cited)
  - Newsworthiness evaluation (article → score + reasoning)
- [ ] Format as instruction/response pairs
- [ ] Fine-tune on task mixture
- [ ] Evaluate on held-out task examples

**Output**: `journo-llm-instruct` checkpoint

### Phase 5: RLHF Alignment
**Goal**: Bake in journalism values through human feedback

- [ ] Deploy Argilla for labeling interface
- [ ] Generate candidate responses for labeling
- [ ] Recruit journalist labelers
- [ ] Collect preference data on:
  - Truthfulness (does it hedge uncertainty appropriately?)
  - Harm awareness (does it consider impact?)
  - Accountability (does it explain reasoning?)
- [ ] Train reward model on preferences
- [ ] PPO training to align model to reward model

**Output**: `journo-llm-aligned` checkpoint

### Phase 6: Evaluation & Benchmarks
**Goal**: Prove it works, measure against baselines

- [ ] Build journalism-specific eval suite:
  - Summarization benchmark (ROUGE + human eval)
  - Citation accuracy benchmark
  - Newsworthiness prediction benchmark
  - Hallucination detection test
- [ ] Run blind human eval with working journalists
- [ ] Compare to base gpt-oss-20b and other models
- [ ] Document results

**Output**: Benchmark suite, evaluation report

### Phase 7: Federated Infrastructure
**Goal**: Enable multi-publisher training

- [ ] Create lightweight local training package
- [ ] Build weight aggregation server
- [ ] Test federated round with simulated publishers
- [ ] Document onboarding process for new publishers
- [ ] Privacy audit of weight sharing

**Output**: Federated training toolkit

---

## Technical Notes

### Tokenizer
Since we're building on gpt-oss-20b, we use their tokenizer. We should analyze how well it handles journalism-specific vocabulary (datelines, AP style, legal terms) and document any gaps.
