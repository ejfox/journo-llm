"""
Modal.com training script for Journo LLM.

This script runs training jobs on Modal's serverless GPUs.
Supports continued pretraining and fine-tuning of gpt-oss-20b.
"""

import modal

# Modal app definition
app = modal.App("journo-llm")

# Docker image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "trl>=0.8.0",
        "mlflow>=2.10.0",
        "wandb>=0.16.0",  # backup tracking
        "bitsandbytes>=0.42.0",  # quantization
        "peft>=0.10.0",  # LoRA
        "sentencepiece>=0.2.0",
    )
    .run_commands("pip install flash-attn --no-build-isolation || true")
)

# Persistent volume for model checkpoints and data
volume = modal.Volume.from_name("journo-llm-data", create_if_missing=True)

VOLUME_PATH = "/data"
MODEL_ID = "openai/gpt-oss-20b"


@app.function(
    image=training_image,
    gpu="A100",  # 80GB for full training, can use A10G for testing
    timeout=60 * 60 * 12,  # 12 hours max
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],  # HF_TOKEN
)
def train_continued_pretraining(
    corpus_path: str = f"{VOLUME_PATH}/corpus.jsonl",
    output_dir: str = f"{VOLUME_PATH}/checkpoints/journo-llm-base",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 8192,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 500,
    use_lora: bool = True,  # LoRA for memory efficiency
    lora_r: int = 64,
    lora_alpha: int = 128,
):
    """
    Continue pretraining gpt-oss-20b on journalism corpus.

    This adapts the base model to journalism domain.
    """
    import json
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    print(f"Loading model: {MODEL_ID}")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # Apply LoRA for efficient fine-tuning
    if use_lora:
        print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load corpus
    print("Loading corpus...")
    articles = []
    with open(corpus_path) as f:
        for line in f:
            record = json.loads(line)
            # Format for pretraining
            text = f"# {record['title']}\n\n{record['content']}<|endofarticle|>"
            articles.append({"text": text})

    dataset = Dataset.from_list(articles)
    print(f"Loaded {len(dataset)} articles")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        report_to=["mlflow"],
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Commit volume changes
    volume.commit()

    return {"status": "completed", "output_dir": output_dir}


@app.function(
    image=training_image,
    gpu="A100",
    timeout=60 * 60 * 6,  # 6 hours
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_instruction_finetuning(
    base_model_path: str = f"{VOLUME_PATH}/checkpoints/journo-llm-base",
    instruction_data_path: str = f"{VOLUME_PATH}/instructions.jsonl",
    output_dir: str = f"{VOLUME_PATH}/checkpoints/journo-llm-instruct",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
):
    """
    Instruction fine-tune the model for journalism tasks.

    Expects JSONL with {"instruction": "...", "response": "..."} format.
    """
    import json

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"Loading base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load instruction data
    instructions = []
    with open(instruction_data_path) as f:
        for line in f:
            record = json.loads(line)
            # Format as chat
            text = f"### Instruction:\n{record['instruction']}\n\n### Response:\n{record['response']}"
            instructions.append({"text": text})

    dataset = Dataset.from_list(instructions)
    print(f"Loaded {len(dataset)} instruction pairs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=4096,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    volume.commit()
    return {"status": "completed", "output_dir": output_dir}


@app.function(
    image=training_image,
    gpu="A10G",  # Smaller GPU for inference
    timeout=60 * 10,
    volumes={VOLUME_PATH: volume},
)
def inference(
    model_path: str = f"{VOLUME_PATH}/checkpoints/journo-llm-instruct",
    prompt: str = "Summarize the following article:",
    max_new_tokens: int = 512,
):
    """Run inference on trained model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    result = pipe(prompt)
    return result[0]["generated_text"]


@app.function(
    image=training_image,
    volumes={VOLUME_PATH: volume},
)
def upload_corpus(corpus_data: bytes, filename: str = "corpus.jsonl"):
    """Upload corpus data to Modal volume."""
    import os

    filepath = f"{VOLUME_PATH}/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(corpus_data)

    volume.commit()
    return {"status": "uploaded", "path": filepath, "size": len(corpus_data)}


@app.local_entrypoint()
def main(
    action: str = "train",
    corpus_path: str = "data/tosd_corpus/corpus.jsonl",
):
    """
    CLI entrypoint for Modal training.

    Usage:
        modal run journo_llm/train_modal.py --action upload --corpus-path data/corpus.jsonl
        modal run journo_llm/train_modal.py --action train
        modal run journo_llm/train_modal.py --action inference --prompt "Summarize:"
    """
    if action == "upload":
        print(f"Uploading corpus from {corpus_path}")
        with open(corpus_path, "rb") as f:
            data = f.read()
        result = upload_corpus.remote(data)
        print(f"Uploaded: {result}")

    elif action == "train":
        print("Starting continued pretraining...")
        result = train_continued_pretraining.remote()
        print(f"Training complete: {result}")

    elif action == "finetune":
        print("Starting instruction fine-tuning...")
        result = train_instruction_finetuning.remote()
        print(f"Fine-tuning complete: {result}")

    elif action == "inference":
        import sys
        prompt = sys.argv[-1] if len(sys.argv) > 1 else "Summarize the following article:"
        print(f"Running inference with prompt: {prompt}")
        result = inference.remote(prompt=prompt)
        print(f"Result:\n{result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: upload, train, finetune, inference")
