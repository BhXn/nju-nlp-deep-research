from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Track SFT training entrypoint for local server models.")
    parser.add_argument("--model-path", required=True, help="Local model directory. The script never downloads models.")
    parser.add_argument("--train-file", default="open_track/data/sft_success.jsonl", help="Local SFT JSONL.")
    parser.add_argument("--output-dir", default="open_track/checkpoints/deepresearch-sft", help="Checkpoint output dir.")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def require_local_path(path: str, description: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    model_path = require_local_path(args.model_path, "model path")
    train_file = require_local_path(args.train_file, "train file")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training dependencies are missing. Install them on the server, for example: "
            "`pip install transformers datasets peft accelerate`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("LoRA training requires `peft`.") from exc
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def formatting_func(example: Dict[str, Any]) -> str:
        messages = example.get("messages")
        if messages and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                return str(example.get("text", ""))
        return str(example.get("text", ""))

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        text = formatting_func(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved trained artifacts to {output_dir}")


if __name__ == "__main__":
    main()
