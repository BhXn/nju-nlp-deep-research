from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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
    parser.add_argument(
        "--truncate-side",
        choices=["left", "right"],
        default="left",
        help="Which side to drop when a trajectory exceeds max_seq_length.",
    )
    parser.add_argument(
        "--train-on-all-tokens",
        action="store_true",
        help="Use full-sequence loss. Default trains only on assistant messages.",
    )
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
            Trainer,
            TrainingArguments,
        )
        import torch
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
        model.config.use_cache = False

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

    def message_payload(message: Dict[str, Any]) -> str:
        pieces: List[str] = []
        content = message.get("content")
        if content:
            pieces.append(str(content))
        if message.get("tool_calls"):
            pieces.append(json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=False))
        return "\n".join(pieces)

    def encode_piece(text: str) -> List[int]:
        if not text:
            return []
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def build_assistant_only_features(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids: List[int] = []
        labels: List[int] = []

        for message in messages:
            role = str(message.get("role", "unknown"))
            payload = message_payload(message)
            is_assistant = role == "assistant"

            segments = [
                (f"<|im_start|>{role}\n", False),
                (payload, is_assistant),
                ("\n<|im_end|>\n", is_assistant),
            ]
            for text, train_segment in segments:
                token_ids = encode_piece(text)
                input_ids.extend(token_ids)
                labels.extend(token_ids if train_segment else [-100] * len(token_ids))

        return {"input_ids": input_ids, "labels": labels}

    def formatting_func(example: Dict[str, Any]) -> str:
        messages = example.get("messages")
        if messages and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                return str(example.get("text", ""))
        return str(example.get("text", ""))

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example.get("messages")
        if messages and not args.train_on_all_tokens:
            features = build_assistant_only_features(messages)
        else:
            text = formatting_func(example)
            input_ids = encode_piece(text)
            features = {"input_ids": input_ids, "labels": list(input_ids)}

        input_ids = features["input_ids"]
        labels = features["labels"]
        if len(input_ids) > args.max_seq_length:
            if args.truncate_side == "left":
                input_ids = input_ids[-args.max_seq_length :]
                labels = labels[-args.max_seq_length :]
            else:
                input_ids = input_ids[: args.max_seq_length]
                labels = labels[: args.max_seq_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "num_train_labels": sum(1 for label in labels if label != -100),
        }

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data",
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: example["num_train_labels"] > 0,
        desc="Filtering empty assistant targets",
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["num_train_labels"])

    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        pad_token_id = tokenizer.pad_token_id

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

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
