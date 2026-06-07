from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a local PEFT LoRA adapter into a local base model.")
    parser.add_argument("--base-model-path", required=True, help="Local base model directory.")
    parser.add_argument("--adapter-path", required=True, help="Local PEFT LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for the merged model.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used while loading the base model.",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Shard size passed to save_pretrained.")
    parser.add_argument(
        "--unsafe-serialization",
        action="store_true",
        help="Save PyTorch .bin weights instead of safetensors.",
    )
    return parser.parse_args()


def require_local_dir(path: str, description: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {resolved}")
    return resolved


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()
    base_model_path = require_local_dir(args.base_model_path, "base model path")
    adapter_path = require_local_dir(args.adapter_path, "adapter path")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Merging requires local server dependencies: `transformers` and `peft`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        local_files_only=True,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=not args.unsafe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")


if __name__ == "__main__":
    main()
