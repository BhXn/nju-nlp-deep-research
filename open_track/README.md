# Open Track

This folder contains the extra-credit path for the course project.

## What is implemented

- New tools: `decompose_question`, `open_doc`, `find_in_doc`, and `verify_claim` are exposed from `agent/tools.py`.
- Multi-agent architecture: `agent/deep_research_agent.py` separates planning, search execution, answer synthesis, and verification.
- Training path: `build_sft_data.py` converts successful trajectories into SFT data, and `train.py` fine-tunes from a local model path only.

## Suggested server workflow

Generate trajectories first:

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_submission.jsonl
```

Evaluate:

```bash
python -m agent.eval \
  --submission runs/deep_research_submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_eval.jsonl
```

Build SFT data from successful trajectories:

```bash
python open_track/build_sft_data.py \
  --submission runs/deep_research_submission.jsonl \
  --eval-results runs/deep_research_eval.jsonl \
  --output open_track/data/sft_success.jsonl
```

Run training with a model that already exists on the server:

```bash
python open_track/train.py \
  --model-path ./Qwen3-8B \
  --train-file open_track/data/sft_success.jsonl \
  --output-dir open_track/checkpoints/qwen3-deepresearch-lora \
  --lora \
  --bf16
```

Do not train on public-test or private-test data. Keep large generated training data and checkpoints on the cloud server if they are too large to submit.

## Ablation records for the report

For the OpenTrack report section, keep these artifacts after the server run:

- Baseline trajectory: `runs/submission.jsonl` from the original single-search notebook or equivalent baseline script.
- Multi-agent trajectory: `runs/deep_research_submission.jsonl`.
- Evaluation output: `runs/deep_research_eval.jsonl`.
- SFT data summary: number of rows written by `build_sft_data.py`.
- Training setting: model path, LoRA/full fine-tuning choice, sequence length, batch size, epochs, and checkpoint path.
