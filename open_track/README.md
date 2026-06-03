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
  --max-tool-chars 1000 \
  --output open_track/data/sft_success.jsonl
```

If you have several legal development runs, build one SFT file per run and merge them:

```bash
python open_track/merge_sft_data.py \
  --inputs \
    open_track/data/sft_success_a.jsonl \
    open_track/data/sft_success_b.jsonl \
    open_track/data/sft_success_c.jsonl \
  --output open_track/data/sft_success_merged.jsonl \
  --dedupe-by query_id
```

Install the optional Open Track dependencies on the server:

```bash
pip install -r open_track/requirements.txt
```

Run training with a model that already exists on the server:

```bash
python open_track/train.py \
  --model-path ./Qwen3-8B \
  --train-file open_track/data/sft_success_merged.jsonl \
  --output-dir open_track/checkpoints/qwen3-deepresearch-lora \
  --lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --learning-rate 1e-5 \
  --num-train-epochs 1 \
  --max-seq-length 8192 \
  --gradient-accumulation-steps 8 \
  --bf16
```

`train.py` uses assistant-only loss by default, so the model learns tool calls and final answers without being trained to copy user prompts or tool outputs. Do not train on public-test or private-test data. If `hard50` is the final scoring set in your course setting, use it for diagnosis only, not as SFT training data. Keep large generated training data and checkpoints on the cloud server if they are too large to submit.

Serve the LoRA adapter with vLLM:

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_base \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --enable-lora \
  --lora-modules qwen_deepresearch=open_track/checkpoints/qwen3-deepresearch-lora \
  --enforce-eager \
  --max-model-len 16384 \
  --host 0.0.0.0 \
  --port 8000
```

When the runtime LoRA server starts successfully, evaluate with the LoRA module name `qwen_deepresearch`. Keeping `qwen` in the served name lets the unchanged evaluator apply its Qwen no-thinking payload.

If the Ascend/vLLM runtime LoRA path fails during model profiling or Torch Dynamo compilation, merge the adapter into the base model and serve the merged checkpoint as a normal model. The merged server also uses the served name `qwen_deepresearch`.

```bash
python open_track/merge_lora.py \
  --base-model-path ./Qwen3-8B \
  --adapter-path open_track/checkpoints/qwen3-deepresearch-lora \
  --output-dir open_track/checkpoints/qwen3-deepresearch-merged \
  --dtype bfloat16

vllm serve open_track/checkpoints/qwen3-deepresearch-merged \
  --served-model-name qwen_deepresearch \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 16384 \
  --host 0.0.0.0 \
  --port 8000
```

Evaluate whichever LoRA deployment starts successfully:

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_deepresearch \
  --base-url http://127.0.0.1:8000/v1 \
  --max-context-chars 18000 \
  --max-evidence-docs 20 \
  --planner-max-tokens 512 \
  --tool-max-tokens 512 \
  --answer-max-tokens 768 \
  --verifier-max-tokens 512 \
  --output runs/deep_research_submission_lora.jsonl

python -m agent.eval \
  --submission runs/deep_research_submission_lora.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_deepresearch \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_eval_lora.jsonl
```

## Ablation records for the report

For the OpenTrack report section, keep these artifacts after the server run:

- Baseline trajectory: `runs/submission.jsonl` from the original single-search notebook or equivalent baseline script.
- Multi-agent trajectory: `runs/deep_research_submission.jsonl`.
- Evaluation output: `runs/deep_research_eval.jsonl`.
- SFT data summary: number of rows written by `build_sft_data.py`.
- Training setting: model path, LoRA/full fine-tuning choice, sequence length, batch size, epochs, and checkpoint path.
