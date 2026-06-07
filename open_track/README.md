# Open Track

This folder contains the extra-credit path for the course project.

## What is implemented

- New tools: `decompose_question`, `get_document`, and `verify_claim` are exposed from `agent/tools.py`.
- Multi-agent architecture: `agent/deep_research_agent.py` separates planning, search execution, answer synthesis, and verification.
- Candidate fusion: `agent/fuse_deep_research_runs.py` fuses several legal local-agent trajectories with a conservative evidence judge.
- Training path: `build_sft_data.py` converts successful trajectories into SFT data, and `train.py` fine-tunes from a local model path only. This path is implemented but is not used for the final submitted score.

## Final OpenTrack result

The final submitted OpenTrack run is `fused_v6_relaxed`:

- Baseline DeepResearch: `7/50 = 14%`
- OpenTrack relaxed fusion: `8/50 = 16%`
- Correct query ids: `5, 53, 159, 314, 380, 651, 1082, 1095`

The improvement comes from multi-trajectory candidate fusion. It does not use gold answers,
eval labels, hard-coded query ids, or SFT data from the evaluation set.

If the source trajectory files already exist in `runs/`, reproduce the final OpenTrack
submission and evaluation with:

```bash
python open_track/run.py \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1
```

By default this fuses:

- `runs/deep_research_submission_v6_docref.jsonl`
- `runs/deep_research_submission_v8_repeatcap.jsonl`
- `runs/deep_research_submission_v11_overnight.jsonl`
- `runs/deep_research_submission_v12_overnight_no_react_verify.jsonl`
- `runs/deep_research_submission_v13_overnight_heuristic.jsonl`
- `runs/deep_research_submission_v14_balanced_overnight.jsonl`

If one of these diagnostic trajectories is intentionally absent, pass `--skip-missing` for
an ablation-only run. The final submitted result should include the exact fused output and
eval file in `opentrack/eval/`.

## Optional SFT workflow

Generate trajectories for a legitimate training/development split first. Do not build SFT data from
`browsecomp_plus_hard50.jsonl` if it is the public-test or final scoring set in your course setting.

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_train.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_submission_train.jsonl
```

Evaluate:

```bash
python -m agent.eval \
  --submission runs/deep_research_submission_train.jsonl \
  --dataset browsecomp_plus_train.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_eval_train.jsonl
```

Build SFT data from successful trajectories:

```bash
python open_track/build_sft_data.py \
  --submission runs/deep_research_submission_train.jsonl \
  --eval-results runs/deep_research_eval_train.jsonl \
  --source-split train \
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

- Baseline trajectory: `runs/deep_research_submission_v6_docref.jsonl`.
- Baseline evaluation output: `runs/deep_research_eval_v6_docref.jsonl`.
- Final OpenTrack trajectory: `runs/deep_research_submission_fused_v6_relaxed.jsonl`.
- Final OpenTrack evaluation output: `runs/deep_research_eval_fused_v6_relaxed.jsonl`.
- SFT data summary: number of rows written by `build_sft_data.py`.
- Training setting: model path, LoRA/full fine-tuning choice, sequence length, batch size, epochs, and checkpoint path.
