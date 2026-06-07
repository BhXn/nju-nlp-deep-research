# OpenTrack 说明

本目录包含本项目的 OpenTrack 部分。最终提交不采用微调模型，而是采用多条合法本地 agent 轨迹的候选融合。

## 最终结果

- 基础 DeepResearch baseline：`7/50 = 14%`
- OpenTrack relaxed fusion：`8/50 = 16%`
- OpenTrack 正确题 query id：`5, 53, 159, 314, 380, 651, 1082, 1095`

最终 OpenTrack 版本为 `fused_v6_relaxed`。该结果来自多轨迹候选融合，不使用 gold answer、eval 标签、硬编码题号或从评测集构造的 SFT 数据。

## 实现内容

- `agent/deep_research_agent.py`：多轮 DeepResearch agent，包含规划、检索执行、证据状态维护、答案综合和验证。
- `agent/tools.py`：本地工具接口，包括 `search`、`get_document`、`decompose_question` 和 `verify_claim`。
- `agent/fuse_deep_research_runs.py`：OpenTrack 候选融合脚本，从多条本地轨迹中抽取候选答案和证据片段，并调用本地模型作为 judge。
- `run.py`：最终 OpenTrack 复现入口，默认执行 relaxed fusion、eval 和 analysis。
- `build_sft_data.py`、`train.py`、`merge_lora.py`：合法 SFT/LoRA 流程的辅助脚本。本次最终分数不使用该训练路径。

## 复现方式

先启动 Qwen3-8B 的 vLLM 服务：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

如果 `runs/` 下已有默认 source submission，或者提交包内存在 `opentrack/source_runs/`，可以一行复现最终 OpenTrack 输出、评测和分析：

```bash
python opentrack/run.py \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1
```

如果是在源码仓库的 `open_track/` 目录结构下运行，将命令中的 `opentrack/run.py` 改为 `open_track/run.py` 即可。

默认参与融合的 source submission 为：

- `runs/deep_research_submission_v6_docref.jsonl`
- `runs/deep_research_submission_v8_repeatcap.jsonl`
- `runs/deep_research_submission_v11_overnight.jsonl`
- `runs/deep_research_submission_v12_overnight_no_react_verify.jsonl`
- `runs/deep_research_submission_v13_overnight_heuristic.jsonl`
- `runs/deep_research_submission_v14_balanced_overnight.jsonl`

默认输出：

- `runs/deep_research_submission_fused_v6_relaxed.jsonl`
- `runs/deep_research_eval_fused_v6_relaxed.jsonl`
- `runs/deep_research_analysis_fused_v6_relaxed.json`

如果只想用当前存在的 source submission 做消融，可以添加 `--skip-missing`。

## 提交文件

最终 OpenTrack 结果应放在 `opentrack/eval/` 下：

- `231300088-赵旭令-opentrack-submission-acc16.jsonl`
- `eval.txt`

其中 `eval.txt` 对应课程评估脚本输出的 JSONL 评测结果。

## 数据泄漏说明

融合脚本只读取题目、已有 submission 轨迹、工具输出和模型自身产生的候选，不读取 `eval` 文件中的标准答案或评测标签。评测结果只用于事后统计和报告分析。

SFT 辅助脚本强制要求声明 `--source-split train` 或 `--source-split dev`；如果声明为 `test`，脚本会拒绝运行。本次没有使用 hard50 成功轨迹或 BrowseComp-Plus 剩余评估数据进行训练。
