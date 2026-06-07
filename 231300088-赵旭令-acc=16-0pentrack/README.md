# 231300088-赵旭令-OpenTrack 提交说明

本提交包包含基础 DeepResearch 任务、OpenTrack 实现、实验报告和已完成的评测结果。

## 目录结构

- `core/agent/`：基础任务代码。入口为 `python -m agent.run_deep_research`。
- `eval/`：基础任务评测结果。当前 baseline 为 `7/50 = 14%`。
- `opentrack/`：OpenTrack 代码与复现入口。入口为 `python opentrack/run.py`。
- `opentrack/source_runs/`：OpenTrack 多轨迹融合所需的合法 source submission。
- `opentrack/eval/`：OpenTrack 最终评测结果目录。
- `report.tex`、`report.pdf`：实验报告。

## 基础任务结果

基础任务使用 `v6_docref` 版本：

- submission：`eval/231300088-赵旭令-submission-acc14.jsonl`
- eval：`eval/eval.txt`
- analysis：`eval/analysis.json`
- accuracy：`7/50 = 14%`

## OpenTrack 结果

最终 OpenTrack 版本为 `fused_v6_relaxed`，服务器运行结果为：

- accuracy：`8/50 = 16%`
- correct query ids：`5, 53, 159, 314, 380, 651, 1082, 1095`

最终文件已放入 `opentrack/eval/`：

- `231300088-赵旭令-opentrack-submission-acc16.jsonl`
- `eval.txt`
- `analysis.json`

## 复现命令

先启动本地模型服务：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

基础任务示例：

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_submission_v6_docref.jsonl
```

OpenTrack 一行复现：

```bash
python opentrack/run.py \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1
```

`opentrack/run.py` 会优先读取 `runs/` 下的 source submission；如果不存在，会自动回退到包内的 `opentrack/source_runs/`。

## 数据使用声明

本提交没有使用 hard50 的标准答案、评测标签或成功轨迹构造训练数据，也没有使用 BrowseComp-Plus 剩余评估数据训练模型。OpenTrack 融合只读取题目、已有 agent 轨迹、工具输出和模型自身产生的候选答案；eval 文件只用于事后统计和报告分析。
