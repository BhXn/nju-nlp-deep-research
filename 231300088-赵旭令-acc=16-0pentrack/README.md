# 231300088-赵旭令-OpenTrack 提交说明

本提交包包含基础 DeepResearch 任务、OpenTrack 实现、评测数据、BrowseComp-Plus 本地语料、实验报告和已完成评测结果。唯一未包含的是 Qwen3-8B 模型权重，复现时需要在本目录下下载或放置 `Qwen3-8B/`。

## 目录结构

- `browsecomp_plus_hard50.jsonl`：本次评测数据。
- `browsecomp-plus-corpus/`：本地 BrowseComp-Plus 语料，用于构建 BM25 索引。
- `core/deepresearch.ipynb`：基础任务 notebook 入口，封装运行和评估命令。
- `core/agent/`：基础任务代码。命令行入口为 `python -m agent.run_deep_research`。
- `eval/`：基础任务已提交评测结果，baseline 为 `7/50 = 14%`。
- `opentrack/`：OpenTrack 代码与复现入口。入口为 `python opentrack/run.py`。
- `opentrack/source_runs/`：OpenTrack 多轨迹融合所需的合法 source submission。
- `opentrack/eval/`：OpenTrack 已提交评测结果，最终结果为 `8/50 = 16%`。
- `231300088-赵旭令-acc=16-opentrack.pdf`：实验报告 PDF。
- `report.tex`、`report.pdf`：报告源文件和同内容 PDF 备份。

## 快速检查已提交结果

如果只检查提交文件中的指标：

```bash
cat eval/eval.txt
cat opentrack/eval/eval.txt
```

其中基础任务结果为 `14.00% (7/50)`，OpenTrack 结果为 `16.00% (8/50)`。

## 从零复现

下面命令假设当前工作目录就是本提交包根目录：

```bash
cd 231300088-赵旭令-acc=16-0pentrack
```

如果已经在该目录中，可以跳过上面这行。

### 1. 准备 Python 依赖

课程平台通常已经安装 vLLM、torch-npu 等基础环境。若缺少本项目依赖，运行：

```bash
pip install -r core/agent/requirements.txt
pip install -r opentrack/requirements.txt
```

### 2. 下载或放置 Qwen3-8B

如果本目录下还没有 `Qwen3-8B/`，可以下载：

```bash
git clone https://atomgit.com/hf_mirrors/MindSpore-Lab/Qwen3-8B.git
```

检查模型配置文件是否存在：

```bash
ls Qwen3-8B/config.json
```

### 3. 构建 BM25 索引

基础任务 agent 需要 BM25 索引。语料已经在 `browsecomp-plus-corpus/`，从提交包根目录运行：

```bash
mkdir -p indexes runs

cd core
python -m agent.build_bm25_index \
  --corpus-path ../browsecomp-plus-corpus \
  --index-path ../indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
cd ..
```

如果 `indexes/browsecomp_plus_bm25.sqlite` 已经存在，可以跳过本步。

### 4. 启动本地模型服务

开一个终端，在提交包根目录运行：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

如果 NPU 显存不足，可以先清理旧进程：

```bash
npu-smi info
ps -ef | grep -E "vllm|python" | grep -v grep
```

确认旧 vLLM 进程无用后再杀掉对应 PID。若仍然 OOM，可用保守启动参数：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

服务启动后，另开一个终端继续下面步骤。

### 5. 复现基础任务 baseline

从提交包根目录运行：

```bash
cd core

python -m agent.run_deep_research \
  --dataset ../browsecomp_plus_hard50.jsonl \
  --index-path ../indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output ../runs/deep_research_submission_v6_docref.jsonl

python -m agent.eval \
  --submission ../runs/deep_research_submission_v6_docref.jsonl \
  --dataset ../browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output ../runs/deep_research_eval_v6_docref.jsonl

python -m agent.analyze_deep_research_run \
  --submission ../runs/deep_research_submission_v6_docref.jsonl \
  --eval ../runs/deep_research_eval_v6_docref.jsonl \
  --dataset ../browsecomp_plus_hard50.jsonl \
  --output ../runs/deep_research_analysis_v6_docref.json

cd ..
```

参考提交结果为 `7/50 = 14%`。课程评估脚本使用本地 8B 模型判断语义等价，可能存在小幅波动。

### 6. 复现 OpenTrack

从提交包根目录运行：

```bash
python opentrack/run.py \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1
```

该脚本会自动执行三步：

1. `agent.fuse_deep_research_runs`
2. `agent.eval`
3. `agent.analyze_deep_research_run`

默认输出：

- `runs/deep_research_submission_fused_v6_relaxed.jsonl`
- `runs/deep_research_eval_fused_v6_relaxed.jsonl`
- `runs/deep_research_analysis_fused_v6_relaxed.json`

`opentrack/run.py` 会优先读取 `runs/` 下的 source submission；如果不存在，会自动回退到包内的 `opentrack/source_runs/`。因此即使没有重新跑所有历史 source runs，也可以复现最终 fusion 流程。

参考提交结果为 `8/50 = 16%`，正确 query id 为：`5, 53, 159, 314, 380, 651, 1082, 1095`。

## 已提交结果文件

基础任务：

- `eval/231300088-赵旭令-submission-acc14.jsonl`
- `eval/eval.txt`
- `eval/analysis.json`

OpenTrack：

- `opentrack/eval/231300088-赵旭令-opentrack-submission-acc16.jsonl`
- `opentrack/eval/eval.txt`
- `opentrack/eval/analysis.json`

## 数据使用声明

本提交没有使用 hard50 的标准答案、评测标签或成功轨迹构造训练数据，也没有使用 BrowseComp-Plus 剩余评估数据训练模型。OpenTrack 融合只读取题目、已有 agent 轨迹、工具输出和模型自身产生的候选答案；eval 文件只用于事后统计和报告分析。
