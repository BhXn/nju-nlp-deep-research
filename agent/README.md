# vLLM 版教学 Agent

这个目录是面向课堂教学的 **精简版实现**，默认假设：

- 模型服务已经由 `vLLM` 启动
- 本地只需要提供：
  - `BrowseComp-Plus` 的 BM25 检索工具
  - 一个最小的 RAG 示例
  - 一个可供参考的完整 agent-loop notebook

## 教学分层

### 学生版

- notebook: [agent_vllm.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm.ipynb)
- 只展示：
  - 单次 `search`
  - 将检索结果拼进 prompt
  - 调用 vLLM 输出最终答案
- notebook: [agent_vllm_weather.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm_weather.ipynb)
  - 用本地模拟 `get_weather` 工具测试 vLLM + agent 的工具调用链路
  - 不依赖检索语料和外部 API

不提供完整 agent loop，让学生自己实现多轮工具调用。

### 标准答案版

- notebook: [agent_vllm_answer.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm_answer.ipynb)
- 展示：
  - OpenAI-compatible `tools`
  - 多轮 agent loop
  - `search` / `get_document` 两个更细粒度工具
  - 轨迹记录

## 目录说明

- [browsecomp_searcher.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/browsecomp_searcher.py)
  - 本地 SQLite FTS5 BM25 检索实现
- [build_bm25_index.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/build_bm25_index.py)
  - 服务器上预构建索引
- [vllm_client.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/vllm_client.py)
  - vLLM OpenAI-compatible client
- [tools.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/tools.py)
  - RAG 检索与 agent tools 定义
- [dataset_utils.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/dataset_utils.py)
  - 读取 `hard50` 测试集等辅助函数

## 服务器上的一次性准备

先构建 BM25 索引：

```bash
python -m agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

之后 notebook 只需要复用 `index_path`。

## 依赖

```bash
pip install -r agent/requirements.txt
```

最小依赖：

- `pyarrow`
- `python-dotenv`

## vLLM 说明

这一版不再自己实现模型服务层，直接依赖已经启动好的 vLLM OpenAI-compatible endpoint。

根据 vLLM 官方文档：

- 自动工具调用需要 `--enable-auto-tool-choice`
- 还需要为对应模型设置 `--tool-call-parser`
- 推理模型如果需要 reasoning 解析，还要匹配 `--reasoning-parser`
- 如果模型输出是自定义 tool 格式，还要通过 `--tool-parser-plugin` 注册本地 parser

参考文档：

- [Tool Calling - vLLM](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [Qwen3ReasoningParser - vLLM](https://docs.vllm.ai/en/latest/api/vllm/reasoning/qwen3_reasoning_parser/)

## Pangu / DeepDiver 兼容

`agent/pangu_tool_parser.py` 提供了一个给 Pangu / DeepDiver 使用的 vLLM tool parser，
默认识别：

- `[unused11] ... [unused12]` 中的 JSON tool call
- `[unused16] ... [unused17]` 中的 reasoning
- markdown 代码块中的 JSON tool call
- `tool` / `name` / `function` / `tool_name` 等常见字段写法

启动示例：

```bash
vllm serve ./openPangu-Embedded-7B-DeepDiver \
  --served-model-name pangu_auto \
  --enable-auto-tool-choice \
  --tool-parser-plugin agent/pangu_tool_parser.py \
  --tool-call-parser pangu_deepdiver \
  --chat-template agent/pangu_chat_template.jinja \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

教学场景下不再优先推荐 vLLM 内置 `pangu` parser，因为 DeepDiver 有时会输出
说明文字加 markdown JSON，而不是单一严格格式。自定义 parser 的作用就是在服务端
把这些差异抹平，尽量稳定地产生 OpenAI `tool_calls`。

如果你的 tokenizer 里已经带了可用的 tool template，也可以把 `--chat-template`
换成模型自带的模板设置。

## Qwen3-8B 兼容

如果你们要直接起 Qwen3-8B 服务，`agent/` 侧可以直接走 vLLM 内置的 Qwen3 parser：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

## 本仓库补充的 Deep Research Agent

新增文件：

- `agent/deep_research_agent.py`
  - 多 agent 架构：Planner、Search Executor、Answer Synthesizer、Verifier
  - 多轮检索 loop、停止条件、上下文压缩、证据验证
- `agent/run_deep_research.py`
  - 批量生成符合提交格式的 `submission.jsonl`
- `agent/fuse_deep_research_runs.py`
  - 读取多份合法 agent 轨迹，用候选融合和证据重判生成新的 `submission.jsonl`
- `open_track/`
  - 成功轨迹转 SFT 数据与本地模型微调脚本

服务器运行示例：

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_submission.jsonl
```

评估：

```bash
python -m agent.eval \
  --submission runs/deep_research_submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_eval.jsonl
```

可调参数：

- `--max-rounds`：初始检索后的 ReAct 轮数
- `--max-initial-queries`：规划阶段生成并执行的初始检索数
- `--top-k`：每次 BM25 检索返回文档数
- `--max-tool-calls-per-round` / `--max-total-tool-calls` / `--max-no-new-info-rounds`：停止条件和工具调用预算
- `--max-context-chars` / `--max-evidence-docs`：压缩后证据上下文预算
- `--snippet-max-chars` / `--doc-max-chars`：搜索摘要与打开文档的字符预算
- `--planner-max-tokens` / `--tool-max-tokens` / `--answer-max-tokens` / `--verifier-max-tokens`：不同子 agent 的生成长度上限
- `--query-focused-snippet`：改用 query 命中位置附近的搜索摘要，适合作为消融项，默认关闭
- `--prefer-heuristic-queries`：优先执行确定性拆解 query，适合作为消融项，默认关闭
- `--answer-audit`：最终答案写出前增加一次 answer-type 审查，主要针对证据已召回但抽错槽位的题
- `--answer-audit-min-confidence`：审查结果覆盖原答案所需最低置信度，默认 `70`
- `--enable-thinking`：允许 Qwen thinking 输出；默认关闭以提高工具调用格式稳定性
- `--no-model-planner` / `--no-model-verifier`：关闭规划或验证 LLM 子 agent，用确定性 fallback

answer-type 审查消融示例：

```bash
python -m agent.run_deep_research \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --answer-audit \
  --answer-audit-min-confidence 70 \
  --output runs/deep_research_submission_v10_answer_audit.jsonl
```

多轨迹候选融合示例：

```bash
python -m agent.fuse_deep_research_runs \
  --dataset browsecomp_plus_hard50.jsonl \
  --submission v6=runs/deep_research_submission_v6_docref.jsonl \
  --submission v8=runs/deep_research_submission_v8_repeatcap.jsonl \
  --submission v10=runs/deep_research_submission_v10_answer_audit.jsonl \
  --submission nr=runs/deep_research_submission_v7_no_react_verify.jsonl \
  --submission broader=runs/deep_research_submission_v7_broader.jsonl \
  --base-label v6 \
  --override-confidence 85 \
  --override-score-margin 20 \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/deep_research_submission_fused.jsonl
```

融合脚本只读取题目、已有 submission 的候选答案和检索证据，不读取 `eval` 结果或标准答案。
它适合在多个合法消融 run 已经生成后使用，用一个严格 judge 重新检查候选是否满足题干约束。
默认策略是保守融合：保留 `--base-label` 指定 run 的答案，只有 judge 的置信度、候选分数差距和
证据 docid 都满足阈值时才覆盖。对于基线 run 自己给出的答案，脚本还会额外提高覆盖门槛，
这样优先避免把当前最强单次运行里的正确答案改错。
