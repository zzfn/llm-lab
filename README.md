# llm-lab

一个**极简、可白板推导、本机即可跑通全链路**的大语言模型学习与 Agent 原理分享仓库（灵感来自 `tinybrain` / `nanochat`）。

不依赖 `transformers` / `trl` 等重型框架，从零手写全部核心模块，在 Apple Silicon（MPS）或单张消费级 GPU 上几分钟即可完整跑通：

**Byte-level BPE 分词 → 从零预训练（WSD + Muon）→ 多轮 Tool-Calling SFT（Loss Mask）→ Agent Dr. GRPO 强化学习 → KV Cache 多轮工具调用推理**

---

## 目录结构

```text
src/
  tokenizer.py   # 纯 Python 字节级 BPE（含控制符防注入隔离）
  model.py       # Decoder-Only Transformer（RMSNorm, RoPE, GQA, QK-Norm, SwiGLU/ReLU², KV Cache）
  muon.py        # Muon 优化器（Newton-Schulz 5 步正交化）
  utils.py       # Cosine / WSD 学习率调度、Muon+AdamW 混合优化器、ChatML+Tool 编码与解析
  data_gen.py    # 零外部依赖的多轮工具调用语料与 RLVR 任务生成器
  pretrain.py    # 阶段 1：Next-Token Prediction 预训练
  sft.py         # 阶段 2：SFT 对话微调（屏蔽 user 与 tool 返回，只训 assistant 动作与 <|im_end|>）
  rl_grpo.py     # 阶段 3：Agent 多轮工具调用 + Dr. GRPO 强化学习
  generate.py    # 带 KV Cache 的自回归推理、Agent 多轮循环演示与 KV Cache 测速对比
tests/
  test_core.py   # 分词防注入、RoPE 相对位置性质、KV Cache 等价性、SFT 掩码、工具解析等单测
docs/
  team_sharing.md # 写给 Agent 研发团队的 30 分钟内部分享讲稿与现场 Demo 指南
  interview_qa.md # 对照源码的 LLM 面试高频问题速查
```

---

## 快速开始

### 1. 安装环境（推荐使用 `uv`）

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. 运行单元测试（验证核心数学与工程等价性）

```bash
pytest -v
```

### 3. 一条龙跑通 Agent 全流程

```bash
# 1. 生成多轮计算器工具调用语料并训练 768 词表的 Byte-BPE 分词器
python -m src.data_gen --out data --vocab-size 768

# 2. 从零预训练（nano 10.03M 预设，WSD 调度 + Muon 优化器）
python -m src.pretrain --preset nano --steps 200 --batch-size 16 --seq-len 128 --schedule wsd --optimizer muon

# 3. 监督微调（学会 <|tool_call|> 格式与 <|im_end|> 停止符，屏蔽 tool 返回损失）
python -m src.sft --init out/pretrain/ckpt.pt --steps 250 --lr-mult 0.2

# 4. Agent Dr. GRPO 强化学习（多轮执行 calc 工具并按最终答案与格式打分）
python -m src.rl_grpo --init out/sft/ckpt.pt --steps 25

# 5. 多轮工具调用推理演示 & KV Cache 加速对比
python -m src.generate --ckpt out/rl/ckpt.pt --mode agent
python -m src.generate --ckpt out/rl/ckpt.pt --mode bench-cache
```

---

## 推荐阅读

- 团队技术分享讲稿（面向 Agent 研发）：[docs/team_sharing.md](docs/team_sharing.md)
- 面试原理逐题对照：[docs/interview_qa.md](docs/interview_qa.md)
