# llm-lab

一个**极简、可白板推导、本机即可跑通全链路**的大语言模型学习与面试复现仓库（灵感来自 `tinybrain` / `nanochat`）。

不依赖 `transformers` / `trl` 等重型框架，从零手写全部核心模块，在 Apple Silicon（MPS）或单张消费级 GPU 上几分钟即可完整跑通：

**Byte-level BPE 分词 → 从零预训练（WSD + Muon）→ 对话微调（SFT Loss Mask）→ 可验证奖励强化学习（Dr. GRPO）→ KV Cache 推理**

---

## 目录结构

```text
src/
  tokenizer.py   # 纯 Python 字节级 BPE（含控制符防注入隔离）
  model.py       # Decoder-Only Transformer（RMSNorm, RoPE, GQA, QK-Norm, SwiGLU/ReLU², KV Cache）
  muon.py        # Muon 优化器（Newton-Schulz 5 步正交化）
  utils.py       # Cosine / WSD 学习率调度、Muon+AdamW 混合优化器、ChatML 编码与掩码
  data_gen.py    # 零外部依赖的学习语料与可验证算术题生成器
  pretrain.py    # 阶段 1：Next-Token Prediction 预训练
  sft.py         # 阶段 2：SFT 对话微调（仅 assistant 回复及 <|im_end|> 算损失）
  rl_grpo.py     # 阶段 3：Dr. GRPO 强化学习（RLVR 可验证算术奖励）
  generate.py    # 带 KV Cache 的自回归推理
tests/
  test_core.py   # 分词 round-trip、RoPE 相对位置性质、KV Cache 等价性、SFT 掩码等单测
docs/
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

### 3. 一条龙跑通全流程（生成数据 → 预训练 → SFT → GRPO → 推理）

```bash
# 1. 生成语料并训练 768 词表的 Byte-BPE 分词器
python -m src.data_gen --out data --vocab-size 768

# 2. 从零预训练（默认 pico 预设，WSD 调度 + Muon 优化器）
python -m src.pretrain --preset pico --steps 200 --schedule wsd --optimizer muon

# 3. 监督微调（仅对 assistant 回复算损失）
python -m src.sft --init out/pretrain/ckpt.pt --steps 120

# 4. Dr. GRPO 强化学习（可验证算术奖励，观察 eval_acc 变化）
python -m src.rl_grpo --init out/sft/ckpt.pt --steps 40

# 5. 使用 KV Cache 推理测试
python -m src.generate --ckpt out/rl/ckpt.pt --prompt "计算 15 * 12 等于多少？请给出最终数字，格式为 #### 答案"
```

---

## 面试复习建议

建议按 `docs/interview_qa.md` 里的清单对照源码逐个过一遍，重点能讲清：
1. **RoPE** 为何能用绝对旋转表达相对位置差；
2. **GQA** 如何压缩 KV Cache 显存，以及 Prefill 与 Decode 阶段掩码差异；
3. **Muon** 为什么只作用于 2D 矩阵及 Newton-Schulz 的作用；
4. **WSD** 调度如何支持训练中途分支退火；
5. **SFT Loss Mask** 对齐偏移与 `<|im_end|>` 处理；
6. **GRPO / Dr. GRPO** 为什么去掉 Critic、去掉 std 归一化、并用常数长度分母消除 length bias。
