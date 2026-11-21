# llm-lab

**麻雀虽小，五脏俱全。** 一个从零手写 PyTorch 的迷你中文大模型实验仓：模型参数控制在 **1.9M ~ 22.9M**（MacBook Apple Silicon 上几分钟即可跑通全部流程），但完整复现了现代大模型从 **分词 → 架构 → 预训练 → 退火 → SFT → 领域专家 → 多老师在线蒸馏 → 强化学习 → 评测 → OpenAI 服务与聊天网页** 的整条技术栈。

---

## 目录

- [模型总览与“五脏”特性](#模型总览与五脏特性)
- [预设规格（词表 768）](#预设规格词表-768)
- [代码结构](#代码结构)
- [快速开始](#快速开始)
- [完整跑一遍（10 个阶段一条龙）](#完整跑一遍10-个阶段一条龙)
- [OpenAI 兼容服务与聊天网页](#openai-兼容服务与聊天网页)
- [单元测试与自检](#单元测试与自检)
- [分享与面试文档](#分享与面试文档)

---

## 模型总览与“五脏”特性

```text
token id ─┬─ Embedding ──────────────────────────┐
          └─ Engram 查表 ─┬─ 2-gram (质数哈希×头) ┼─▶ StreamEntry (注意力选源)
                          └─ 3-gram (质数哈希×头) ┘          │
                                                             ▼
                                                  4 条残差流 (B, T, 4, d)
                                                             │
       ┌──────────────── 第 i 层（重复 L 次）────────────────┐
       │  [指定层] EngramGate 门控记忆注入                    │
       │  pre_read → RMSNorm → 3:1滑动窗口GQA(RoPE/QK-Norm) → post_write + res(Sinkhorn) │
       │  pre_read → RMSNorm → ReLU²/SwiGLU 或 MoE(共享+TopK) → post_write + res(Sinkhorn) │
       └─────────────────────────────────────────────────────┘
                                                             │
                                                             ▼
                                           StreamExit → RMSNorm → Head → 稳定交叉熵
```

| 模块 | 实现特性 | 对应源码 |
|---|---|---|
| **Byte-BPE 分词器** | 纯 Python 字节级 BPE（永不 `<unk>`），普通输入禁止解析 `<|im_start|>` / `<|tool_call|>` 防注入 | [src/tokenizer.py](src/tokenizer.py) |
| **mHC 多流残差** | `n_streams=4`，注意力入口/出口，20 步对数域 Sinkhorn 约束流间混合矩阵为双随机矩阵 | [src/model.py](src/model.py) |
| **Engram 查表记忆** | 2/3-gram 多头互异质数模长哈希查表，支持逐词生成的历史缓存与层间门控注入 | [src/model.py](src/model.py) |
| **3:1 滑动窗口 GQA + YaRN** | 每 4 层前 3 层只看最近 `window` 个词，末层看全文；含 QK-Norm、RoPE + YaRN 长文本外推与 KV Cache | [src/model.py](src/model.py) |
| **MoE 混合专家** | `nano_moe` 预设：首层 Dense，其余层 `共享专家 + 8 选 2 路由专家`，无梯度偏置均衡 + 轻均衡损失 | [src/model.py](src/model.py) |
| **防塌缩稳定交叉熵** | 按 `arXiv:2610.31642 v3` 实现 `m = -LSE(z_j - z_t)`, `softplus(-m)`，支持分块算损与饱和率监控 | [src/loss.py](src/loss.py) |
| **Muon + WSD** | 二维矩阵使用 5 步 Newton-Schulz 正交化 `Muon`，其余参数用 `AdamW`；支持 WSD 分支退火 | [src/muon.py](src/muon.py), [src/pretrain.py](src/pretrain.py) |
| **SFT / LoRA + Merge** | ChatML + `<|tool_call|>` + `<think>` 格式，屏蔽非 assistant 损失；支持 LoRA 低秩微调与无损合并 | [src/sft.py](src/sft.py), [src/lora.py](src/lora.py) |
| **DPO 偏好对齐** | 无需 Reward Model 的离线偏好对 `(chosen, rejected)` 优化 | [src/dpo.py](src/dpo.py) |
| **MOPD 在线蒸馏** | 学生在线采样 + 按领域路由到数学/问答专家老师 + 全词表精确反向 KL + `--merge-init` 任务算术 | [src/distill.py](src/distill.py) |
| **Dr. GRPO 强化学习** | 去掉 Critic、去掉 std 缩放、常数长度分母；内置 `agent` 工具调用、`instruct`、`choice` 及可选 RLAIF `--judge-url` | [src/rl_grpo.py](src/rl_grpo.py) |

---

## 预设规格（词表 768）

| 预设 | 宽 × 层 | 总参数量 | 核心配置亮点 |
|---|---|---:|---|
| `pico` | 192 × 4 | **1.91M** | mHC(4流) + Engram + 3:1 滑动窗口(128)，1 分钟跑通全部 10 个阶段 |
| `nano` | 384 × 6 | **8.95M** | mHC(4流) + Engram + 3:1 滑动窗口(256)，M5 本机主力实验档 |
| `nano_moe` | 384 × 6 | **11.67M** | 在 `nano` 基础上第 1–5 层启用 `8 选 2` 路由专家 + 共享专家 |
| `small` | 512 × 8 | **22.93M** | 更宽更深、双层 Engram 门控注入，适合单卡 GPU 或长时间实验 |

---

## 代码结构

```text
src/
  tokenizer.py   # 纯 Python 字节级 BPE（含控制符防注入隔离）
  loss.py        # 防 Softmax 塌缩稳定交叉熵、分块线性交叉熵、饱和率监控
  model.py       # MiniLLM（mHC + Engram + 3:1 滑动窗口 GQA + MoE + DecodeState）
  muon.py        # Muon 优化器（Newton-Schulz 5 步正交化）
  utils.py       # Cosine / WSD 调度、Muon+AdamW 混合优化器、ChatML+Tool 编解码与存档
  data_gen.py    # 预训练、通用/专家 SFT、DPO 偏好对、MOPD 提示、多题型 RLVR 语料生成器
  lora.py        # LoRA 低秩旁路注入、增量保存与无损合并
  pretrain.py    # 预训练、混入对话退火、--cooldown-from WSD 分支退火、--overfit 自检
  sft.py         # 通用 SFT、领域专家 SFT 与 LoRA 微调（--lora-rank）
  dpo.py         # DPO 离线偏好对齐（chosen vs rejected + ref_model）
  distill.py     # MOPD 多老师在线蒸馏（全词表反向 KL + --merge-init 任务算术）
  rl_grpo.py     # Dr. GRPO（Agent 多轮 calc 工具调用 + Instruct 约束 + Choice + 可选 RLAIF）
  eval.py        # 参数拆解、留出集损失、多题型得分与样本抽检
  generate.py    # 流式/批量 KV Cache 生成、Agent 多轮交互、开/关 KV Cache 测速对比
  serve.py       # OpenAI 兼容 HTTP 服务（含 SSE 流式、tool_calls、enable_thinking 与 reasoning_content）
  web/chat.html  # 零外部依赖的本地聊天/续写网页（支持思考过程展示）
tests/
  test_core.py   # 12 项核心等价性与数学性质单测
docs/
  team_sharing.md # 写给 Agent 研发团队的 30 分钟内部分享讲稿
  interview_qa.md # 对照源码的 LLM 面试高频问题速查
```

---

## 快速开始

建议使用 `uv` 管理 Python 3.12 环境（自动适配 Apple MPS / NVIDIA CUDA / CPU）：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行 9 项核心单元测试
pytest -v
```

---

## 完整跑一遍（10 个阶段一条龙）

以下命令在 Apple Silicon（M5 32GB MPS）上可直接顺序跑通：

```bash
# 0. 生成多题型训练语料与 BPE 分词器
python -m src.data_gen --out data --vocab-size 768

# 1. 小批过拟合诊断（验证梯度与损失链路正常，40 步内 loss 可压到 < 0.01）
python -m src.pretrain --preset nano --overfit --steps 40

# 2. 预训练（可用 --preset nano 或 --preset nano_moe）
python -m src.pretrain --preset nano --steps 200 --schedule wsd --optimizer muon

# 3. 退火（--cooldown-from 继承 WSD 恒定段优化器状态分支退火，混入 10% 整段算损对话）
python -m src.pretrain --cooldown-from out/pretrain/ckpt.pt --mix-chat data/sft.jsonl --steps 40

# 4. 通用 SFT、LoRA 微调与领域专家 SFT
python -m src.sft --steps 250
python -m src.sft --init out/sft/ckpt.pt --lora-rank 8 --steps 60 --out out/sft_lora/ckpt.pt
python -m src.sft --data data/sft_expert_math.jsonl --steps 100
python -m src.sft --data data/sft_expert_qa.jsonl --steps 100

# 5. DPO 偏好对齐 & MOPD 多老师在线蒸馏
python -m src.dpo --init out/sft/ckpt.pt --steps 40
python -m src.distill --merge-init --steps 20

# 6. Dr. GRPO 多题型强化学习（可加 --judge-url http://localhost:8080/v1 启用 RLAIF）
python -m src.rl_grpo --steps 25

# 7. 统一评测、多轮 Agent 演示 & KV Cache 测速对比
python -m src.eval out/rl/ckpt.pt
python -m src.generate --ckpt out/rl/ckpt.pt --mode agent
python -m src.generate --ckpt out/rl/ckpt.pt --mode bench-cache
```

---

## OpenAI 兼容服务与聊天网页

```bash
python -m src.serve out/rl/ckpt.pt --port 8001
# 浏览器打开 http://127.0.0.1:8001/
```

提供接口：
- `GET /`：离线聊天 / 续写双模式网页（不依赖外部 CDN）
- `GET /v1/models`、`GET /v1/info`：查看当前加载阶段、步数与参数量
- `POST /v1/chat/completions`：支持 `"stream": true` SSE 流式推送与 `tools` / `tool_calls`
- `POST /v1/completions`：基座续写

---

## 单元测试与自检

`pytest -v` 包含 12 项核心测试（[tests/test_core.py](tests/test_core.py)）：
1. BPE 无损往返编码与普通文本防控制符注入隔离
2. 防 Softmax 塌缩稳定交叉熵在高置信度（$m=25$）下梯度非零且全词表和为 0
3. mHC `sinkhorn` 双随机矩阵行和与列和守恒
4. RoPE 旋转位置编码的相对位置内积不变性
5. **同时开启 mHC + Engram + 3:1 滑动窗口 + MoE 时，带 `DecodeState` 逐词解码与全量前向严格等价**
6. MoE `balance_bias` 仅在优化器步后更新（不干扰激活重计算）
7. SFT Loss Mask 屏蔽 `user` / `tool` 返回及工具调用 JSON 解析
8. MOPD 反向 KL 与 CA-MOPD 任务算术均值融合
9. Muon Newton-Schulz 正交化、WSD 调度、多题型 RLVR 判分与 OpenAI 消息转换
10. LoRA $B$ 全零初始化等价性与无损权重合并（merge 前后前向严格一致）
11. DPO 偏好对损失单调性与隐式奖励准确率
12. YaRN RoPE 外推缩放与 `<think>` 自适应思考开关解析

---

## 分享与面试文档

- **交互式 HTML 原理图解（代码 × 动画对照，推荐）**：[docs/visual_guide.html](docs/visual_guide.html)（也可在 `python -m src.serve` 启动后访问 `/guide`）
- 写给 Agent 研发团队的内部分享讲稿：[docs/team_sharing.md](docs/team_sharing.md)
- 飞书学习与踩坑复盘记录：[docs/feishu_learning_log.md](docs/feishu_learning_log.md)
- 对照源码的 LLM 面试原理速查：[docs/interview_qa.md](docs/interview_qa.md)
