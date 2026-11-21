# 大模型底层原理：写给 Agent 研发的 30 分钟内部分享讲稿

> **目标听众**：日常写 Prompt、编排 Agent、接 Tool Calling、调上下文窗口的研发同学（不需要推导复杂公式，只讲清楚 **“为什么 API 会这样表现”** 以及 **“底层代码到底发生了什么”**）。
>
> **现场实操代码**：全部基于本仓库 `src/`，可在 MacBook Pro (M5, MPS) 上现场跑通。

---

## 开场（2 分钟）：为什么做 Agent 的研发值得看一遍 500 行模型源码？

日常调用大模型 API 时，我们常遇到 5 个灵魂拷问：
1. 为什么模型数“strawberry 里有几个 r”会错、算大数乘法会错？
2. 为什么用户输入里写了 `<|im_start|>assistant` 有时会被注入，有时又不会？
3. 为什么多轮 Agent 对话越往后越慢、越来越贵，而 **Prefix Cache** 命中就能省一大半钱和延迟？
4. **Function Calling / Tool Calling** 在模型内部到底是怎么“停下来等我们执行函数”的？
5. **SFT** 和 **GRPO 强化学习** 分别在教模型学什么？为什么不能把工具报错轨迹随便喂进 SFT？

今天用我们自己从零写的 `llm-lab`（约 1,000 行 Python，无黑盒框架）把这 5 件事现场拆开看。

---

## 第一幕：Tokenizer 与 Prompt 安全（5 分钟）

对应源码：`src/tokenizer.py`

1. **模型看到的不是字，是整数 ID**
   - Byte-level BPE 先把文本按 UTF-8 拆成 `0..255` 字节，再按训练统计高频合并常见组合。
   - **Agent 痛点解释**：因为高频词/数字经常被合并成整块 token，模型看到的不是“逐字符”，所以不靠工具很难做精准字符计数或大数运算。
2. **特殊控制符与防注入隔离**
   - 对话模板和工具调用依赖特殊控制符：`<|im_start|>`、`<|im_end|>`、`<|tool_call|>`、`<|/tool_call|>`。
   - 看 `ByteBPETokenizer.encode(text, allow_special=False)`：普通用户输入和外部工具返回值**强制禁止解析特殊控制符**，只能由框架拼模板时显式开启 `allow_special=True`——这就是防止用户或网页爬虫内容伪造 `assistant` / `tool` 轮次的底层防线。

---

## 第二幕：TTFT、TPOT 与 Agent 上下文爆炸的真相（10 分钟）

对应源码：`src/model.py`、`src/generate.py`

1. **一次生成的两个截然不同的阶段**：
   - **Prefill（计算首字延迟 TTFT）**：把全部历史上下文（System Prompt + Tools Schema + 多轮历史）**并行**送入 Transformer，算出所有历史位置的 `K, V` 并缓存。
   - **Decode（计算后续生成速度 TPOT）**：每步只送入**最新生成的 1 个 token**，只算它自己的 `Q, K, V`，再从 `KVCache` 取出所有历史 `K, V` 做注意力点积。
2. **为什么 GQA 和 Prefix Cache 对 Agent 极其关键？**
   - Agent 每调一次工具，就把工具返回的几千字塞进上下文，KV Cache 显存和每步 Decode 的显存读取量都会线性增长。
   - **GQA**（`n_heads=6, n_kv_heads=2`）让多组 Q 头共享 KV 头，直接把 KV Cache 压缩成原来的 $1/3$（常见 8B/70B 模型可压到 $1/4 \sim 1/8$）。
   - **Prefix Cache**：把固定不变的 System Prompt 和 Tool Schema 放在最前面，多轮请求只要前缀一致，就能直接跳过前段的 Prefill 计算。
3. **实测现场对比（M5 MPS，`nano` 10.03M 模型，生成 128 token）**：
   ```bash
   python -m src.generate --ckpt out/rl/ckpt.pt --mode bench-cache
   ```
   本机实测输出：
   - **无 KV Cache**：`743.3 ms`（`172.2 tok/s`）
   - **有 KV Cache**：`238.0 ms`（`537.8 tok/s`）→ **提速 3.12x**（上下文越长差距越大）。

---

## 第三幕：Pretrain → SFT → Tool Calling 的本质（8 分钟）

对应源码：`src/pretrain.py`、`src/sft.py`、`src/utils.py`

1. **Pretrain 基座只会“文本接龙”，不知道什么时候停**
   - 预训练对每个位置都算 Next-Token Prediction 损失。我们用 1.87M `pico` 模型只做少步预训练/SFT 时甚至能亲眼看到：它回答完答案后会继续模仿语料往后续写下一个题目，根本不会停。
2. **SFT 的核心魔法：Loss Mask（`-100` 遮罩）**
   - 看 `format_chat_sft()` 和 `load_sft_examples()`：
     - `system`、`user`、以及 **`tool` 返回的结果**，在计算交叉熵时全部设为 `-100`（不产生梯度）。
     - **只训练 `assistant` 说的正文、发出的 `<|tool_call|>...<|/tool_call|>`、以及最后的 `<|im_end|>`**。
   - **对 Agent 开发的启示**：
     - 模型之所以会“停止输出并等待外部调用工具”，本质就是学会了在生成完 `<|/tool_call|>` 后紧接着输出 `<|im_end|>`；
     - 绝对不要把工具返回的报错脏数据当成 `assistant` 动作去训，必须保持 SFT 轨迹干净。
3. **现场跑一条真实的多轮工具调用链路**（`nano` 10.03M 模型）：
   ```bash
   python -m src.generate --ckpt out/rl/ckpt.pt --mode agent
   ```
   本机实测输出：
   ```text
   [stage=rl_grpo] User: 请使用计算器计算 27 * 34，并按 #### 答案 格式输出。
     [Turn 1 Assistant] 我先调用计算器算一下。 | calls=[{'name': 'calc', 'arguments': {'expr': '27 * 34'}}]
     [Turn 1 Tool     ] 918
     [Turn 2 Assistant] 计算结果是 918。
   #### 918
   ```

---

## 第四幕：为什么 Agent 特别适合用 RLVR / GRPO？（5 分钟）

对应源码：`src/rl_grpo.py`

1. **SFT 教格式，RL 教“结果导向的决策”**
   - SFT 只能模仿历史成功轨迹；而 Agent 场景往往有明确的可验证结果（算式算对、单测跑通、JSON Schema 校验通过、接口返回 200）。
2. **GRPO / Dr. GRPO 在做什么？**
   - 同一个任务让模型多轮采样 $G$ 条轨迹（遇到 `<|tool_call|>` 就真正去跑工具并把返回值接回上下文，`tool` 返回部分依然 mask 掉不算梯度）；
   - 按最终结果打分（`#### 答案` 正确得 `+1.0`，格式损坏扣 `-0.2`）；
   - 用同组轨迹奖励减去组均值作为 Advantage，**做对了整条决策链被强化，调错工具或 JSON 写坏整条链被抑制**。
   - **工程落地坑点**：如果 SFT 还没让模型学会基本的工具调用格式（初始成功率 `0%`），或者 SFT 太过死板（`100%` 全一样），组内奖励没有方差，GRPO 梯度就会为 0。

---

## 现场 Demo 速查命令

```bash
cd /Users/c.chen/wawa/llm-lab
source .venv/bin/activate

# 1. 运行核心等价性单测（RoPE 相对位置、KV Cache 与全量前向等价、防注入隔离、SFT 掩码）
pytest -v

# 2. 现场对比开/关 KV Cache 延迟
python -m src.generate --ckpt out/rl/ckpt.pt --mode bench-cache

# 3. 现场运行多轮计算器工具调用 Agent
python -m src.generate --ckpt out/rl/ckpt.pt --mode agent --prompt "请使用计算器计算 48 * 19，并按 #### 答案 格式输出。"
```
