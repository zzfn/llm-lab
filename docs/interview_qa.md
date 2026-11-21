# 大模型手写项目面试题速查（对照本项目源码）

每道题都直接对应 `src/` 里的具体代码实现，建议结合源码逐行口述。

---

### 1. 分词器（对应 `src/tokenizer.py`）
- **Q: 为什么现在主流大模型都用 Byte-level BPE？遇到生僻字或乱码会 `<unk>` 吗？**
  - 先把字符串按 UTF-8 编码成 `0..255` 的字节序列，再迭代合并最高频相邻对。因为任意 Unicode 字符都能拆成 1~4 个字节，词表天然兜底，**永远不会产生 `<unk>`**。
- **Q: 对话控制符（如 `<|im_start|>`）如果出现在用户正文里，会不会触发提示词注入？**
  - 不会。见 `encode(text, allow_special=False)`：普通用户输入强制走 `encode_ordinary()`，把 `<|im_start|>` 当普通字符切分；只有构造模板的系统代码才开启 `allow_special=True` 写入控制符 ID。

---

### 2. 模型架构（对应 `src/model.py`）
- **Q: RMSNorm 和 LayerNorm 的区别？为什么要放在 Residual 之前（Pre-Norm）？**
  - `RMSNorm` 去掉了减均值和偏置项，只做均方根归一化 $x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$，少一次归约统计量、省显存；且代码里先转 `float32` 再归一化，防止半精度平方溢出。
  - **Pre-Norm** 把残差主干留成一条直通恒等通道，梯度可以无衰减地从顶层直达底层；Post-Norm 在深层容易梯度消失或爆炸。
- **Q: 推导一下 RoPE 为什么能用绝对位置编码表达相对位置？**
  - 把向量按相邻两维看成复数（或二维平面向量），位置 $m$ 乘以旋转矩阵 $R(m\theta)$。两个位置内积 $(R(m\theta)q)^\top (R(n\theta)k) = q^\top R((n-m)\theta) k$，只依赖相对距离 $n-m$（对应 `tests/test_core.py` 的 `test_rope_relative_position_property`）。
- **Q: GQA 是什么？能省多少 KV Cache 显存？**
  - 让 $H_q$ 个查询头分组共享 $H_{kv}$ 个键值头（本项目 `n_heads=8, n_kv_heads=2`）。KV Cache 显存公式为 $2 \times L \times B \times S \times H_{kv} \times d_{\text{head}} \times \text{bytes}$，GQA 直接把 KV Cache 显存与 Decode 阶段的访存带宽压缩为原来的 $H_{kv}/H_q$（即 $1/4$）。
- **Q: 为什么 QK-Norm 在现代小模型/长上下文训练中很常见？**
  - 随着训练步数增加，注意力 logits 容易出现异常增长导致 softmax 熵坍缩。在施加 RoPE 之前对 Q 和 K 的每个头做 `RMSNorm`，能把点积尺度稳定在可控范围内。
- **Q: Decode 阶段有 KV Cache 时，输入只有 1 个 token，还需要 Causal Mask 吗？**
  - **不需要**。此时 $T=1$，唯一的新 Query 位于当前最新位置，缓存里存的全是历史和当前 K/V，它本来就应该关注全部历史；只有 Prefill（$T>1$）时才需要下三角掩码（见 `CausalSelfAttention.forward`）。

---

### 3. 优化器与学习率调度（对应 `src/muon.py`, `src/utils.py`）
- **Q: Muon 优化器解决什么问题？为什么不用 SVD 而用 Newton-Schulz？**
  - 二维线性层梯度矩阵的奇异值分布往往高度倾斜，普通 SGD/Adam 容易被少数大奇异值方向主导。Muon 先对梯度做 Nesterov 动量，再用 5 次 Newton-Schulz 矩阵乘迭代把更新矩阵近似正交化（使奇异值趋近于 1），近似谱范数下的最速下降；相比精确 SVD，Newton-Schulz 只含矩阵乘，在 GPU/MPS 上可充分并行。
  - 一维参数（Norm）和查表参数（Embedding）不具备线性映射的奇异值结构，因此仍交由 AdamW。
- **Q: WSD 学习率调度相比 Cosine 有什么工程优势？**
  - Cosine 必须预先定死总步数 `total_steps`，中途拿出来的模型还没退火到低学习率，往往不是最佳性能；WSD（Warmup-Stable-Decay）在恒定高原期可以随时保存 checkpoint，拉出一条分支用较短步数做 $1-\sqrt{p}$ 退火就能立刻评估或交付，主线继续往后训。

---

### 4. 后训练与强化学习（对应 `src/sft.py`, `src/rl_grpo.py`）
- **Q: SFT 时为什么要做 Loss Mask？实现时有什么容易写错的细节？**
  - 只对 `assistant` 的回复以及紧跟其后的 `<|im_end|>` 计算交叉熵损失，把 `system` / `user` / `padding` 位置在 `targets` 中设为 `-100`。
  - **易错点**：计算 next-token prediction 时 `inputs = ids[:-1], targets = ids[1:]`，因此 mask 也要对齐偏移一位（`mask[1:]`），**特别注意 `<|im_end|>` 必须保留在 loss 中**，否则模型学不会自己停止生成。
- **Q: PPO、GRPO 和 Dr. GRPO 的核心区别是什么？**
  - **PPO**：需要一个同等参数量的 Critic 网络估计状态价值 $V(s)$，显存与训练成本极高。
  - **GRPO**：对同一提示词采样一组 $G$ 个回答，用组内奖励均值和标准差归一化算 Advantage：$A_i = (r_i - \mu) / \sigma$，去掉了 Critic。
  - **Dr. GRPO**：针对 GRPO 的两个偏差做修正——① **去掉除以 $\sigma$**，避免组内奖励方差极小时过度放大简单题/难题的梯度权重；② **损失分母改用常数 $N \cdot L_{\max}$** 而非回答实际长度 $|y_i|$，消除模型在答错时倾向于输出长废话来稀释单 token 惩罚的 length bias。
