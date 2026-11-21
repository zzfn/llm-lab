"""生成本地学习与实验用的数据，并训练 BPE 分词器。

包含三类数据（零外部依赖，开箱即用）：
1. pretrain.txt: 中英双语常识短文 + 算术推导文本，用于跑通 Pretrain
2. sft.jsonl: 对话与算术思维链样本，用于跑通 SFT（含 loss mask 验证）
3. rl_tasks.jsonl: 可验证算术问答（RLVR / GRPO 用）
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .tokenizer import ByteBPETokenizer

PRETRAIN_TEMPLATES = [
    "Transformer 使用自注意力机制建模序列依赖关系，通过 Query、Key 和 Value 之间的点积计算上下文权重。",
    "旋转位置编码 RoPE 通过对向量的相邻维度施加二维旋转，使内积天然只依赖两个词之间的相对距离。",
    "分组查询注意力 GQA 让多个查询头共享同一组键值头，从而在不明显掉点的前提下成倍压缩 KV Cache 显存。",
    "RMSNorm 去掉了 LayerNorm 的减均值操作，只按均方根缩放向量幅度，在深层网络中更高效也更稳定。",
    "Muon 优化器把二维权重矩阵的更新量通过 Newton-Schulz 迭代做近似正交化，使不同奇异值方向保持均衡步长。",
    "WSD 学习率调度包含预热、恒定高原期和末段退火三部分，可以在训练中途随时分叉退火评估模型能力。",
    "GRPO 去掉了传统 PPO 中的价值网络，改用同一问题采样多条回答的组内相对奖励来估计优势函数。",
    "Language models learn next-token prediction during pretraining, then align with human preferences via SFT and RL.",
]

SFT_QA = [
    ("什么是 RoPE 旋转位置编码？", "RoPE 通过对每个位置上的 Q 和 K 向量做与位置角相关的二维旋转，使点积结果自然编码相对位置差，具备良好的外推与相对位置性质。"),
    ("GQA 相比 MHA 解决了什么问题？", "GQA 让多组查询头共享同一组 K/V 头，大幅减少了自回归推理阶段 KV Cache 的显存占用与访存带宽压力。"),
    ("SFT 训练时为什么要对 prompt 部分做 loss mask？", "因为 SFT 的目标是学习给定问题后的回答分布，而不是背诵用户的提问文本，屏蔽输入部分的交叉熵梯度能避免无效拟合。"),
    ("GRPO 为什么不需要 Critic 模型？", "GRPO 对同一个问题采样一组输出，直接用组内奖励的均值作为基线计算相对优势，从而省去与策略同等大小的 Critic 显存开销。"),
    ("你好，请介绍一下你自己。", "你好！我是一个从零实现并训练的迷你语言模型，用来演示 BPE 分词、RoPE、GQA、Muon、SFT 与 GRPO 的完整原理。"),
]


def make_arithmetic_sample(rng: random.Random) -> tuple[str, str, int]:
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a, b = rng.randint(2, 19), rng.randint(2, 19)
        ans = a * b
    elif op == "+":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        ans = a + b
    else:
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        ans = a - b
    q = f"计算 {a} {op} {b} 等于多少？请给出最终数字，格式为 #### 答案"
    cot = f"先看算式 {a} {op} {b}，直接计算得到结果是 {ans}。\n#### {ans}"
    return q, cot, ans


def build_datasets(out_dir: Path, vocab_size: int = 768, seed: int = 42) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 预训练语料（约 4000 条混合文档）
    pretrain_lines: list[str] = []
    for _ in range(1500):
        pretrain_lines.append(rng.choice(PRETRAIN_TEMPLATES))
        q, cot, _ = make_arithmetic_sample(rng)
        pretrain_lines.append(f"题目：{q} 解答：{cot.replace(chr(10), ' ')}")
    rng.shuffle(pretrain_lines)
    (out_dir / "pretrain.txt").write_text("\n".join(pretrain_lines), encoding="utf-8")

    # 2. SFT 对话数据
    sft_rows: list[dict] = []
    for _ in range(120):
        for q, a in SFT_QA:
            sft_rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    for _ in range(600):
        q, cot, _ = make_arithmetic_sample(rng)
        sft_rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": cot}]})
    rng.shuffle(sft_rows)
    with (out_dir / "sft.jsonl").open("w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3. RLVR 可验证算术任务（训练集 + 测试集不重叠）
    rl_rows: list[dict] = []
    for i in range(300):
        q, _, ans = make_arithmetic_sample(rng)
        rl_rows.append({"prompt": q, "answer": str(ans), "split": "eval" if i < 60 else "train"})
    with (out_dir / "rl_tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in rl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4. 训练并保存 BPE 分词器
    sample_texts = pretrain_lines[:800] + [m["content"] for r in sft_rows[:200] for m in r["messages"]]
    tok = ByteBPETokenizer.train(sample_texts, vocab_size=vocab_size)
    tok.save(out_dir / "tokenizer.json")
    print(
        f"数据准备完成: {out_dir}\n"
        f"  pretrain.txt: {len(pretrain_lines)} 行\n"
        f"  sft.jsonl:    {len(sft_rows)} 条对话\n"
        f"  rl_tasks:     {len(rl_rows)} 道算术题\n"
        f"  tokenizer:    vocab_size={tok.vocab_size}"
    )


def main():
    p = argparse.ArgumentParser(description="生成学习用语料并训练 BPE 分词器")
    p.add_argument("--out", default="data")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_datasets(Path(args.out), vocab_size=args.vocab_size, seed=args.seed)


if __name__ == "__main__":
    main()
