"""生成 Agent 研发向的学习语料：包含常规对话 + 多轮 Tool-Calling（计算器工具调用）+ 可验证算术 RL 任务。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .tokenizer import ByteBPETokenizer

PRETRAIN_TEMPLATES = [
    "Transformer 使用自注意力机制建模序列依赖关系，通过 Query、Key 和 Value 之间的点积计算上下文权重。",
    "旋转位置编码 RoPE 通过对向量的相邻维度施加二维旋转，使内积天然只依赖两个词之间的相对距离。",
    "分组查询注意力 GQA 让多个查询头共享同一组键值头，从而大幅压缩推理阶段的 KV Cache 显存占用。",
    "Agent 的工具调用本质是模型生成结构化 JSON 控制符后暂停，运行时执行函数并把 tool 结果拼回上下文。",
    "SFT 阶段会对用户输入和工具返回值做 loss mask，只训练模型自己的回复与停止符。",
    "GRPO 不需要单独的 Critic 网络，而是同一道题采样多条轨迹，用组内相对奖励来优化策略。",
]

SFT_QA = [
    ("什么是 KV Cache？为什么 Agent 多轮对话越往后越慢？", "KV Cache 会缓存历史 token 的 Key 和 Value 张量。每轮工具返回都会拉长上下文，Decode 每步都要扫描全部历史 KV，所以轮次越多访存和耗时越高。"),
    ("为什么 Prefix Cache 对 Agent 这么重要？", "因为 Agent 的 system prompt 和工具列表通常很长且固定。命中 Prefix Cache 后可以跳过前面固定前缀的 Prefill 计算，直接复用已算好的 KV。"),
    ("SFT 时为什么要把 tool 返回的结果 mask 掉不算 loss？", "因为工具返回的是外部环境状态（例如 API 响应或报错），不是模型该生成的动作；如果算 loss，模型会去死记外部返回值而不是学会决策。"),
    ("你好，请介绍一下你自己。", "你好！我是一个从零实现的迷你 Agent 语言模型，会使用 calc 工具进行多轮计算并返回最终结果。"),
]


def make_tool_task(rng: random.Random) -> tuple[str, str, int, list[dict]]:
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a, b = rng.randint(11, 49), rng.randint(11, 49)
        ans = a * b
    elif op == "+":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        ans = a + b
    else:
        a, b = rng.randint(200, 999), rng.randint(10, 199)
        ans = a - b
    expr = f"{a} {op} {b}"
    prompt = f"请使用计算器计算 {expr}，并按 #### 答案 格式输出。"
    msgs = [
        {"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "我先调用计算器算一下。",
            "tool_calls": [{"name": "calc", "arguments": {"expr": expr}}],
        },
        {"role": "tool", "content": str(ans)},
        {"role": "assistant", "content": f"计算结果是 {ans}。\n#### {ans}"},
    ]
    return prompt, expr, ans, msgs


def build_datasets(out_dir: Path, vocab_size: int = 768, seed: int = 42) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 预训练语料
    pretrain_lines: list[str] = []
    for _ in range(1500):
        pretrain_lines.append(rng.choice(PRETRAIN_TEMPLATES))
        prompt, expr, ans, _ = make_tool_task(rng)
        pretrain_lines.append(f"问题：{prompt} 算式：{expr} 结果：{ans} #### {ans}")
    rng.shuffle(pretrain_lines)
    (out_dir / "pretrain.txt").write_text("\n".join(pretrain_lines), encoding="utf-8")

    # 2. SFT 语料（多轮 tool calling + 概念问答）
    sft_rows: list[dict] = []
    for _ in range(100):
        for q, a in SFT_QA:
            sft_rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    for _ in range(800):
        _, _, _, msgs = make_tool_task(rng)
        sft_rows.append({"messages": msgs})
    rng.shuffle(sft_rows)
    with (out_dir / "sft.jsonl").open("w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3. Agent RL 任务集（训练集 + 验证集严格不重复）
    rl_rows: list[dict] = []
    seen = set()
    while len(rl_rows) < 260:
        prompt, expr, ans, _ = make_tool_task(rng)
        if expr in seen:
            continue
        seen.add(expr)
        rl_rows.append({
            "messages": [
                {"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"},
                {"role": "user", "content": prompt},
            ],
            "answer": str(ans),
            "split": "eval" if len(rl_rows) < 50 else "train",
        })
    with (out_dir / "rl_tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in rl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4. 训练 BPE 分词器
    sample_texts = pretrain_lines[:800] + [m["content"] for r in sft_rows[:250] for m in r["messages"]]
    tok = ByteBPETokenizer.train(sample_texts, vocab_size=vocab_size)
    tok.save(out_dir / "tokenizer.json")
    print(
        f"数据准备完成: {out_dir}\n"
        f"  pretrain.txt: {len(pretrain_lines)} 行\n"
        f"  sft.jsonl:    {len(sft_rows)} 条多轮/问答对话\n"
        f"  rl_tasks:     {len(rl_rows)} 道 Agent 工具调用任务\n"
        f"  tokenizer:    vocab_size={tok.vocab_size}"
    )


def main():
    p = argparse.ArgumentParser(description="生成 Agent 研发学习语料并训练 BPE 分词器")
    p.add_argument("--out", default="data")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_datasets(Path(args.out), vocab_size=args.vocab_size, seed=args.seed)


if __name__ == "__main__":
    main()
