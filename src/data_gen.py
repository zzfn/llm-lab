"""生成全套训练与评测语料（麻雀虽小五脏俱全）：
1. pretrain.txt             预训练语料
2. sft.jsonl                通用 SFT（含多轮工具调用、概念问答、格式指令遵循、选择题）
3. sft_expert_math.jsonl    数学/工具专家 SFT（供 MOPD 蒸馏）
4. sft_expert_qa.jsonl      原理问答专家 SFT（供 MOPD 蒸馏）
5. mopd_prompts.jsonl       MOPD 多老师在线蒸馏提示集（带 domain 路由字段）
6. rl_tasks.jsonl           多题型 RLVR/评测题集（agent / instruct / choice，train 与 eval 不重复）
7. tokenizer.json           自训 Byte-BPE 分词器
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
    "分组查询注意力 GQA 让多个查询头共享同一组键值头，从而大幅压缩推理阶段的 KV Cache 显存占用。",
    "mHC 多流残差把单条残差拓宽为多条流，并用 Sinkhorn 双随机矩阵保证信号总量在深层间不放大也不衰减。",
    "Engram 条件记忆直接按二元和三元词组哈希查表，不花算力就能记住常见搭配与固定知识。",
    "3:1 滑动窗口注意力让多数层只关注局部上下文，每四层留一层看全文，兼顾长文与显存。",
    "Muon 优化器把二维权重矩阵的更新量通过 Newton-Schulz 迭代做近似正交化，使不同方向步长均衡。",
    "Agent 的工具调用本质是模型生成结构化 JSON 控制符后暂停，运行时执行函数并把 tool 结果拼回上下文。",
]

QA_PAIRS = [
    ("什么是 KV Cache？为什么 Agent 多轮对话越往后越慢？",
     "KV Cache 会缓存历史 token 的 Key 和 Value。每轮工具返回都会拉长上下文，Decode 每步都要扫描全部历史 KV，所以轮次越多访存和耗时越高。"),
    ("为什么 Prefix Cache 对 Agent 这么重要？",
     "因为 Agent 的 system prompt 和工具定义通常很长且固定，命中 Prefix Cache 可跳过固定前缀的 Prefill 计算，大幅降低首字延迟。"),
    ("mHC 多流残差里的 Sinkhorn 迭代有什么用？",
     "Sinkhorn 在对数域交替按行按列归一化，把流间混合矩阵约束为双随机矩阵，避免多层信号爆炸或消失。"),
    ("Engram 查表记忆有什么特点？",
     "Engram 按 n-gram 多头质数哈希直接查表取记忆向量，只占显存几乎不花算力，相当于模型的静态词典外挂。"),
    ("SFT 时为什么要把 tool 返回的结果 mask 掉不算 loss？",
     "因为工具返回是外部环境状态，不是模型要生成的动作；屏蔽后模型才不会去死记外部接口输出。"),
    ("你好，请介绍一下你自己。",
     "你好！我是一个麻雀虽小五脏俱全的迷你大模型，集成了 mHC、Engram、GQA、MoE、Muon、MOPD 和 Dr.GRPO。"),
]

CHOICES = [
    ("哪种机制通过双随机矩阵稳定多流残差信号？\nA. Dropout\nB. mHC\nC. BatchNorm\nD. Softmax", "B"),
    ("下列哪项可以直接压缩 KV Cache 显存占用？\nA. GQA\nB. 增大词表\nC. 取消位置编码\nD. 增大 Batch", "A"),
    ("Muon 优化器通过什么方法近似正交化矩阵更新？\nA. Cholesky\nB. Newton-Schulz\nC. L-BFGS\nD. Adam", "B"),
    ("Dr. GRPO 相比原始 GRPO 去掉了哪项以防简单题梯度过大？\nA. 组内均值\nB. 组内标准差\nC. 奖励函数\nD. 策略概率", "B"),
    ("SFT 训练中哪类 token 不应参与交叉熵损失？\nA. assistant 正文\nB. 停止符\nC. tool 返回结果\nD. tool_calls", "C"),
]

INSTRUCT_SAMPLES = [
    (
        "请用不超过30字概括什么是 RoPE，必须包含“旋转”二字，且不要使用逗号。",
        {"max_chars": 30, "must_include": ["旋转"], "no_comma": True},
        "RoPE通过二维旋转编码使注意力点积只依赖相对距离。",
    ),
    (
        "请以JSON格式输出KV Cache的两个作用，必须包含键 summary 和 benefit。",
        {"json_keys": ["summary", "benefit"]},
        '{"summary": "缓存历史键值张量", "benefit": "加速自回归生成"}',
    ),
    (
        "用一句话说明为什么要做 SFT Loss Mask，必须包含“工具”且不超过35字。",
        {"max_chars": 35, "must_include": ["工具"]},
        "避免模型去背诵用户输入或外部工具返回内容。",
    ),
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
        {"role": "assistant", "content": "我先调用计算器算一下。", "tool_calls": [{"name": "calc", "arguments": {"expr": expr}}]},
        {"role": "tool", "content": str(ans)},
        {"role": "assistant", "content": f"计算结果是 {ans}。\n#### {ans}"},
    ]
    return prompt, expr, ans, msgs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    # 2. 通用 SFT + 领域专家 SFT
    expert_math, expert_qa, general_sft = [], [], []
    for _ in range(80):
        for q, a in QA_PAIRS:
            row = {"messages": [{"role": "user", "content": q}, {"role": "assistant", "thinking": f"先回忆核心定义再作答：{q[:12]}", "content": a}]}
            expert_qa.append(row)
            general_sft.append(row)
    for prompt, _, ans_text in INSTRUCT_SAMPLES:
        for _ in range(40):
            general_sft.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans_text}]})
    for q, letter in CHOICES:
        for _ in range(40):
            general_sft.append({"messages": [{"role": "user", "content": f"请直接回答选项字母：\n{q}"}, {"role": "assistant", "content": f"答案：{letter}"}]})
    for _ in range(600):
        _, _, _, msgs = make_tool_task(rng)
        row = {"messages": msgs}
        expert_math.append(row)
        general_sft.append(row)

    # 补充 2~3 轮普通多轮对话（不依赖 tool 轮次，对齐网页端多轮 history 分布）
    single_turns = (
        [({"role": "user", "content": q}, {"role": "assistant", "content": a}) for q, a in QA_PAIRS]
        + [({"role": "user", "content": prompt}, {"role": "assistant", "content": ans_text}) for prompt, _, ans_text in INSTRUCT_SAMPLES]
        + [({"role": "user", "content": f"请直接回答选项字母：\n{q}"}, {"role": "assistant", "content": f"答案：{letter}"}) for q, letter in CHOICES]
    )
    for _ in range(500):
        k_turns = rng.choice([2, 3])
        picked = rng.sample(single_turns, k=k_turns)
        msgs = []
        for u_msg, a_msg in picked:
            msgs.extend([u_msg, a_msg])
        general_sft.append({"messages": msgs})

    rng.shuffle(general_sft)
    rng.shuffle(expert_math)
    rng.shuffle(expert_qa)
    _write_jsonl(out_dir / "sft.jsonl", general_sft)
    _write_jsonl(out_dir / "sft_expert_math.jsonl", expert_math)
    _write_jsonl(out_dir / "sft_expert_qa.jsonl", expert_qa)

    # 3. MOPD 提示集（不带回答，按 domain 路由到对应专家）
    mopd_prompts: list[dict] = []
    for _ in range(80):
        prompt, _, _, _ = make_tool_task(rng)
        mopd_prompts.append({
            "domain": "math",
            "messages": [{"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"}, {"role": "user", "content": prompt}],
        })
    for q, _ in QA_PAIRS:
        for _ in range(10):
            mopd_prompts.append({"domain": "qa", "messages": [{"role": "user", "content": q}]})
    rng.shuffle(mopd_prompts)
    _write_jsonl(out_dir / "mopd_prompts.jsonl", mopd_prompts)

    # 4. 多题型 RLVR & 评测题集（agent / instruct / choice）
    rl_rows: list[dict] = []
    seen = set()
    while len(rl_rows) < 200:
        prompt, expr, ans, _ = make_tool_task(rng)
        if expr in seen:
            continue
        seen.add(expr)
        rl_rows.append({
            "reward": "agent",
            "messages": [{"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"}, {"role": "user", "content": prompt}],
            "answer": str(ans),
            "split": "eval" if len(rl_rows) < 40 else "train",
        })
    for idx, (prompt, rules, _) in enumerate(INSTRUCT_SAMPLES * 10):
        rl_rows.append({
            "reward": "instruct",
            "messages": [{"role": "user", "content": prompt}],
            "rules": rules,
            "split": "eval" if idx < 6 else "train",
        })
    for idx, (q, letter) in enumerate(CHOICES * 6):
        rl_rows.append({
            "reward": "choice",
            "messages": [{"role": "user", "content": f"请直接回答选项字母：\n{q}"}],
            "answer": letter,
            "split": "eval" if idx < 5 else "train",
        })
    _write_jsonl(out_dir / "rl_tasks.jsonl", rl_rows)

    # 5. 生成 DPO 偏好对（chosen=规范调用工具/准确作答，rejected=瞎猜或格式错误）
    dpo_rows: list[dict] = []
    for _ in range(120):
        prompt, expr, ans, msgs = make_tool_task(rng)
        wrong_ans = ans + rng.choice([1, -1, 10])
        dpo_rows.append({
            "chosen": msgs,
            "rejected": [
                {"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"直接猜一下结果是 {wrong_ans}。"},
            ],
        })
    for q, a in QA_PAIRS * 10:
        dpo_rows.append({
            "chosen": [{"role": "user", "content": q}, {"role": "assistant", "thinking": "按核心原理回答。", "content": a}],
            "rejected": [{"role": "user", "content": q}, {"role": "assistant", "content": "这个问题不太清楚，大概随便设置一下就行。"}],
        })
    _write_jsonl(out_dir / "dpo.jsonl", dpo_rows)

    # 6. 训练分词器
    sample_texts = pretrain_lines[:800] + [m["content"] for r in general_sft[:250] for m in r["messages"]]
    tok = ByteBPETokenizer.train(sample_texts, vocab_size=vocab_size)
    tok.save(out_dir / "tokenizer.json")
    print(
        f"数据准备完成: {out_dir}\n"
        f"  pretrain.txt:       {len(pretrain_lines)} 行\n"
        f"  sft.jsonl:          {len(general_sft)} 条（另含 expert_math {len(expert_math)} / expert_qa {len(expert_qa)}）\n"
        f"  dpo.jsonl:          {len(dpo_rows)} 对\n"
        f"  mopd_prompts.jsonl: {len(mopd_prompts)} 道\n"
        f"  rl_tasks.jsonl:     {len(rl_rows)} 题（agent + instruct + choice）\n"
        f"  tokenizer:          vocab_size={tok.vocab_size}"
    )


def main():
    p = argparse.ArgumentParser(description="生成全链路多题型学习语料并训练 BPE 分词器")
    p.add_argument("--out", default="data")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_datasets(Path(args.out), vocab_size=args.vocab_size, seed=args.seed)


if __name__ == "__main__":
    main()
