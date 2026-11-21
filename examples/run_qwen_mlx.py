"""对比实验：用 Apple 原生 MLX-LM 对 Qwen3.5-0.8B (4-bit) 做 LoRA 微调，并对比微调前后效果。

使用方法（需先安装 mlx-lm）：
    python -m examples.run_qwen_mlx --iters 80 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mlx_lm import generate, load
import mlx.core as mx

from src.rl_grpo import execute_tool
from src.utils import SYSTEM_PROMPT

CALC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算数学表达式并返回数值结果，仅在需要精确计算时调用。",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "要计算的四则运算表达式"}},
                "required": ["expr"],
            },
        },
    }
]

XML_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([a-zA-Z0-9_]+)>\s*(.*?)</function>\s*</tool_call>",
    re.S,
)
PARAM_RE = re.compile(r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>", re.S)
STOP_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>")


def _convert_messages(raw_msgs: list[dict]) -> list[dict]:
    """将 sft_mixed.jsonl 消息转为 Qwen3.5 原生 ChatDataset 消息格式。"""
    out: list[dict] = []
    for m in raw_msgs:
        msg: dict = {"role": m["role"], "content": m.get("content", "")}
        if m.get("thinking"):
            msg["reasoning_content"] = m["thinking"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                c if "function" in c else {"type": "function", "function": c}
                for c in m["tool_calls"]
            ]
        out.append(msg)
    return out


def prepare_mlx_data(
    src_jsonl: Path,
    out_dir: Path,
    tokenizer,
    max_seq_length: int = 640,
    seed: int = 42,
) -> tuple[int, int]:
    """去重、平衡工具/通用样本并按 Qwen3.5 原生模板过滤超长样本，输出 {train,valid,test}.jsonl。"""
    rng = random.Random(seed)
    seen: set[str] = set()
    tool_rows: list[dict] = []
    chat_rows: list[dict] = []

    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_msgs = json.loads(line)["messages"]
        key = json.dumps(raw_msgs, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        msgs = _convert_messages(raw_msgs)
        has_tool = any(m.get("tool_calls") for m in msgs)
        prefix: list[dict] = []
        for m in msgs:
            prefix.append(m)
            if m["role"] == "assistant" and (m["content"] or m.get("tool_calls")):
                tok_len = len(
                    tokenizer.apply_chat_template(prefix, tools=CALC_TOOLS, tokenize=True)
                )
                if tok_len <= max_seq_length:
                    sample = {"messages": [dict(x) for x in prefix], "tools": CALC_TOOLS}
                    (tool_rows if has_tool else chat_rows).append(sample)

    # 工具拆分后较多，保持 工具:通用/领域问答 大致 1:1 均衡，避免首词偏置压倒免工具问答
    rng.shuffle(tool_rows)
    max_tool = min(len(tool_rows), max(len(chat_rows), 40))
    rows = chat_rows + tool_rows[:max_tool]
    rng.shuffle(rows)

    n_val = max(12, min(40, int(len(rows) * 0.1)))
    n_test = min(20, max(8, int(len(rows) * 0.05)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [
        ("valid.jsonl", rows[:n_val]),
        ("test.jsonl", rows[n_val : n_val + n_test]),
        ("train.jsonl", rows[n_val + n_test :]),
    ]:
        (out_dir / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in split) + "\n",
            encoding="utf-8",
        )
    return len(rows) - n_val - n_test, n_val


def _parse_tool_call(reply: str) -> tuple[str, dict] | None:
    m = XML_TOOL_RE.search(reply)
    if not m:
        return None
    name = m.group(1).strip()
    args = {pm.group(1).strip(): pm.group(2).strip() for pm in PARAM_RE.finditer(m.group(2))}
    return name, args


def run_agent_chat(model, tokenizer, user_prompt: str, max_tokens: int = 384) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[str] = []
    for _ in range(3):
        prompt = tokenizer.apply_chat_template(
            messages, tools=CALC_TOOLS, tokenize=False, add_generation_prompt=True
        )
        raw_reply = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        reply = STOP_RE.split(raw_reply, maxsplit=1)[0].strip()
        parsed = _parse_tool_call(reply)
        if not parsed:
            trace.append(reply)
            break
        name, args = parsed
        trace.append(reply)
        res = execute_tool(name, args)
        trace.append(f"[Tool 返回] {res}")
        # 解析出思维与纯工具调用，便于多轮模板正确拼接
        thinking = ""
        if "</think>" in reply:
            thinking = reply.split("</think>", 1)[0].replace("<think>", "").strip()
        assistant_msg: dict = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
        }
        if thinking:
            assistant_msg["reasoning_content"] = thinking
        messages.append(assistant_msg)
        messages.append({"role": "tool", "name": name, "content": str(res)})
    return "\n  ".join(trace)


def eval_prompts(model, tokenizer, label: str, prompts: list[str]) -> None:
    print(f"\n=== {label} ===")
    for p in prompts:
        t0 = time.perf_counter()
        out = run_agent_chat(model, tokenizer, p)
        dt = time.perf_counter() - t0
        print(f"Q: {p} ({dt:.2f}s)\n  {out}\n")


def main():
    p = argparse.ArgumentParser(description="MLX-LM Qwen3.5-0.8B 4-bit LoRA 对比实验")
    p.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-4bit")
    p.add_argument("--src-data", default="data/sft_mixed.jsonl")
    p.add_argument("--mlx-data", default="data/mlx_sft")
    p.add_argument("--adapter-path", default="out/qwen35_lora")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=640)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--skip-base", action="store_true", help="跳过基座评测直接微调并评测 LoRA")
    args = p.parse_args()

    prompts = [
        "请使用计算器计算 1 * 0，并按 #### 答案 格式输出。",
        "帮我算一下三十八乘以二十四等于多少？",
        "什么是 KV Cache？为什么 Agent 多轮对话越往后越慢？",
        "用一句话解释为什么地球会有四季更替？",
    ]

    print(f"加载基座模型: {args.model}")
    base_model, tok = load(args.model)
    n_train, n_val = prepare_mlx_data(
        Path(args.src_data), Path(args.mlx_data), tok, max_seq_length=args.max_seq_length
    )
    print(f"[MLX Data] train={n_train} valid={n_val} -> {args.mlx_data}")

    if not args.skip_base:
        eval_prompts(base_model, tok, "微调前 (Base 4-bit)", prompts)
    del base_model, tok
    mx.clear_cache()

    shutil.rmtree(args.adapter_path, ignore_errors=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", args.mlx_data,
        "--mask-prompt",
        "--num-layers", str(args.num_layers),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation-steps", str(args.grad_accum),
        "--max-seq-length", str(args.max_seq_length),
        "--grad-checkpoint",
        "--learning-rate", str(args.lr),
        "--steps-per-report", "20",
        "--steps-per-eval", "40",
        "--adapter-path", args.adapter_path,
    ]
    print("开始 MLX LoRA 微调:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    tuned_model, tok = load(args.model, adapter_path=args.adapter_path)
    eval_prompts(tuned_model, tok, "微调后 (Qwen3.5-0.8B + LoRA)", prompts)


if __name__ == "__main__":
    main()
