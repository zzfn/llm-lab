"""统一评测脚本：模型参数拆解 + 预训练留出集损失 + 多题型 RLVR 正确率 + 固定提问样本。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pretrain import build_stream, eval_loss
from .rl_grpo import evaluate_tasks, run_episode
from .tokenizer import ByteBPETokenizer
from .utils import load_ckpt, pick_device


def main():
    p = argparse.ArgumentParser(description="评测检查点：参数构成、留出集损失与多题型正确率")
    p.add_argument("ckpt", nargs="?", default="out/rl/ckpt.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--pretrain-data", default="data/pretrain.txt")
    p.add_argument("--tasks", default="data/rl_tasks.jsonl")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, ckpt = load_ckpt(args.ckpt, device)

    print(f"=== 模型评测报告: {args.ckpt} (stage={ckpt.get('stage')}, step={ckpt.get('step')}) ===")
    print(f"总参数: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    for k, v in model.param_breakdown().items():
        print(f"  - {k}: {v:,}")

    if Path(args.pretrain_data).exists():
        _, val_stream = build_stream(Path(args.pretrain_data), tok)
        seq_len = min(128, model.cfg.max_seq_len)
        if len(val_stream) > seq_len + 2:
            vloss, sat = eval_loss(model, val_stream, batch_size=8, seq_len=seq_len, device=device)
            print(f"预训练留出集损失: {vloss:.4f} (softmax 饱和占比: {sat:.2%})")

    if Path(args.tasks).exists():
        eval_tasks = [json.loads(l) for l in Path(args.tasks).read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("split") == "eval"]
        scores = evaluate_tasks(model, tok, eval_tasks, device)
        print("多题型评测得分:", {k: round(v, 3) for k, v in scores.items()})

        print("=== 固定样本抽检 ===")
        for kind in ("agent", "instruct", "choice"):
            sample = next((t for t in eval_tasks if t.get("reward") == kind), None)
            if sample:
                _, _, text, broken = run_episode(model, tok, sample, max_new=64, temperature=0.0, device=device)
                print(f"[{kind}] Q: {sample['messages'][-1]['content'].splitlines()[0]} -> A: {text!r} (broken={broken})")


if __name__ == "__main__":
    main()
