"""阶段 3：Agent Tool-Calling + Dr. GRPO 强化学习。

核心流程（向 Agent 研发同学展示“工具调用 + RL”底层发生了什么）：
1. 构造提示 `<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant\n`
2. 模型采样生成一轮，遇到 `<|im_end|>` 停止；解析 `<|tool_call|>...<|/tool_call|>`
3. 若有工具调用，Python 运行时执行安全的 `calc` 表达式，把结果包成 `<|im_start|>tool\n结果<|im_end|>` 接回上下文，
   并把 tool 返回部分的 learn mask 置为 0（只对模型自己写的 token 计算梯度）
4. 模型继续生成最终答复，用 `#### 答案` 判分，损坏的工具调用 JSON 额外扣分
5. 用 Dr. GRPO（组内去均值基线 + 常数长度分母）更新策略
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .generate import generate_ids
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_prompt, format_tool_turn, load_ckpt, parse_tool_calls, pick_device, save_ckpt

ANS_RE = re.compile(r"####\s*(-?\d+)")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def safe_eval_calc(expr: str) -> str:
    """安全计算简单整数四则表达式，不直接 eval 任意代码。"""
    try:
        node = ast.parse(expr.strip(), mode="eval").body
        def _eval(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in OPS:
                return OPS[type(n.op)](_eval(n.left), _eval(n.right))
            raise ValueError("不支持的表达式")
        return str(_eval(node))
    except Exception as exc:
        return f"error: {exc}"


def execute_tool(name: str, arguments: dict) -> str:
    if name == "calc":
        return safe_eval_calc(str(arguments.get("expr", "")))
    return f"error: unknown tool {name}"


def run_agent_rollout(
    model,
    tok: ByteBPETokenizer,
    messages: list[dict],
    max_new: int,
    temperature: float,
    device: torch.device,
    max_turns: int = 3,
) -> tuple[list[int], list[int], str, int]:
    """执行多轮 Agent 采样，返回 (完整 ids, learn_mask, 最终回复正文, 格式损坏次数)。"""
    ids = format_chat_prompt(tok, messages)
    learn = [0] * len(ids)
    broken_count = 0
    final_text = ""

    for _ in range(max_turns):
        room = min(max_new, model.cfg.max_seq_len - len(ids))
        if room <= 0:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate_ids(model, x, max_new_tokens=room, temperature=temperature, top_k=40, eos_id=tok.im_end_id)
        resp = out[0, len(ids) :].tolist()
        if tok.im_end_id in resp:
            resp = resp[: resp.index(tok.im_end_id) + 1]
        ids.extend(resp)
        learn.extend([1] * len(resp))

        text, calls, broken = parse_tool_calls(tok, resp)
        final_text = text
        broken_count += len(broken)

        if not calls and not broken:
            break
        if not calls:
            # 损坏的工具调用不再继续下一轮，直接结算惩罚
            break
        tool_results = [execute_tool(c["name"], c["arguments"]) for c in calls]
        back = format_tool_turn(tok, "\n".join(tool_results))
        if len(ids) + len(back) >= model.cfg.max_seq_len:
            break
        ids.extend(back)
        learn.extend([0] * len(back))

    return ids, learn, final_text, broken_count


def score_episode(final_text: str, gold: str, broken_count: int) -> float:
    m = ANS_RE.search(final_text)
    if m and m.group(1) == gold.strip():
        r = 1.0
    elif m:
        r = 0.1
    else:
        r = 0.0
    return max(-0.5, r - 0.2 * broken_count)


def dr_grpo_advantages(rewards: list[float]) -> list[float]:
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


@torch.no_grad()
def evaluate_accuracy(model, tok: ByteBPETokenizer, tasks: list[dict], device: torch.device, max_new: int = 48) -> float:
    model.eval()
    correct = 0
    for task in tasks:
        _, _, final_text, broken = run_agent_rollout(model, tok, task["messages"], max_new=max_new, temperature=0.0, device=device)
        if score_episode(final_text, task["answer"], broken) == 1.0:
            correct += 1
    model.train()
    return correct / max(1, len(tasks))


def train(args) -> dict:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, _ = load_ckpt(args.init, device)

    all_tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    train_tasks = [t for t in all_tasks if t.get("split") == "train"]
    eval_tasks = [t for t in all_tasks if t.get("split") == "eval"]

    optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
    init_acc = evaluate_accuracy(model, tok, eval_tasks, device, max_new=args.max_new)
    print(f"[Agent GRPO] device={device} train={len(train_tasks)} eval={len(eval_tasks)} init_eval_acc={init_acc:.1%}")

    for step in range(args.steps):
        batch_tasks = rng.sample(train_tasks, k=min(args.prompts_per_step, len(train_tasks)))
        episodes: list[tuple[list[int], list[int], float]] = []
        raw_rewards: list[float] = []

        model.eval()
        with torch.no_grad():
            for task in batch_tasks:
                group_seqs, group_masks, group_rewards = [], [], []
                for _ in range(args.group_size):
                    ids, learn, final_text, broken = run_agent_rollout(
                        model, tok, task["messages"], max_new=args.max_new, temperature=args.temperature, device=device
                    )
                    group_seqs.append(ids)
                    group_masks.append(learn)
                    group_rewards.append(score_episode(final_text, task["answer"], broken))
                raw_rewards.extend(group_rewards)
                for seq, learn, adv in zip(group_seqs, group_masks, dr_grpo_advantages(group_rewards)):
                    if adv != 0.0 and len(seq) <= model.cfg.max_seq_len:
                        episodes.append((seq, learn, adv))

        loss_val = 0.0
        if episodes:
            model.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            max_t = max(len(seq) for seq, _, _ in episodes)
            n_total = args.prompts_per_step * args.group_size
            ids_t = torch.full((len(episodes), max_t), tok.pad_id, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(episodes), max_t), dtype=torch.float32, device=device)
            adv_t = torch.tensor([adv for _, _, adv in episodes], dtype=torch.float32, device=device)

            for i, (seq, learn, _) in enumerate(episodes):
                ids_t[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask_t[i, : len(learn)] = torch.tensor(learn, dtype=torch.float32, device=device)

            logits, _ = model(ids_t[:, :-1])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_lp = log_probs.gather(-1, ids_t[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Dr. GRPO: 仅统计模型自己写的 token（tool 返回结果已被 mask 为 0），分母用常数
            pg_loss = -(adv_t[:, None] * target_lp * mask_t[:, 1:]).sum() / (n_total * args.max_new * 2)
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
            loss_val = pg_loss.item()

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            eval_acc = evaluate_accuracy(model, tok, eval_tasks, device, max_new=args.max_new)
            avg_r = sum(raw_rewards) / max(1, len(raw_rewards))
            print(
                f"step {step+1:>3}/{args.steps} | avg_reward {avg_r:+.3f} | "
                f"pg_loss {loss_val:.5f} | eval_acc {eval_acc:.1%}"
            )

    final_acc = evaluate_accuracy(model, tok, eval_tasks, device, max_new=args.max_new)
    save_ckpt(args.out, model, args.steps, stage="rl_grpo", extra={"eval_acc": final_acc})
    print(f"Agent GRPO 完成，最终评测准确率: {final_acc:.1%}，权重已保存至 {args.out}")
    return {"init_acc": init_acc, "final_acc": final_acc}


def main():
    p = argparse.ArgumentParser(description="Agent Tool-Calling Dr. GRPO 强化学习")
    p.add_argument("--init", default="out/sft/ckpt.pt")
    p.add_argument("--tasks", default="data/rl_tasks.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/rl/ckpt.pt")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
