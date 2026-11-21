"""阶段 3：GRPO / Dr. GRPO 强化学习（RLVR 可验证算术奖励）。

面试核心要点：
1. 为什么用 GRPO 而不是 PPO？
   PPO 需要一个与策略同规模的 Critic（价值网络）估计 V(s)，显存翻倍；
   GRPO 对每道题采样 G 个回答，用组内奖励的均值作为基线估计 Advantage。
2. Dr. GRPO 做了什么修正？
   - 去掉除以组内 std（防止极简单/极难题方差过小导致梯度权重爆炸）
   - 损失分母用常数 (N * max_new) 而不是每条回答的实际长度 |y_i|，消除“答错时越写越长以稀释单步惩罚”的 length bias。
3. 在线单步更新（on-policy, 1 update per batch）：
   因为 theta == theta_old，概率比 rho_t = 1，clip 不触发，PPO 目标退化为带组内基线的策略梯度：
   L = - 1/(N * L_max) * sum_i A_i * sum_t log pi_theta(y_{i,t} | h_{i,t})
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .generate import generate_ids
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_prompt, load_ckpt, pick_device, save_ckpt

ANS_RE = re.compile(r"####\s*(-?\d+)")


def score_answer(text: str, gold: str) -> float:
    """可验证奖励：格式正确且数值完全匹配得 1.0；仅有格式得 0.1；否则 0.0。"""
    m = ANS_RE.search(text)
    if not m:
        return 0.0
    return 1.0 if m.group(1) == gold.strip() else 0.1


def dr_grpo_advantages(rewards: list[float]) -> list[float]:
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


@torch.no_grad()
def evaluate_accuracy(model, tok: ByteBPETokenizer, tasks: list[dict], device: torch.device, max_new: int = 48) -> float:
    model.eval()
    correct = 0
    for task in tasks:
        prompt_ids = format_chat_prompt(tok, [{"role": "user", "content": task["prompt"]}])
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate_ids(model, idx, max_new_tokens=max_new, temperature=0.0, eos_id=tok.im_end_id)
        ans_text = tok.decode(out[0, len(prompt_ids) :].tolist(), skip_special=True)
        if score_answer(ans_text, task["answer"]) == 1.0:
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
    print(f"[GRPO] device={device} train={len(train_tasks)} eval={len(eval_tasks)} init_eval_acc={init_acc:.1%}")

    for step in range(args.steps):
        batch_tasks = rng.sample(train_tasks, k=min(args.prompts_per_step, len(train_tasks)))
        episodes: list[tuple[list[int], int, float]] = []  # (full_ids, prompt_len, advantage)
        raw_rewards: list[float] = []

        model.eval()
        with torch.no_grad():
            for task in batch_tasks:
                prompt = format_chat_prompt(tok, [{"role": "user", "content": task["prompt"]}])
                idx = torch.tensor([prompt] * args.group_size, dtype=torch.long, device=device)
                out = generate_ids(
                    model, idx, max_new_tokens=args.max_new, temperature=args.temperature, eos_id=tok.im_end_id
                )
                group_rewards: list[float] = []
                group_seqs: list[list[int]] = []
                for row in out.tolist():
                    resp = row[len(prompt) :]
                    if tok.im_end_id in resp:
                        resp = resp[: resp.index(tok.im_end_id) + 1]
                    text = tok.decode(resp, skip_special=True)
                    group_rewards.append(score_answer(text, task["answer"]))
                    group_seqs.append(prompt + resp)
                raw_rewards.extend(group_rewards)
                for seq, adv in zip(group_seqs, dr_grpo_advantages(group_rewards)):
                    if adv != 0.0 and len(seq) <= model.cfg.max_seq_len:
                        episodes.append((seq, len(prompt), adv))

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

            for i, (seq, plen, _) in enumerate(episodes):
                ids_t[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask_t[i, plen : len(seq)] = 1.0

            logits, _ = model(ids_t[:, :-1])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_lp = log_probs.gather(-1, ids_t[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Dr. GRPO: 常数分母 n_total * max_new
            pg_loss = -(adv_t[:, None] * target_lp * mask_t[:, 1:]).sum() / (n_total * args.max_new)
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
            loss_val = pg_loss.item()

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            eval_acc = evaluate_accuracy(model, tok, eval_tasks, device, max_new=args.max_new)
            avg_r = sum(raw_rewards) / max(1, len(raw_rewards))
            print(
                f"step {step+1:>3}/{args.steps} | avg_reward {avg_r:.3f} | "
                f"pg_loss {loss_val:.5f} | eval_acc {eval_acc:.1%}"
            )

    final_acc = evaluate_accuracy(model, tok, eval_tasks, device, max_new=args.max_new)
    save_ckpt(args.out, model, args.steps, stage="rl_grpo", extra={"eval_acc": final_acc})
    print(f"GRPO 完成，最终评测准确率: {final_acc:.1%}，权重已保存至 {args.out}")
    return {"init_acc": init_acc, "final_acc": final_acc}


def main():
    p = argparse.ArgumentParser(description="Dr. GRPO 可验证算术强化学习")
    p.add_argument("--init", default="out/sft/ckpt.pt")
    p.add_argument("--tasks", default="data/rl_tasks.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/rl/ckpt.pt")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
