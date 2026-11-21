"""阶段 2.5：DPO（Direct Preference Optimization）离线偏好对齐。

核心原理（无需 Reward Model，也无需在线采样）：
给定偏好对 (x, y_w=chosen, y_l=rejected) 与冻结的参考模型 ref_model：
    pi_logratios  = log pi(y_w|x) - log pi(y_l|x)
    ref_logratios = log pi_ref(y_w|x) - log pi_ref(y_l|x)
    loss = -log sigmoid(beta * (pi_logratios - ref_logratios))
仅在 assistant 回复位置计算序列对数概率之和。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .loss import token_log_probs
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_sft, load_ckpt, lr_schedule, pick_device, save_ckpt, set_lr_factor, step_all


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    # 隐式奖励准确率：chosen 相对 ref 的提升大于 rejected 的比例
    acc = (logits > 0).float().mean()
    return loss, acc


def _sequence_logps(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits, _ = model(ids[:, :-1], return_logits=True)
    lp = token_log_probs(logits, ids[:, 1:])
    return (lp * mask[:, 1:]).sum(dim=1)


def load_dpo_pairs(jsonl_path: Path, tok: ByteBPETokenizer, max_len: int) -> list[dict]:
    pairs = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        c_ids, c_mask = format_chat_sft(tok, row["chosen"], train_on="assistant")
        r_ids, r_mask = format_chat_sft(tok, row["rejected"], train_on="assistant")
        if 4 <= len(c_ids) <= max_len + 1 and 4 <= len(r_ids) <= max_len + 1 and any(c_mask[1:]) and any(r_mask[1:]):
            pairs.append({"c_ids": c_ids, "c_mask": c_mask, "r_ids": r_ids, "r_mask": r_mask})
    if not pairs:
        raise ValueError("没有可用的 DPO 偏好对")
    return pairs


def _collate_side(seqs: list[list[int]], masks: list[list[int]], pad_id: int, device: torch.device):
    max_t = max(len(s) for s in seqs)
    ids_t = torch.full((len(seqs), max_t), pad_id, dtype=torch.long, device=device)
    mask_t = torch.zeros((len(seqs), max_t), dtype=torch.float32, device=device)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask_t[i, : len(m)] = torch.tensor(m, dtype=torch.float32, device=device)
    return ids_t, mask_t


def train(args) -> dict:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)

    policy, _ = load_ckpt(args.init, device)
    ref_model, _ = load_ckpt(args.init, device)
    ref_model.eval().requires_grad_(False)

    pairs = load_dpo_pairs(Path(args.data), tok, max_len=policy.cfg.max_seq_len)
    optimizers = build_optimizers(policy, opt_name=args.optimizer, lr_mult=args.lr_mult)
    print(f"[DPO] device={device} init={args.init} pairs={len(pairs)} beta={args.beta} steps={args.steps}")

    policy.train()
    last_loss, last_acc = 0.0, 0.0
    for step in range(args.steps):
        factor = lr_schedule(step, args.steps, warmup_steps=max(1, args.steps // 10), schedule="cosine", min_lr_frac=0.1)
        set_lr_factor(optimizers, factor)
        batch = [rng.choice(pairs) for _ in range(args.batch_size)]
        c_ids, c_mask = _collate_side([b["c_ids"] for b in batch], [b["c_mask"] for b in batch], tok.pad_id, device)
        r_ids, r_mask = _collate_side([b["r_ids"] for b in batch], [b["r_mask"] for b in batch], tok.pad_id, device)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            ref_c = _sequence_logps(ref_model, c_ids, c_mask)
            ref_r = _sequence_logps(ref_model, r_ids, r_mask)
        pol_c = _sequence_logps(policy, c_ids, c_mask)
        pol_r = _sequence_logps(policy, r_ids, r_mask)

        loss, acc = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=args.beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        step_all(policy, optimizers)

        last_loss, last_acc = loss.item(), acc.item()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"step {step+1:>3}/{args.steps} | dpo_loss {last_loss:.4f} | pref_acc {last_acc:.1%}")

    save_ckpt(args.out, policy, args.steps, stage="dpo", extra={"dpo_loss": last_loss, "pref_acc": last_acc})
    print(f"DPO 完成，权重已保存至 {args.out}")
    return {"dpo_loss": last_loss, "pref_acc": last_acc}


def main():
    p = argparse.ArgumentParser(description="DPO 离线偏好对齐")
    p.add_argument("--init", default="out/sft/ckpt.pt")
    p.add_argument("--data", default="data/dpo.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/dpo/ckpt.pt")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
