"""阶段 2：监督微调（SFT）—— 仅对 assistant 回复部分计算交叉熵损失。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_sft, load_ckpt, lr_schedule, pick_device, save_ckpt, set_lr_factor


def load_sft_examples(
    jsonl_path: Path, tok: ByteBPETokenizer, max_len: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """读取对话数据并构建 (x, y)，其中非 assistant 位置以及 padding 的 y 设为 -100。"""
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids, mask = format_chat_sft(tok, row["messages"])
        if len(ids) < 4 or len(ids) > max_len + 1:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long)
        # 预测下一 token：若下一个位置 mask==0，则 target 设为 -100（忽略损失）
        y = torch.tensor(
            [tok_id if m == 1 else -100 for tok_id, m in zip(ids[1:], mask[1:])],
            dtype=torch.long,
        )
        if (y != -100).any():
            examples.append((x, y))
    if not examples:
        raise ValueError("没有可用的 SFT 样本（可能全部超出 max_seq_len）")
    return examples


def collate_sft(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_t = max(x.size(0) for x, _ in batch)
    xs = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), max_t), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : x.size(0)] = x
        ys[i, : y.size(0)] = y
    return xs.to(device), ys.to(device)


def train(args) -> dict:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, _ = load_ckpt(args.init, device)
    examples = load_sft_examples(Path(args.data), tok, max_len=model.cfg.max_seq_len)

    rng.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_ex, val_ex = examples[:split], examples[split:] or examples[:16]

    optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
    print(f"[SFT] device={device} train={len(train_ex)} val={len(val_ex)} steps={args.steps}")

    model.train()
    last_val = float("nan")
    for step in range(args.steps):
        factor = lr_schedule(step, args.steps, warmup_steps=max(1, args.steps // 10), schedule="cosine", min_lr_frac=0.05)
        set_lr_factor(optimizers, factor)
        batch = [rng.choice(train_ex) for _ in range(args.batch_size)]
        x, y = collate_sft(batch, tok.pad_id, device)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = collate_sft(val_ex[: min(32, len(val_ex))], tok.pad_id, device)
                _, vloss = model(vx, vy)
                last_val = vloss.item()
            model.train()
            print(f"step {step+1:>4}/{args.steps} | train_loss {loss.item():.4f} | val_loss {last_val:.4f}")

    save_ckpt(args.out, model, args.steps, stage="sft", extra={"val_loss": last_val})
    print(f"SFT 完成，权重已保存至 {args.out}")
    return {"train_loss": loss.item(), "val_loss": last_val}


def main():
    p = argparse.ArgumentParser(description="SFT 对话微调")
    p.add_argument("--init", default="out/pretrain/ckpt.pt")
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/sft/ckpt.pt")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
