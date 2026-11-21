"""阶段 2：监督微调（SFT）—— 通用 SFT 与领域专家 SFT 共用，仅对 assistant 回复计算损失。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .lora import apply_lora, merge_lora, save_lora_only
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_sft, load_ckpt, lr_schedule, pick_device, save_ckpt, set_lr_factor, step_all


def load_sft_examples(
    jsonl_path: Path, tok: ByteBPETokenizer, max_len: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids, mask = format_chat_sft(tok, row["messages"], train_on="assistant")
        if len(ids) < 4 or len(ids) > max_len + 1:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long)
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
    init_path = args.init or ("out/anneal/ckpt.pt" if Path("out/anneal/ckpt.pt").exists() else "out/pretrain/ckpt.pt")
    model, _ = load_ckpt(init_path, device)
    examples = load_sft_examples(Path(args.data), tok, max_len=model.cfg.max_seq_len)

    rng.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_ex, val_ex = examples[:split], examples[split:] or examples[:16]

    out_path = args.out
    if not out_path:
        stem = Path(args.data).stem
        out_path = f"out/{stem.removeprefix('sft_')}/ckpt.pt" if stem.startswith("sft_expert_") else "out/sft/ckpt.pt"

    if args.lora_rank > 0:
        trainable, total = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f"[LoRA] rank={args.lora_rank} alpha={args.lora_alpha} trainable={trainable:,}/{total:,} ({trainable/total:.2%})")

    optimizers = build_optimizers(model, opt_name="adamw" if args.lora_rank > 0 else args.optimizer, lr_mult=args.lr_mult)
    print(f"[SFT] init={init_path} data={args.data} train={len(train_ex)} val={len(val_ex)} steps={args.steps} -> {out_path}")

    model.train()
    last_val = float("nan")
    loss = torch.tensor(0.0)
    for step in range(args.steps):
        factor = lr_schedule(step, args.steps, warmup_steps=max(1, args.steps // 10), schedule="cosine", min_lr_frac=0.05)
        set_lr_factor(optimizers, factor)
        batch = [rng.choice(train_ex) for _ in range(args.batch_size)]
        x, y = collate_sft(batch, tok.pad_id, device)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _, loss = model(x, y, return_logits=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_all(model, optimizers)

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = collate_sft(val_ex[: min(32, len(val_ex))], tok.pad_id, device)
                _, vloss = model(vx, vy, return_logits=False)
                last_val = vloss.item()
            model.train()
            print(f"step {step+1:>4}/{args.steps} | train_loss {loss.item():.4f} | val_loss {last_val:.4f}")

    if args.lora_rank > 0:
        lora_path = str(Path(out_path).with_name("lora.pt"))
        save_lora_only(model, lora_path)
        print(f"LoRA 增量权重已保存至 {lora_path}")
        if args.lora_merge:
            merge_lora(model)
            print("已完成 LoRA 无损合并 (W + B@A * scaling)")

    save_ckpt(out_path, model, args.steps, stage="sft", extra={"val_loss": last_val, "lora_rank": args.lora_rank})
    print(f"SFT 完成，权重已保存至 {out_path}")
    return {"train_loss": loss.item(), "val_loss": last_val, "out": out_path}


def main():
    p = argparse.ArgumentParser(description="通用 SFT / LoRA 微调 / 领域专家 SFT")
    p.add_argument("--init", default="", help="起始权重；默认优先用 out/anneal/ckpt.pt，其次 out/pretrain/ckpt.pt")
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="", help="输出路径；专家数据默认写到 out/expert_<domain>/ckpt.pt")
    p.add_argument("--lora-rank", type=int, default=0, help=">0 时启用 LoRA 微调（冻结基座只训低秩旁路）")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--no-lora-merge", dest="lora_merge", action="store_false", help="不合并直接保留（默认训练后自动无损合并到 ckpt）")
    p.set_defaults(lora_merge=True)
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
