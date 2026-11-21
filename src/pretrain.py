"""阶段 1：从零预训练（Next-Token Prediction）。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .model import PRESETS, MiniLLM, ModelConfig
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, lr_schedule, pick_device, save_ckpt, set_lr_factor


def build_stream(text_path: Path, tok: ByteBPETokenizer, val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: list[int] = []
    for line in lines:
        ids.extend([tok.bos_id] + tok.encode_ordinary(line) + [tok.eos_id])
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data) * (1.0 - val_ratio))
    return data[:split], data[split:]


def sample_batch(stream: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(stream) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"语料过短 ({len(stream)} tokens)，至少需要 seq_len + 2 = {seq_len + 2}")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([stream[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([stream[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model: MiniLLM, val_stream: torch.Tensor, batch_size: int, seq_len: int,
              device: torch.device, batches: int = 10) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(val_stream, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    train_stream, val_stream = build_stream(Path(args.data), tok)

    cfg = ModelConfig(vocab_size=tok.vocab_size, ffn_type=args.ffn, **PRESETS[args.preset])
    model = MiniLLM(cfg).to(device)
    optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
    seq_len = min(args.seq_len, cfg.max_seq_len)

    print(
        f"[Pretrain] device={device} preset={args.preset} params={model.count_parameters()/1e6:.2f}M "
        f"opt={args.optimizer} schedule={args.schedule} train_tokens={len(train_stream):,} steps={args.steps}"
    )

    t0 = time.perf_counter()
    last_val = float("nan")
    for step in range(args.steps):
        factor = lr_schedule(step, args.steps, args.warmup_steps, schedule=args.schedule)
        set_lr_factor(optimizers, factor)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = sample_batch(train_stream, args.batch_size, seq_len, device)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            last_val = eval_loss(model, val_stream, args.batch_size, seq_len, device)
            dt = time.perf_counter() - t0
            toks = (step + 1) * args.batch_size * args.grad_accum * seq_len
            print(
                f"step {step+1:>4}/{args.steps} | lr_scale {factor:.3f} | "
                f"train_loss {total_loss:.4f} | val_loss {last_val:.4f} | {toks/dt:,.0f} tok/s"
            )

    save_ckpt(args.out, model, args.steps, stage="pretrain", extra={"val_loss": last_val})
    print(f"预训练完成，权重已保存至 {args.out}")
    return {"train_loss": total_loss, "val_loss": last_val}


def main():
    p = argparse.ArgumentParser(description="从零预训练 MiniLLM")
    p.add_argument("--data", default="data/pretrain.txt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/pretrain/ckpt.pt")
    p.add_argument("--preset", default="pico", choices=list(PRESETS))
    p.add_argument("--ffn", default="swiglu", choices=["swiglu", "relu2"])
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    p.add_argument("--lr-mult", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
