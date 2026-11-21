"""阶段 1：从零预训练 + 退火（--init 混入对话）+ WSD 分支退火（--cooldown-from）+ 过拟合自检（--overfit）。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .loss import margin_saturation
from .model import PRESETS, MiniLLM, ModelConfig
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_sft, load_ckpt, lr_schedule, pick_device, save_ckpt, set_lr_factor, step_all


def build_stream(text_path: Path, tok: ByteBPETokenizer, val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: list[int] = []
    for line in lines:
        ids.extend([tok.bos_id] + tok.encode_ordinary(line) + [tok.eos_id])
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data) * (1.0 - val_ratio))
    return data[:split], data[split:]


def build_chat_all_stream(jsonl_path: Path, tok: ByteBPETokenizer) -> torch.Tensor:
    """退火期混入的对话流：整段均计算损失（不含特殊 mask）。"""
    ids: list[int] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            msg_ids, _ = format_chat_sft(tok, json.loads(line)["messages"], train_on="all")
            ids.extend(msg_ids)
    return torch.tensor(ids, dtype=torch.long)


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
              device: torch.device, batches: int = 8) -> tuple[float, float]:
    model.eval()
    losses, sat = [], 0.0
    for i in range(batches):
        x, y = sample_batch(val_stream, batch_size, seq_len, device)
        logits, loss = model(x, y, return_logits=(i == 0))
        losses.append(loss.item())
        if i == 0 and logits is not None:
            sat = margin_saturation(logits[:1], y[:1])
    model.train()
    return sum(losses) / len(losses), sat


def run_overfit_check(model: MiniLLM, train_stream: torch.Tensor, seq_len: int, device: torch.device,
                      steps: int = 80) -> dict:
    """单批过拟合自检：验证梯度回传、mHC/Engram/MoE/稳定交叉熵无任何链路 bug。"""
    x, y = sample_batch(train_stream, 2, seq_len, device)
    optimizers = build_optimizers(model, opt_name="muon", lr_mult=1.0)
    model.train()
    loss_val = float("nan")
    for step in range(steps):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_all(model, optimizers)
        loss_val = loss.item()
        if (step + 1) % 20 == 0 or step + 1 == steps:
            print(f"[Overfit Check] step {step+1:>3}/{steps} | loss {loss_val:.4f}")
    return {"final_loss": loss_val}


def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    train_stream, val_stream = build_stream(Path(args.data), tok)
    chat_stream = build_chat_all_stream(Path(args.mix_chat), tok) if args.mix_chat else None

    if args.cooldown_from:
        model, ckpt = load_ckpt(args.cooldown_from, device)
        optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
        if "optimizers" in ckpt and len(ckpt["optimizers"]) == len(optimizers):
            for opt, state in zip(optimizers, ckpt["optimizers"]):
                opt.load_state_dict(state)
        stage = "anneal"
        out_path = args.out if args.out != "out/pretrain/ckpt.pt" else "out/anneal/ckpt.pt"
    elif args.init:
        model, _ = load_ckpt(args.init, device)
        optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
        stage = "anneal"
        out_path = args.out if args.out != "out/pretrain/ckpt.pt" else "out/anneal/ckpt.pt"
    else:
        cfg = ModelConfig(vocab_size=tok.vocab_size, ffn_type=args.ffn, **PRESETS[args.preset])
        model = MiniLLM(cfg).to(device)
        optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
        stage = "pretrain"
        out_path = args.out

    seq_len = min(args.seq_len, model.cfg.max_seq_len)
    if args.overfit:
        return run_overfit_check(model, train_stream, seq_len, device, steps=args.steps)

    print(
        f"[{stage.upper()}] device={device} preset={args.preset} params={model.count_parameters()/1e6:.2f}M "
        f"mHC={model.cfg.n_streams} engram={bool(model.cfg.engram_orders)} moe={model.cfg.moe_experts} "
        f"opt={args.optimizer} schedule={args.schedule} steps={args.steps}"
    )

    t0 = time.perf_counter()
    last_val = float("nan")
    total_loss = 0.0
    for step in range(args.steps):
        if args.cooldown_from:
            # WSD 分支退火：从 1.0 按 1 - sqrt(p) 降到 min_lr_frac
            p = (step + 1) / max(1, args.steps)
            factor = 1.0 - (1.0 - 0.05) * (p ** 0.5)
        else:
            factor = lr_schedule(step, args.steps, args.warmup_steps, schedule=args.schedule)
        set_lr_factor(optimizers, factor)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        for micro in range(args.grad_accum):
            use_chat = chat_stream is not None and (micro == args.grad_accum - 1) and (step % 10 == 0 or args.chat_ratio >= 0.1)
            src_stream = chat_stream if (use_chat and len(chat_stream) > seq_len + 2) else train_stream
            x, y = sample_batch(src_stream, args.batch_size, seq_len, device)
            _, loss = model(x, y, return_logits=False)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_all(model, optimizers)

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            last_val, sat = eval_loss(model, val_stream, args.batch_size, seq_len, device)
            dt = time.perf_counter() - t0
            toks = (step + 1) * args.batch_size * args.grad_accum * seq_len
            print(
                f"step {step+1:>4}/{args.steps} | lr {factor:.3f} | "
                f"train_loss {total_loss:.4f} | val_loss {last_val:.4f} | sat {sat:.1%} | {toks/dt:,.0f} tok/s"
            )

    save_ckpt(out_path, model, args.steps, stage=stage, optimizers=optimizers, extra={"val_loss": last_val})
    print(f"{stage} 完成，权重已保存至 {out_path}")
    return {"train_loss": total_loss, "val_loss": last_val}


def main():
    p = argparse.ArgumentParser(description="从零预训练 / 退火 / WSD 分支退火 / 过拟合检查")
    p.add_argument("--data", default="data/pretrain.txt")
    p.add_argument("--mix-chat", default="", help="退火时混入的对话 jsonl（整段算损失）")
    p.add_argument("--chat-ratio", type=float, default=0.1)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--init", default="", help="从指定权重初始化（退火用，重置优化器）")
    p.add_argument("--cooldown-from", default="", help="WSD 分支退火起点（连优化器状态一起继承）")
    p.add_argument("--out", default="out/pretrain/ckpt.pt")
    p.add_argument("--preset", default="nano", choices=list(PRESETS))
    p.add_argument("--ffn", default="relu2", choices=["swiglu", "relu2"])
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    p.add_argument("--lr-mult", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--overfit", action="store_true", help="小批过拟合诊断检查")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
