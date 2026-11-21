"""共用训练工具：设备选择、学习率调度（Cosine / WSD）、混合优化器、ChatML 模板与存档读写。"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch

from .model import MiniLLM, ModelConfig
from .muon import Muon
from .tokenizer import ByteBPETokenizer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_schedule(step: int, total_steps: int, warmup_steps: int, schedule: str = "wsd",
                decay_frac: float = 0.2, min_lr_frac: float = 0.1) -> float:
    """返回当前步的学习率倍率（0..1）。
    - cosine: 线性预热 + 余弦衰减到 min_lr_frac
    - wsd: 线性预热 + 恒定高原期 + 最后 decay_frac 比例按 1 - sqrt(p) 退火到 min_lr_frac
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "wsd":
        decay_steps = max(1, int(total_steps * decay_frac))
        stable_end = max(warmup_steps, total_steps - decay_steps)
        if step < stable_end:
            return 1.0
        p = min(1.0, (step - stable_end + 1) / decay_steps)
        return 1.0 - (1.0 - min_lr_frac) * math.sqrt(p)
    raise ValueError(f"未知调度策略: {schedule}")


def build_optimizers(model: MiniLLM, opt_name: str = "muon", lr_mult: float = 1.0) -> list[torch.optim.Optimizer]:
    """构建优化器：
    - adamw: 全模型按是否需要 weight_decay 分两组用 AdamW
    - muon: 隐藏层 2D 矩阵用 Muon；Embedding / Head / Norm 等用 AdamW
    """
    if opt_name == "adamw":
        decay, no_decay = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            if p.ndim >= 2 and "embed" not in name and "head" not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "lr": 3e-4 * lr_mult, "weight_decay": 0.01},
            {"params": no_decay, "lr": 3e-4 * lr_mult, "weight_decay": 0.0},
        ]
        return [torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95))]

    if opt_name == "muon":
        matrix_params, embed_params, head_params, norm_params = [], [], [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            if "embed" in name:
                embed_params.append(p)
            elif "head" in name:
                head_params.append(p)
            elif p.ndim == 2:
                matrix_params.append(p)
            else:
                norm_params.append(p)

        opts: list[torch.optim.Optimizer] = []
        if matrix_params:
            opts.append(Muon(matrix_params, lr=0.02 * lr_mult, weight_decay=0.01))
        adam_groups = []
        if embed_params:
            adam_groups.append({"params": embed_params, "lr": 0.01 * lr_mult, "weight_decay": 0.0})
        if head_params:
            adam_groups.append({"params": head_params, "lr": 0.002 * lr_mult, "weight_decay": 0.0})
        if norm_params:
            adam_groups.append({"params": norm_params, "lr": 0.003 * lr_mult, "weight_decay": 0.0})
        if adam_groups:
            opts.append(torch.optim.AdamW(adam_groups, betas=(0.85, 0.95)))
        for opt in opts:
            for g in opt.param_groups:
                g["base_lr"] = g["lr"]
        return opts

    raise ValueError(f"未知优化器: {opt_name}")


def set_lr_factor(optimizers: list[torch.optim.Optimizer], factor: float) -> None:
    for opt in optimizers:
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * factor


def format_chat_sft(
    tok: ByteBPETokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """把多轮对话编码成 (input_ids, loss_mask)。
    只有 assistant 的回复内容和其后的 <|im_end|> 的 mask 为 1，其余全为 0。
    """
    ids: list[int] = []
    mask: list[int] = []
    nl = tok.encode_ordinary("\n")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        head = [tok.im_start_id] + tok.encode_ordinary(f"{role}\n")
        body = tok.encode_ordinary(content) + [tok.im_end_id]
        tail = nl
        ids.extend(head + body + tail)
        if role == "assistant":
            mask.extend([0] * len(head) + [1] * len(body) + [0] * len(tail))
        else:
            mask.extend([0] * (len(head) + len(body) + len(tail)))
    return ids, mask


def format_chat_prompt(tok: ByteBPETokenizer, messages: list[dict[str, str]]) -> list[int]:
    """构造推理/RL 提示词前缀，以 `<|im_start|>assistant\n` 结尾。"""
    ids: list[int] = []
    nl = tok.encode_ordinary("\n")
    for msg in messages:
        ids.extend([tok.im_start_id] + tok.encode_ordinary(f"{msg['role']}\n"))
        ids.extend(tok.encode_ordinary(msg["content"]) + [tok.im_end_id] + nl)
    ids.extend([tok.im_start_id] + tok.encode_ordinary("assistant\n"))
    return ids


def save_ckpt(path: str | Path, model: MiniLLM, step: int, stage: str, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "step": step,
            "stage": stage,
            "extra": extra or {},
        },
        path,
    )


def load_ckpt(path: str | Path, device: torch.device) -> tuple[MiniLLM, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
