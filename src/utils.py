"""共用训练工具：设备选择、学习率调度（Cosine / WSD）、混合优化器（Muon+AdamW）、ChatML+Tool 模板与存档读写。"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .model import MiniLLM, ModelConfig
from .muon import Muon
from .tokenizer import ByteBPETokenizer

SYSTEM_PROMPT = "你是 llm-lab 助手，必要时调用 calc(expr)。"


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
    """返回当前步的学习率倍率（0..1）。"""
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
    - adamw: 全模型按是否需要 weight_decay 分组用 AdamW
    - muon: 线性投影/专家矩阵用 Muon；Embedding / Engram 记忆表 / Head / Norm / mHC 标量系数用 AdamW
    """
    if opt_name == "adamw":
        decay, no_decay = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            if p.ndim >= 2 and "embed" not in name and "head" not in name and "memory.table" not in name and "hyper." not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "lr": 3e-4 * lr_mult, "weight_decay": 0.01},
            {"params": no_decay, "lr": 3e-4 * lr_mult, "weight_decay": 0.0},
        ]
        opts = [torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95))]
        for g in opts[0].param_groups:
            g["base_lr"] = g["lr"]
        return opts

    if opt_name == "muon":
        matrix_params, embed_params, engram_params, head_params, scalar_params = [], [], [], [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            if "embed" in name:
                embed_params.append(p)
            elif "memory.table" in name:
                engram_params.append(p)
            elif "head." in name:
                head_params.append(p)
            elif "hyper." in name or name.endswith(".query") or p.ndim != 2:
                scalar_params.append(p)
            else:
                matrix_params.append(p)

        opts: list[torch.optim.Optimizer] = []
        if matrix_params:
            opts.append(Muon(matrix_params, lr=0.02 * lr_mult, weight_decay=0.01))
        adam_groups = []
        if embed_params:
            adam_groups.append({"params": embed_params, "lr": 0.01 * lr_mult, "weight_decay": 0.0})
        if engram_params:
            adam_groups.append({"params": engram_params, "lr": 0.02 * lr_mult, "weight_decay": 0.0})
        if head_params:
            adam_groups.append({"params": head_params, "lr": 0.002 * lr_mult, "weight_decay": 0.0})
        if scalar_params:
            adam_groups.append({"params": scalar_params, "lr": 0.005 * lr_mult, "weight_decay": 0.0})
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


def step_all(model: MiniLLM, optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()
    model.update_balance()


def encode_message_body(tok: ByteBPETokenizer, msg: dict) -> list[int]:
    body: list[int] = []
    if msg.get("thinking"):
        body.extend([tok.think_id] + tok.encode_ordinary(msg["thinking"]) + [tok.think_end_id])
    body.extend(tok.encode_ordinary(msg.get("content", "")))
    for call in msg.get("tool_calls") or []:
        call_json = json.dumps(call, ensure_ascii=False)
        body.extend([tok.tool_call_id] + tok.encode_ordinary(call_json) + [tok.tool_call_end_id])
    body.append(tok.im_end_id)
    return body


def format_chat_sft(
    tok: ByteBPETokenizer,
    messages: list[dict],
    train_on: str = "assistant",
) -> tuple[list[int], list[int]]:
    """把多轮对话编码成 (input_ids, loss_mask)。
    - train_on="assistant": 仅 assistant 的思考、正文、tool_calls 与 <|im_end|> mask=1；其余全为 0
    - train_on="all": 整段均算损失（退火混入对话数据时使用）
    """
    ids: list[int] = []
    mask: list[int] = []
    nl = tok.encode_ordinary("\n")
    everything = train_on == "all"
    for msg in messages:
        role = msg["role"]
        head = [tok.im_start_id] + tok.encode_ordinary(f"{role}\n")
        body = encode_message_body(tok, msg)
        tail = nl
        ids.extend(head + body + tail)
        learn = everything or role == "assistant"
        mask.extend([int(everything)] * len(head) + [int(learn)] * len(body) + [int(everything)] * len(tail))
    return ids, mask


def format_chat_prompt(
    tok: ByteBPETokenizer,
    messages: list[dict],
    enable_thinking: bool | None = None,
) -> list[int]:
    """构造提示词前缀，支持自适应思考开关（True 强制开启 <think>，False 跳过思考，None 模型自决）。"""
    ids: list[int] = []
    nl = tok.encode_ordinary("\n")
    for msg in messages:
        ids.extend([tok.im_start_id] + tok.encode_ordinary(f"{msg['role']}\n"))
        ids.extend(encode_message_body(tok, msg) + nl)
    ids.extend([tok.im_start_id] + tok.encode_ordinary("assistant\n"))
    if enable_thinking is True:
        ids.append(tok.think_id)
    elif enable_thinking is False:
        ids.extend([tok.think_id, tok.think_end_id])
    return ids


def split_thinking(text: str) -> tuple[str, str]:
    """拆分 <think>...</think> 思考过程与最终正文，返回 (reasoning_content, content)。"""
    if "<think>" in text and "</think>" in text:
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>", start)
        return text[start:end].strip(), (text[: text.index("<think>")] + text[end + len("</think>") :]).strip()
    if "</think>" in text:
        end = text.index("</think>")
        return text[:end].strip(), text[end + len("</think>") :].strip()
    return "", text.strip()


def format_tool_turn(tok: ByteBPETokenizer, result_text: str) -> list[int]:
    nl = tok.encode_ordinary("\n")
    return (
        [tok.im_start_id]
        + tok.encode_ordinary("tool\n")
        + tok.encode_ordinary(result_text)
        + [tok.im_end_id]
        + nl
        + [tok.im_start_id]
        + tok.encode_ordinary("assistant\n")
    )


def parse_tool_calls(tok: ByteBPETokenizer, resp_ids: list[int]) -> tuple[str, list[dict], list[str]]:
    ids = list(resp_ids)
    if tok.im_end_id in ids:
        ids = ids[: ids.index(tok.im_end_id)]
    text_ids: list[int] = []
    calls: list[dict] = []
    broken: list[str] = []
    i = 0
    while i < len(ids):
        if ids[i] == tok.tool_call_id:
            if tok.tool_call_end_id in ids[i + 1 :]:
                j = ids.index(tok.tool_call_end_id, i + 1)
                raw = tok.decode(ids[i + 1 : j], skip_special=True)
                try:
                    obj = json.loads(raw)
                    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                        raise ValueError
                    calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
                except ValueError:
                    broken.append(raw)
                i = j + 1
            else:
                broken.append(tok.decode(ids[i + 1 :], skip_special=True))
                break
        else:
            if ids[i] not in (tok.pad_id, tok.bos_id, tok.eos_id, tok.im_start_id, tok.im_end_id):
                text_ids.append(ids[i])
            i += 1
    return tok.decode(text_ids, skip_special=False), calls, broken


def save_ckpt(
    path: str | Path,
    model: MiniLLM,
    step: int,
    stage: str,
    optimizers: list[torch.optim.Optimizer] | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "step": step,
        "stage": stage,
        "extra": extra or {},
    }
    if optimizers is not None:
        payload["optimizers"] = [opt.state_dict() for opt in optimizers]
    torch.save(payload, path)


def load_ckpt(path: str | Path, device: torch.device) -> tuple[MiniLLM, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
