"""带 DecodeState（KV Cache + Engram 历史）的自回归生成、流式输出、Agent 多轮工具调用与 KV Cache 测速。"""
from __future__ import annotations

import argparse
import time
from typing import Iterator

import torch
import torch.nn.functional as F

from .model import MiniLLM
from .tokenizer import ByteBPETokenizer
from .utils import format_chat_prompt, load_ckpt, pick_device


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / temperature
    if 0 < top_k < scaled.size(-1):
        vals, _ = torch.topk(scaled, top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < vals[:, [-1]], float("-inf"))
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def stream_ids(
    model: MiniLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    eos_id: int | None = None,
) -> Iterator[int]:
    """单条请求流式生成 token id（供 serve.py SSE 边生成边推送使用）。"""
    was_training = model.training
    model.eval()
    try:
        state = model.new_state()
        T = prompt_ids.size(1)
        room = min(max_new_tokens, model.cfg.max_seq_len - T)
        if room <= 0:
            return
        logits, _ = model(prompt_ids, state=state, return_logits=True)
        next_logits = logits[:, -1, :]
        for step in range(room):
            next_tok = _sample_next(next_logits, temperature, top_k)
            tok_id = int(next_tok.item())
            yield tok_id
            if eos_id is not None and tok_id == eos_id:
                break
            if step + 1 < room:
                logits, _ = model(next_tok, state=state, return_logits=True)
                next_logits = logits[:, -1, :]
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_ids(
    model: MiniLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    eos_id: int | None = None,
) -> torch.Tensor:
    """批量 prefill + 逐词 decode（带 KV Cache 与 Engram 历史状态）。"""
    was_training = model.training
    model.eval()
    state = model.new_state()
    B, T = prompt_ids.shape
    room = min(max_new_tokens, model.cfg.max_seq_len - T)
    if room <= 0:
        return prompt_ids

    logits, _ = model(prompt_ids, state=state, return_logits=True)
    next_logits = logits[:, -1, :]
    out = prompt_ids
    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)

    for step in range(room):
        next_tok = _sample_next(next_logits, temperature, top_k)
        out = torch.cat([out, next_tok], dim=1)
        if eos_id is not None:
            finished |= next_tok.squeeze(-1) == eos_id
            if finished.all():
                break
        if step + 1 < room:
            logits, _ = model(next_tok, state=state, return_logits=True)
            next_logits = logits[:, -1, :]

    model.train(was_training)
    return out


@torch.no_grad()
def bench_kv_cache(model: MiniLLM, device: torch.device, prompt_len: int = 128, gen_tokens: int = 128) -> None:
    model.eval()
    prompt_len = min(prompt_len, model.cfg.max_seq_len // 2)
    gen_tokens = min(gen_tokens, model.cfg.max_seq_len - prompt_len)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)

    def _sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    _ = generate_ids(model, prompt, max_new_tokens=8, temperature=0.0)
    _sync()

    t0 = time.perf_counter()
    _ = generate_ids(model, prompt, max_new_tokens=gen_tokens, temperature=0.0)
    _sync()
    dt_cached = time.perf_counter() - t0

    cur = prompt.clone()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        logits, _ = model(cur, return_logits=True)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    _sync()
    dt_nocache = time.perf_counter() - t0

    print(
        f"[KV Cache 对比] prompt={prompt_len}, gen={gen_tokens} tokens, device={device}\n"
        f"  无 KV Cache: {dt_nocache*1000:.1f} ms ({gen_tokens/dt_nocache:.1f} tok/s)\n"
        f"  有 KV Cache: {dt_cached*1000:.1f} ms ({gen_tokens/dt_cached:.1f} tok/s) -> 提速 {dt_nocache/dt_cached:.2f}x"
    )


def main():
    p = argparse.ArgumentParser(description="KV Cache 生成 / Agent 工具调用 / 测速对比")
    p.add_argument("--ckpt", default="out/rl/ckpt.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--prompt", default="请使用计算器计算 27 * 34，并按 #### 答案 格式输出。")
    p.add_argument("--mode", default="agent", choices=["agent", "chat", "raw", "bench-cache"])
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, ckpt = load_ckpt(args.ckpt, device)

    if args.mode == "bench-cache":
        bench_kv_cache(model, device)
        return

    if args.mode == "agent":
        from .rl_grpo import execute_tool
        from .utils import SYSTEM_PROMPT, format_tool_turn, parse_tool_calls

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ]
        ids = format_chat_prompt(tok, messages)
        print(f"[stage={ckpt.get('stage')}] User: {args.prompt}")
        for turn in range(3):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = generate_ids(model, x, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k, eos_id=tok.im_end_id)
            resp = out[0, len(ids) :].tolist()
            if tok.im_end_id in resp:
                resp = resp[: resp.index(tok.im_end_id) + 1]
            ids.extend(resp)
            text, calls, broken = parse_tool_calls(tok, resp)
            print(f"  [Turn {turn+1} Assistant] {text}" + (f" | calls={calls}" if calls else "") + (f" | broken={broken}" if broken else ""))
            if not calls:
                break
            res = "\n".join(execute_tool(c["name"], c["arguments"]) for c in calls)
            print(f"  [Turn {turn+1} Tool     ] {res}")
            ids.extend(format_tool_turn(tok, res))
        return

    if args.mode == "chat":
        ids = format_chat_prompt(tok, [{"role": "user", "content": args.prompt}])
        stop = tok.im_end_id
    else:
        ids = [tok.bos_id] + tok.encode_ordinary(args.prompt)
        stop = tok.eos_id

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate_ids(model, x, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k, eos_id=stop)
    reply = tok.decode(out[0, len(ids) :].tolist(), skip_special=True)
    print(f"[stage={ckpt.get('stage')}] Prompt: {args.prompt}\nOutput: {reply}")


if __name__ == "__main__":
    main()
