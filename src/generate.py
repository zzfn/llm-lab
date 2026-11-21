"""带 KV Cache 的自回归生成、Agent 多轮工具调用演示，以及 KV Cache 速度对比测试。"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from .model import MiniLLM
from .tokenizer import ByteBPETokenizer
from .utils import format_chat_prompt, load_ckpt, pick_device


@torch.no_grad()
def generate_ids(
    model: MiniLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    eos_id: int | None = None,
) -> torch.Tensor:
    """prefill 一次送入全部 prompt，后续每步只喂最后一个 token 并复用 KVCache。"""
    was_training = model.training
    model.eval()
    caches = model.new_caches()
    B, T = prompt_ids.shape
    room = min(max_new_tokens, model.cfg.max_seq_len - T)
    if room <= 0:
        return prompt_ids

    logits, _ = model(prompt_ids, caches=caches, start_pos=0)
    next_logits = logits[:, -1, :]
    out = prompt_ids
    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)

    for step in range(room):
        if temperature <= 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = next_logits / temperature
            if top_k > 0 and top_k < scaled.size(-1):
                vals, _ = torch.topk(scaled, top_k, dim=-1)
                scaled = scaled.masked_fill(scaled < vals[:, [-1]], float("-inf"))
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, next_tok], dim=1)
        if eos_id is not None:
            finished |= next_tok.squeeze(-1) == eos_id
            if finished.all():
                break
        if step + 1 < room:
            logits, _ = model(next_tok, caches=caches, start_pos=T + step)
            next_logits = logits[:, -1, :]

    model.train(was_training)
    return out


@torch.no_grad()
def bench_kv_cache(model: MiniLLM, device: torch.device, prompt_len: int = 128, gen_tokens: int = 128) -> None:
    """现场对比开与不开 KV Cache 的生成耗时（团队分享演示用）。"""
    model.eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)

    def _sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # 预热
    _ = generate_ids(model, prompt, max_new_tokens=8, temperature=0.0)
    _sync()

    # 1. 带 KV Cache
    t0 = time.perf_counter()
    _ = generate_ids(model, prompt, max_new_tokens=gen_tokens, temperature=0.0)
    _sync()
    dt_cached = time.perf_counter() - t0

    # 2. 不带 KV Cache（每步重算全量上下文）
    cur = prompt.clone()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        logits, _ = model(cur)
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
    p = argparse.ArgumentParser(description="使用 KV Cache 生成文本或演示 Agent 工具调用")
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
        from .utils import format_tool_turn, parse_tool_calls

        messages = [
            {"role": "system", "content": "你可以使用工具 calc(expr) 计算算术表达式。"},
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
