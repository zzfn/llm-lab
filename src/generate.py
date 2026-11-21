"""带 KV Cache 的自回归生成与命令行对话。"""
from __future__ import annotations

import argparse

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


def main():
    p = argparse.ArgumentParser(description="使用 KV Cache 生成文本或对话")
    p.add_argument("--ckpt", default="out/rl/ckpt.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--prompt", default="计算 15 * 12 等于多少？请给出最终数字，格式为 #### 答案")
    p.add_argument("--mode", default="chat", choices=["chat", "raw"])
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, ckpt = load_ckpt(args.ckpt, device)

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
