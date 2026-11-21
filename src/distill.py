"""阶段 2.5：MOPD（多老师在线蒸馏）。

原理：
1. 学生自己在线生成回答 y ~ pi_student（消除普通离线蒸馏的暴露偏差 exposure bias）
2. 按题目 domain 路由到对应领域专家老师 T(d)（如 math -> expert_math, qa -> expert_qa）
3. 老师和学生共用同一词表，在学生写的每个 token 上精确最小化全词表反向 KL:
   KL(student || teacher) = sum_v pi_s(v|h) * [log pi_s(v|h) - log pi_T(v|h)]
4. 可选 --merge-init（CA-MOPD 任务算术初始化）：先把学生移到各专家参数平均处再蒸馏。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .generate import generate_ids
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_prompt, load_ckpt, pick_device, save_ckpt, set_lr_factor, step_all


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    return (s.exp() * (s - t)).sum(dim=-1)


@torch.no_grad()
def merge_into(student, experts: list) -> None:
    """CA-MOPD 任务算术初始化：student <- student + mean(expert - student)。"""
    params = dict(student.named_parameters())
    for name, p in params.items():
        delta = sum(dict(e.named_parameters())[name] - p for e in experts) / len(experts)
        p.add_(delta)


def parse_teachers(pairs: list[str] | None) -> dict[str, str]:
    if pairs:
        out = {}
        for p in pairs:
            k, sep, v = p.partition("=")
            if not sep:
                raise ValueError(f"--teachers 需写成 domain=path，收到 {p!r}")
            out[k] = v
        return out
    found = {}
    for path in sorted(Path("out").glob("expert_*/ckpt.pt")):
        domain = path.parent.name.removeprefix("expert_")
        found[domain] = str(path)
    if not found:
        raise FileNotFoundError("未找到 out/expert_*/ckpt.pt，请先训练领域专家或传入 --teachers")
    return found


def train(args) -> dict:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)

    teacher_map = parse_teachers(args.teachers)
    student, _ = load_ckpt(args.init, device)
    teachers = {d: load_ckpt(p, device)[0].eval() for d, p in teacher_map.items()}
    for t in teachers.values():
        t.requires_grad_(False)
    if args.merge_init:
        merge_into(student, list(teachers.values()))

    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    prompts = [p for p in prompts if p.get("domain") in teachers]
    if not prompts:
        raise ValueError(f"没有匹配老师 {list(teachers)} 的 MOPD 提示")

    optimizers = build_optimizers(student, opt_name=args.optimizer, lr_mult=args.lr_mult)
    print(f"[MOPD] device={device} teachers={list(teachers)} prompts={len(prompts)} steps={args.steps}")

    last_kl = 0.0
    for step in range(args.steps):
        set_lr_factor(optimizers, min(1.0, (step + 1) / max(1, args.warmup_steps)))
        batch = rng.sample(prompts, k=min(args.prompts_per_step, len(prompts)))

        by_domain: dict[str, list[tuple[list[int], list[int]]]] = {}
        student.eval()
        with torch.no_grad():
            for p in batch:
                prompt_ids = format_chat_prompt(tok, p["messages"])
                x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                out = generate_ids(student, x, max_new_tokens=args.max_new, temperature=args.temperature, eos_id=tok.im_end_id)
                resp = out[0, len(prompt_ids) :].tolist()
                if tok.im_end_id in resp:
                    resp = resp[: resp.index(tok.im_end_id) + 1]
                seq = prompt_ids + resp
                mask = [0] * len(prompt_ids) + [1] * len(resp)
                if len(seq) <= student.cfg.max_seq_len and any(mask):
                    by_domain.setdefault(p["domain"], []).append((seq, mask))

        n_tokens = sum(sum(m) for eps in by_domain.values() for _, m in eps)
        if n_tokens == 0:
            continue

        student.train()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        total_kl = 0.0
        for domain, eps in by_domain.items():
            max_t = max(len(s) for s, _ in eps)
            ids_t = torch.full((len(eps), max_t), tok.pad_id, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(eps), max_t), dtype=torch.float32, device=device)
            for i, (seq, m) in enumerate(eps):
                ids_t[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask_t[i, : len(m)] = torch.tensor(m, dtype=torch.float32, device=device)

            s_logits, _ = student(ids_t[:, :-1], return_logits=True)
            with torch.no_grad():
                t_logits, _ = teachers[domain](ids_t[:, :-1], return_logits=True)
            kl = reverse_kl(s_logits, t_logits) * mask_t[:, 1:]
            loss = kl.sum() / n_tokens
            loss.backward()
            total_kl += kl.sum().item()

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        step_all(student, optimizers)
        last_kl = total_kl / n_tokens
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"step {step+1:>3}/{args.steps} | reverse_kl {last_kl:.4f} | resp_tokens {n_tokens}")

    save_ckpt(args.out, student, args.steps, stage="mopd", extra={"reverse_kl": last_kl})
    print(f"MOPD 完成，权重已保存至 {args.out}")
    return {"reverse_kl": last_kl}


def main():
    p = argparse.ArgumentParser(description="MOPD 多老师在线蒸馏")
    p.add_argument("--init", default="out/sft/ckpt.pt")
    p.add_argument("--prompts", default="data/mopd_prompts.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--teachers", nargs="+", default=None, help="格式 domain=ckpt_path，默认自动读取 out/expert_*/ckpt.pt")
    p.add_argument("--merge-init", action="store_true", help="先做 CA-MOPD 参数均值融合再蒸馏")
    p.add_argument("--out", default="out/mopd/ckpt.pt")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--prompts-per-step", type=int, default=6)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
