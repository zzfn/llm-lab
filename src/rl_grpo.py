"""阶段 3：Dr. GRPO 强化学习（RLVR 多题型：Agent 工具调用 + Instruct 约束遵循 + Choice 选择题 + 可选 RLAIF 评审）。"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import random
import re
import urllib.request
from pathlib import Path

import torch

from .generate import generate_ids
from .loss import token_log_probs
from .tokenizer import ByteBPETokenizer
from .utils import build_optimizers, format_chat_prompt, format_tool_turn, load_ckpt, parse_tool_calls, pick_device, save_ckpt, step_all

ANS_RE = re.compile(r"####\s*(-?\d+)")
CHOICE_RE = re.compile(r"(?:答案[：:\s]*)?\b([A-D])\b")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def safe_eval_calc(expr: str) -> str:
    try:
        node = ast.parse(expr.strip(), mode="eval").body
        def _eval(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and type(n.op) in OPS:
                return OPS[type(n.op)](_eval(n.left), _eval(n.right))
            raise ValueError("不支持的表达式")
        return str(_eval(node))
    except Exception as exc:
        return f"error: {exc}"


def execute_tool(name: str, arguments: dict) -> str:
    if name == "calc":
        return safe_eval_calc(str(arguments.get("expr", "")))
    return f"error: unknown tool {name}"


def score_instruct(text: str, rules: dict) -> float:
    """可程序化验证的指令跟随判分（0..1）。"""
    checks: list[bool] = []
    t = text.strip()
    if "max_chars" in rules:
        checks.append(0 < len(t) <= int(rules["max_chars"]))
    for kw in rules.get("must_include", []):
        checks.append(kw in t)
    if rules.get("no_comma"):
        checks.append("," not in t and "，" not in t)
    if "json_keys" in rules:
        try:
            obj = json.loads(t)
            checks.append(isinstance(obj, dict) and all(k in obj for k in rules["json_keys"]))
        except ValueError:
            checks.append(False)
    return sum(checks) / len(checks) if checks else 0.0


def score_choice(text: str, gold: str) -> float:
    m = CHOICE_RE.search(text.strip())
    return 1.0 if (m and m.group(1) == gold.strip()) else 0.0


def score_judge(prompt: str, response: str, judge_url: str, judge_model: str) -> float:
    """可选 RLAIF：调用 OpenAI 兼容评审服务给 1..10 分并映射到 [-1, 1]。"""
    payload = {
        "model": judge_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "请从准确性、格式与简洁度给助手回答打 1 到 10 分，只输出一个整数。"},
            {"role": "user", "content": f"问题：{prompt}\n\n回答：{response}"},
        ],
    }
    req = urllib.request.Request(
        f"{judge_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"]
        m = re.search(r"\b(10|[1-9])\b", content)
        score = int(m.group(1)) if m else 5
        return (score - 5.5) / 4.5
    except Exception:
        return 0.0


def score_task(task: dict, final_text: str, broken_count: int, judge_url: str = "", judge_model: str = "") -> float:
    kind = task.get("reward", "agent")
    if kind == "agent":
        m = ANS_RE.search(final_text)
        r = 1.0 if (m and m.group(1) == str(task["answer"]).strip()) else (0.1 if m else 0.0)
        return max(-0.5, r - 0.2 * broken_count)
    if kind == "instruct":
        return score_instruct(final_text, task.get("rules", {}))
    if kind == "choice":
        return score_choice(final_text, str(task["answer"]))
    if kind == "judge" and judge_url:
        return score_judge(task["messages"][-1]["content"], final_text, judge_url, judge_model)
    raise ValueError(f"未知奖励类型或缺少 judge_url: {kind}")


def run_episode(
    model,
    tok: ByteBPETokenizer,
    task: dict,
    max_new: int,
    temperature: float,
    device: torch.device,
    max_turns: int = 3,
) -> tuple[list[int], list[int], str, int]:
    ids = format_chat_prompt(tok, task["messages"])
    learn = [0] * len(ids)
    broken_count = 0
    final_text = ""
    turns = max_turns if task.get("reward", "agent") == "agent" else 1

    for _ in range(turns):
        room = min(max_new, model.cfg.max_seq_len - len(ids))
        if room <= 0:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate_ids(model, x, max_new_tokens=room, temperature=temperature, top_k=40, eos_id=tok.im_end_id)
        resp = out[0, len(ids) :].tolist()
        if tok.im_end_id in resp:
            resp = resp[: resp.index(tok.im_end_id) + 1]
        ids.extend(resp)
        learn.extend([1] * len(resp))

        text, calls, broken = parse_tool_calls(tok, resp)
        final_text = text
        broken_count += len(broken)
        if not calls:
            break
        tool_results = [execute_tool(c["name"], c["arguments"]) for c in calls]
        back = format_tool_turn(tok, "\n".join(tool_results))
        if len(ids) + len(back) >= model.cfg.max_seq_len:
            break
        ids.extend(back)
        learn.extend([0] * len(back))

    return ids, learn, final_text, broken_count


def dr_grpo_advantages(rewards: list[float]) -> list[float]:
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


@torch.no_grad()
def evaluate_tasks(model, tok: ByteBPETokenizer, tasks: list[dict], device: torch.device,
                   max_new: int = 64, judge_url: str = "", judge_model: str = "") -> dict[str, float]:
    model.eval()
    by_kind: dict[str, list[float]] = {}
    for task in tasks:
        _, _, final_text, broken = run_episode(model, tok, task, max_new=max_new, temperature=0.0, device=device)
        s = score_task(task, final_text, broken, judge_url, judge_model)
        by_kind.setdefault(task.get("reward", "agent"), []).append(s)
    model.train()
    all_scores = [s for vals in by_kind.values() for s in vals]
    res = {k: sum(v) / len(v) for k, v in by_kind.items()}
    res["overall"] = sum(all_scores) / max(1, len(all_scores))
    return res


def train(args) -> dict:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    init_path = args.init or ("out/mopd/ckpt.pt" if Path("out/mopd/ckpt.pt").exists() else "out/sft/ckpt.pt")
    model, _ = load_ckpt(init_path, device)

    all_tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    train_tasks = [t for t in all_tasks if t.get("split") == "train"]
    eval_tasks = [t for t in all_tasks if t.get("split") == "eval"]

    optimizers = build_optimizers(model, opt_name=args.optimizer, lr_mult=args.lr_mult)
    init_scores = evaluate_tasks(model, tok, eval_tasks, device, max_new=args.max_new, judge_url=args.judge_url, judge_model=args.judge_model)
    print(f"[Dr. GRPO] init={init_path} train={len(train_tasks)} eval={len(eval_tasks)} init_eval={init_scores['overall']:.3f}")

    for step in range(args.steps):
        batch_tasks = rng.sample(train_tasks, k=min(args.prompts_per_step, len(train_tasks)))
        episodes: list[tuple[list[int], list[int], float]] = []
        raw_rewards: list[float] = []

        model.eval()
        with torch.no_grad():
            for task in batch_tasks:
                group_seqs, group_masks, group_rewards = [], [], []
                for _ in range(args.group_size):
                    ids, learn, final_text, broken = run_episode(model, tok, task, max_new=args.max_new, temperature=args.temperature, device=device)
                    group_seqs.append(ids)
                    group_masks.append(learn)
                    group_rewards.append(score_task(task, final_text, broken, args.judge_url, args.judge_model))
                raw_rewards.extend(group_rewards)
                for seq, learn, adv in zip(group_seqs, group_masks, dr_grpo_advantages(group_rewards)):
                    if adv != 0.0 and len(seq) <= model.cfg.max_seq_len:
                        episodes.append((seq, learn, adv))

        loss_val = 0.0
        if episodes:
            model.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            max_t = max(len(seq) for seq, _, _ in episodes)
            n_total = args.prompts_per_step * args.group_size
            ids_t = torch.full((len(episodes), max_t), tok.pad_id, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(episodes), max_t), dtype=torch.float32, device=device)
            adv_t = torch.tensor([adv for _, _, adv in episodes], dtype=torch.float32, device=device)

            for i, (seq, learn, _) in enumerate(episodes):
                ids_t[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask_t[i, : len(learn)] = torch.tensor(learn, dtype=torch.float32, device=device)

            logits, _ = model(ids_t[:, :-1], return_logits=True)
            # 使用防塌缩 token_log_probs（RL 后期概率极高时梯度依然精准）
            target_lp = token_log_probs(logits, ids_t[:, 1:])
            pg_loss = -(adv_t[:, None] * target_lp * mask_t[:, 1:]).sum() / (n_total * args.max_new * 2)
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step_all(model, optimizers)
            loss_val = pg_loss.item()

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            eval_scores = evaluate_tasks(model, tok, eval_tasks, device, max_new=args.max_new, judge_url=args.judge_url, judge_model=args.judge_model)
            avg_r = sum(raw_rewards) / max(1, len(raw_rewards))
            detail = " ".join(f"{k}:{v:.2f}" for k, v in eval_scores.items() if k != "overall")
            print(f"step {step+1:>3}/{args.steps} | avg_reward {avg_r:+.3f} | pg_loss {loss_val:.5f} | eval {eval_scores['overall']:.3f} ({detail})")

    final_scores = evaluate_tasks(model, tok, eval_tasks, device, max_new=args.max_new, judge_url=args.judge_url, judge_model=args.judge_model)
    save_ckpt(args.out, model, args.steps, stage="rl_grpo", extra={"eval_scores": final_scores})
    print(f"Dr. GRPO 完成，最终评测得分: {final_scores['overall']:.3f}，权重已保存至 {args.out}")
    return final_scores


def main():
    p = argparse.ArgumentParser(description="Dr. GRPO 多题型强化学习（RLVR + Agent + 可选 RLAIF）")
    p.add_argument("--init", default="", help="起始权重；默认优先 out/mopd/ckpt.pt，其次 out/sft/ckpt.pt")
    p.add_argument("--tasks", default="data/rl_tasks.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="out/rl/ckpt.pt")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr-mult", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-url", default="", help="可选 OpenAI 兼容评审服务地址（用于 RLAIF）")
    p.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    train(p.parse_args())


if __name__ == "__main__":
    main()
