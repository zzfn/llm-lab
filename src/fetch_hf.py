"""从 HuggingFace Datasets Server 拉取少量高质量中文后训练语料（零额外依赖）。

默认每个来源只抽几百条短样本（总量约数 MB），适合在 MacBook M5 或 Colab 上直接混入 SFT/Pretrain：
  1. opencsg/smoltalk-chinese                   中文多轮对话
  2. m-a-p/COIG-CQIA (zhihu / ruozhiba)         中文精选问答与脑筋急转弯
  3. Congliu/Chinese-DeepSeek-R1-Distill-data-110k  带 reasoning_content (<think>) 的蒸馏推理数据
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .tokenizer import ByteBPETokenizer
from .utils import SYSTEM_PROMPT

HF_API = "https://datasets-server.huggingface.co/rows"


def _fetch_slice(dataset: str, config: str, offset: int, length: int) -> list[dict]:
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": length})
    url = f"{HF_API}?{qs}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-lab/0.2"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return [item["row"] for item in json.loads(resp.read()).get("rows", [])]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_rows(dataset: str, config: str, target: int, max_scan: int = 1200) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < target and offset < max_scan:
        batch = min(100, max_scan - offset)
        got = _fetch_slice(dataset, config, offset, batch)
        if not got:
            break
        rows.extend(got)
        offset += len(got)
    return rows


def to_sft_dialogues(limit_per_source: int = 300, max_chars: int = 450) -> list[dict]:
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    out: list[dict] = []

    # 1. opencsg/smoltalk-chinese
    for row in fetch_rows("opencsg/smoltalk-chinese", "default", target=limit_per_source * 2, max_scan=1200):
        conv = row.get("conversations") or []
        msgs = [sys_msg]
        total_len = 0
        valid = True
        for turn in conv:
            role, content = turn.get("role"), (turn.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                valid = False
                break
            total_len += len(content)
            msgs.append({"role": role, "content": content})
        if valid and len(msgs) >= 3 and total_len <= max_chars:
            out.append({"messages": msgs, "source": "smoltalk-chinese"})
            if sum(1 for x in out if x["source"] == "smoltalk-chinese") >= limit_per_source:
                break

    # 2. m-a-p/COIG-CQIA (zhihu + ruozhiba)
    coig_count = 0
    for cfg in ("ruozhiba", "zhihu"):
        for row in fetch_rows("m-a-p/COIG-CQIA", cfg, target=limit_per_source, max_scan=800):
            q = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            a = (row.get("output") or "").strip()
            if inp:
                q = f"{q}\n{inp}" if q else inp
            if q and a and len(q) + len(a) <= max_chars:
                out.append({"messages": [sys_msg, {"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": f"coig-{cfg}"})
                coig_count += 1
                if coig_count >= limit_per_source:
                    break

    # 3. Congliu/Chinese-DeepSeek-R1-Distill-data-110k (带 <think>)
    r1_count = 0
    for row in fetch_rows("Congliu/Chinese-DeepSeek-R1-Distill-data-110k", "default", target=limit_per_source * 3, max_scan=1800):
        q = (row.get("input") or "").strip()
        think = (row.get("reasoning_content") or "").strip()
        a = (row.get("content") or "").strip()
        if q and a and len(q) + len(think) + len(a) <= max_chars:
            asst = {"role": "assistant", "content": a}
            if think:
                asst["thinking"] = think
            out.append({"messages": [sys_msg, {"role": "user", "content": q}, asst], "source": "r1-distill"})
            r1_count += 1
            if r1_count >= limit_per_source:
                break

    return out


def main():
    p = argparse.ArgumentParser(description="从 HuggingFace 拉取轻量中文后训练语料")
    p.add_argument("--out", default="data")
    p.add_argument("--limit", type=int, default=200, help="每个来源最多保留多少条短样本")
    p.add_argument("--max-chars", type=int, default=800, help="单条对话总字符数上限（适配小模型上下文）")
    p.add_argument("--vocab-size", type=int, default=2048, help="重训词表大小（0 表示不重训）")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_rows = to_sft_dialogues(limit_per_source=args.limit, max_chars=args.max_chars)

    hf_path = out_dir / "sft_hf.jsonl"
    with hf_path.open("w", encoding="utf-8") as f:
        for r in hf_rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    # 顺便与本地 sft.jsonl 合并成 sft_mixed.jsonl，并把文本追加进 pretrain_mixed.txt
    mixed_rows = [{"messages": r["messages"]} for r in hf_rows]
    local_sft = out_dir / "sft.jsonl"
    if local_sft.exists():
        mixed_rows.extend(json.loads(l) for l in local_sft.read_text(encoding="utf-8").splitlines() if l.strip())
    mixed_path = out_dir / "sft_mixed.jsonl"
    with mixed_path.open("w", encoding="utf-8") as f:
        for r in mixed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pretrain_lines = []
    local_pre = out_dir / "pretrain.txt"
    if local_pre.exists():
        pretrain_lines.extend(l for l in local_pre.read_text(encoding="utf-8").splitlines() if l.strip())
    for r in hf_rows:
        text = " ".join(m.get("content", "").replace("\n", " ") for m in r["messages"] if m["role"] in ("user", "assistant"))
        if text:
            pretrain_lines.append(text)
    (out_dir / "pretrain_mixed.txt").write_text("\n".join(pretrain_lines), encoding="utf-8")

    by_src: dict[str, int] = {}
    for r in hf_rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"拉取完成: {hf_path} 共 {len(hf_rows)} 条（分布: {by_src}），混合语料: {mixed_path} 共 {len(mixed_rows)} 条")

    if args.vocab_size > 0:
        sample_texts = pretrain_lines[:1500]
        tok = ByteBPETokenizer.train(sample_texts, vocab_size=args.vocab_size)
        tok.save(out_dir / "tokenizer_hf.json")
        print(f"已基于混合语料训练分词器: {out_dir / 'tokenizer_hf.json'} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
