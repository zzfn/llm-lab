"""极简字节级 BPE 分词器（纯 Python 实现，约 130 行）。

核心原理（面试高频）：
1. 把文本转成 UTF-8 字节序列（0..255），天然做到 byte-fallback，永不出现 <unk>。
2. 训练阶段：统计相邻 token 对的出现频率，反复把最高频的 pair 合并成新 token，直到达到目标词表大小。
3. 编码阶段：先按正则/空白粗切或直接按字节切分，再按训练学到的 merge 优先级贪心合并。
4. 特殊控制符（如 <|im_start|>、<|im_end|>）不参与普通文本切分，直接映射为固定高 ID，避免注入歧义。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>"]


def _get_stats(ids_list: list[list[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for ids in ids_list:
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int] | None = None):
        # 0..255 是单字节基础词表
        self.merges: dict[tuple[int, int], int] = merges or {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        base = 256 + len(self.merges)
        self.special_to_id: dict[str, int] = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_to_id)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def im_start_id(self) -> int:
        return self.special_to_id["<|im_start|>"]

    @property
    def im_end_id(self) -> int:
        return self.special_to_id["<|im_end|>"]

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 512) -> "ByteBPETokenizer":
        """在给定语料上训练 BPE merge 表。"""
        min_size = 256 + len(SPECIAL_TOKENS)
        if vocab_size < min_size:
            raise ValueError(f"vocab_size 至少需要 {min_size}，收到 {vocab_size}")
        num_merges = vocab_size - min_size
        # 按行/段转为 UTF-8 字节 ID 列表
        chunks = [list(t.encode("utf-8")) for t in texts if t]
        merges: dict[tuple[int, int], int] = {}
        for i in range(num_merges):
            stats = _get_stats(chunks)
            if not stats:
                break
            best_pair = max(stats, key=lambda p: stats[p])
            if stats[best_pair] < 2:
                break
            new_id = 256 + i
            merges[best_pair] = new_id
            chunks = [_merge(c, best_pair, new_id) for c in chunks]
        return cls(merges=merges)

    def encode_ordinary(self, text: str) -> list[int]:
        """编码普通文本（不解析特殊控制符，防止提示词注入）。"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats([ids])
            # 按 merge 学到的先后顺序（new_id 越小优先级越高）合并
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if not allow_special:
            return self.encode_ordinary(text)
        # 简单扫描特殊 token
        out: list[int] = []
        i = 0
        while i < len(text):
            matched = None
            for tok in SPECIAL_TOKENS:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                out.append(self.special_to_id[matched])
                i += len(matched)
            else:
                j = i + 1
                while j < len(text) and not any(text.startswith(tok, j) for tok in SPECIAL_TOKENS):
                    j += 1
                out.extend(self.encode_ordinary(text[i:j]))
                i = j
        return out

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        buf = bytearray()
        parts: list[str] = []
        for idx in ids:
            if idx in self.id_to_special:
                if buf:
                    parts.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                if not skip_special:
                    parts.append(self.id_to_special[idx])
            else:
                buf.extend(self.vocab[idx])
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "special_tokens": SPECIAL_TOKENS,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        return cls(merges=merges)
