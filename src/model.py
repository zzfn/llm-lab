"""麻雀虽小、五脏俱全的现代 Decoder-Only 语言模型。

完整包含 tinybrain 全部核心架构机制（均支持开关控制，可做消融对照）：
1. mHC 多流残差连接（n_streams=4 + 20 步对数域 Sinkhorn 双随机矩阵 + StreamEntry/StreamExit 注意力进出）
2. Engram 条件查表记忆（2-gram & 3-gram 多头素数模长哈希查表 + 指定层前门控注入）
3. 3:1 滑动窗口注意力 + GQA 分组查询注意力 + QK-Norm + RoPE + 完整等价的 KV Cache
4. SwiGLU / ReLU² 前馈网络 + 可选 MoE（共享专家 + Top-K 路由 + 无梯度偏置均衡 + 轻均衡损失）
5. 防 Softmax 塌缩稳定分块交叉熵（src/loss.py）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import linear_cross_entropy, stable_cross_entropy


@dataclass
class ModelConfig:
    vocab_size: int = 768
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 512
    ffn_mult: float = 8 / 3
    ffn_type: str = "relu2"        # "relu2" 或 "swiglu"
    qk_norm: bool = True
    rope_base: float = 10000.0
    tie_embeddings: bool = False
    norm_eps: float = 1e-6

    # 3:1 滑动窗口注意力：每 long_every 层最后一层及末层看全文，其余层仅看最近 window 个词
    window: int = 128
    long_every: int = 4

    # mHC 多流残差（n_streams=1 时退化为普通残差但仍走统一代码路径）
    n_streams: int = 4
    sinkhorn_iters: int = 20
    stream_key_dim: int = 64

    # Engram 查表记忆（engram_orders=() 时关闭）
    engram_orders: tuple[int, ...] = (2, 3)
    engram_heads: int = 4
    engram_head_dim: int = 16
    engram_rows: int = 4096
    engram_layers: tuple[int, ...] = (1,)

    # MoE 混合专家（moe_experts=0 时使用普通前馈层）
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_hidden_mult: float = 0.5
    moe_shared_mult: float = 0.5
    moe_dense_layers: int = 1
    moe_balance_rate: float = 1e-3
    moe_balance_loss: float = 1e-4

    def __post_init__(self):
        self.engram_orders = tuple(self.engram_orders)
        self.engram_layers = tuple(self.engram_layers) if self.engram_orders else ()
        assert self.dim % self.n_heads == 0, "dim 必须能被 n_heads 整除"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        assert (self.dim // self.n_heads) % 2 == 0, "head_dim 必须是偶数"
        assert self.ffn_type in ("swiglu", "relu2")
        assert 1 <= self.window <= self.max_seq_len
        assert self.n_streams >= 1 and self.long_every >= 1
        if self.moe_experts:
            assert 1 <= self.moe_top_k <= self.moe_experts
            assert 0 <= self.moe_dense_layers < self.n_layers

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def engram_dim(self) -> int:
        return len(self.engram_orders) * self.engram_heads * self.engram_head_dim

    def window_of(self, layer: int) -> int | None:
        is_long = (layer % self.long_every == self.long_every - 1) or (layer == self.n_layers - 1)
        return None if is_long or self.window >= self.max_seq_len else self.window

    def is_moe_layer(self, layer: int) -> bool:
        return bool(self.moe_experts) and layer >= self.moe_dense_layers


PRESETS: dict[str, dict] = {
    # 约 2.7M 参数（含 mHC + Engram）：1 分钟跑通全套演示
    "pico": dict(
        dim=192, n_layers=4, n_heads=6, n_kv_heads=2, max_seq_len=256, window=128,
        n_streams=4, engram_heads=4, engram_head_dim=16, engram_rows=2048, engram_layers=(1,),
    ),
    # 约 11M 参数（含 mHC + Engram + 3:1 窗口）：M5 本机做完整对比实验与 Agent RLVR 的主力档
    "nano": dict(
        dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512, window=256,
        n_streams=4, engram_heads=4, engram_head_dim=24, engram_rows=4096, engram_layers=(1,),
    ),
    # 约 16M 总参数（激活约 12M），开启 MoE（第 1 层 Dense，其余层 8 选 2 路由专家 + 1 共享专家）
    "nano_moe": dict(
        dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512, window=256,
        n_streams=4, engram_heads=4, engram_head_dim=24, engram_rows=4096, engram_layers=(1,),
        moe_experts=8, moe_top_k=2, moe_hidden_mult=0.5, moe_shared_mult=0.5, moe_dense_layers=1,
    ),
    # 约 28M 参数：更大容量的单机实验档
    "small": dict(
        dim=512, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=1024, window=256,
        n_streams=4, engram_heads=8, engram_head_dim=24, engram_rows=8192, engram_layers=(1, 4),
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


def primes_below(limit: int, count: int) -> list[int]:
    found, n = [], limit
    while len(found) < count:
        if n < 2:
            raise ValueError(f"{limit} 以内不足 {count} 个质数")
        if all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
        n -= 1
    return found


class EngramMemory(nn.Module):
    """N-gram 哈希查表条件记忆，多头使用互异质数模长以分散哈希碰撞。"""

    def __init__(self, vocab_size: int, orders: tuple[int, ...], heads: int, head_dim: int, rows: int, seed: int = 0):
        super().__init__()
        self.orders, self.heads, self.pad_id = orders, heads, vocab_size
        sizes = torch.tensor(primes_below(rows, len(orders) * heads)).view(len(orders), heads)
        gen = torch.Generator().manual_seed(seed)
        mult = torch.randint(1 << 20, 1 << 30, (len(orders), heads, max(orders)), generator=gen) * 2 + 1
        self.register_buffer("sizes", sizes)
        self.register_buffer("offsets", (sizes.flatten().cumsum(0) - sizes.flatten()).view_as(sizes))
        self.register_buffer("mult", mult)
        self.table = nn.Embedding(int(sizes.sum()), head_dim)
        self.norm = RMSNorm(heads * head_dim)

    @property
    def history_len(self) -> int:
        return max(self.orders) - 1

    def slots(self, idx: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        m = self.history_len
        if history is None:
            history = idx.new_full((B, m), self.pad_id)
        ctx = torch.cat([history[:, -m:], idx], dim=1)
        out = []
        for o, n in enumerate(self.orders):
            h = torch.zeros(B, T, self.heads, dtype=torch.long, device=idx.device)
            for j in range(n):
                h = h ^ (ctx[:, m - j : m - j + T, None] * self.mult[o, :, j])
            out.append(h % self.sizes[o] + self.offsets[o])
        return torch.stack(out, dim=2)

    def forward(self, idx: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        e = self.table(self.slots(idx, history))
        return self.norm(e.view(B, T, len(self.orders), -1))


class EngramGate(nn.Module):
    def __init__(self, memory_dim: int, dim: int, n_streams: int):
        super().__init__()
        self.n_streams, self.dim = n_streams, dim
        self.value = nn.Linear(memory_dim, dim, bias=False)
        self.keys = nn.Linear(memory_dim, n_streams * dim, bias=False)
        self.state_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T = memory.shape[:2]
        keys = self.keys(memory).view(B, T, self.n_streams, self.dim)
        gate = torch.sigmoid((self.state_norm(x) * self.key_norm(keys)).sum(-1, keepdim=True) / math.sqrt(self.dim))
        return x + gate * self.value(memory)[:, :, None, :]


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """在对数域迭代按列、按行归一化，生成双随机矩阵（每行每列之和为 1）。"""
    log = logits.float()
    for _ in range(iters):
        log = log - log.logsumexp(dim=-2, keepdim=True)
        log = log - log.logsumexp(dim=-1, keepdim=True)
    return log.exp()


class HyperConnections(nn.Module):
    """mHC 多流残差系数生成器（静态版），初始化等价于标准残差。"""

    def __init__(self, n_sublayers: int, n_streams: int, iters: int = 20):
        super().__init__()
        self.iters = iters
        focus = torch.eye(n_streams)[torch.arange(n_sublayers) % n_streams]
        self.pre_logit = nn.Parameter(4.0 * focus - 2.0)
        self.post_logit = nn.Parameter(torch.zeros(n_sublayers, n_streams))
        self.res_logit = nn.Parameter(5.0 * torch.eye(n_streams).repeat(n_sublayers, 1, 1))

    def mappings(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = torch.sigmoid(self.pre_logit.float()).to(dtype)
        post = (2.0 * torch.sigmoid(self.post_logit.float())).to(dtype)
        res = sinkhorn(self.res_logit, self.iters).to(dtype)
        return pre, post, res


class StreamEntry(nn.Module):
    def __init__(self, dim: int, n_streams: int, key_dim: int):
        super().__init__()
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.key_norm = RMSNorm(key_dim)
        self.query = nn.Parameter(torch.randn(n_streams, key_dim))

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        keys = self.key_norm(self.key(sources))
        weights = torch.softmax(torch.einsum("btsk,mk->btms", keys, self.query.to(keys.dtype)) / math.sqrt(keys.size(-1)), dim=-1)
        return torch.einsum("btms,btsd->btmd", weights, sources)


class StreamExit(nn.Module):
    def __init__(self, dim: int, key_dim: int):
        super().__init__()
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.in_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(key_dim)
        self.query = nn.Parameter(torch.zeros(key_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_norm(self.key(self.in_norm(x)))
        weights = torch.softmax(keys @ self.query.to(keys.dtype) / math.sqrt(keys.size(-1)), dim=-1)
        return torch.einsum("btn,btnd->btd", weights, x)


class KVCache:
    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None


class DecodeState:
    def __init__(self, n_layers: int):
        self.layers = [KVCache() for _ in range(n_layers)]
        self.pos = 0
        self.history: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, window: int | None):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.window = window

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None and cache.k is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)

        if cache is not None:
            keep = slice(None) if self.window is None else slice(-self.window, None)
            cache.k, cache.v = k[:, :, keep], v[:, :, keep]

        # 单步 decode 且已按窗口裁剪时，当前查询可看全部留存的 KV
        if cache is not None and T == 1:
            if self.window is not None and k.size(2) > self.window:
                k = k[:, :, -self.window :]
                v = v[:, :, -self.window :]
            k_rep = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
            v_rep = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
        else:
            S = k.size(2)
            k_rep = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
            v_rep = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v
            if self.window is not None and S > self.window:
                # 滑动窗口因果掩码：0 <= q_abs - k_abs < window
                q_pos = torch.arange(S - T, S, device=x.device)[:, None]
                k_pos = torch.arange(S, device=x.device)[None, :]
                mask = (q_pos >= k_pos) & ((q_pos - k_pos) < self.window)
                out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
            elif cache is not None and S != T:
                mask = torch.ones((T, S), dtype=torch.bool, device=x.device).tril(diagonal=S - T)
                out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
            else:
                out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, ffn_type: str = "relu2"):
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == "swiglu":
            self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.w_down(F.relu(self.w_up(x)).square())


class MoE(nn.Module):
    """混合专家前馈层：共享专家 + Top-K 路由专家 + 无梯度偏置均衡 + 轻均衡损失。"""

    def __init__(self, dim: int, n_experts: int, top_k: int, hidden: int, shared_hidden: int,
                 ffn_type: str = "relu2", balance_rate: float = 1e-3):
        super().__init__()
        self.n_experts, self.top_k, self.balance_rate = n_experts, top_k, balance_rate
        self.shared = FeedForward(dim, shared_hidden, ffn_type)
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.w_in = nn.ParameterList(nn.Parameter(torch.empty(hidden, dim)) for _ in range(n_experts))
        self.w_out = nn.ParameterList(nn.Parameter(torch.zeros(dim, hidden)) for _ in range(n_experts))
        self.register_buffer("balance_bias", torch.zeros(n_experts))
        self.pending_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        z = x.reshape(-1, shape[-1])
        N, E, k = z.size(0), self.n_experts, self.top_k
        scores = torch.sqrt(F.softplus(self.router(z).float()))
        chosen = (scores.detach() + self.balance_bias).topk(k, dim=-1).indices
        gates = scores.gather(-1, chosen)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-9)

        expert = chosen.reshape(-1)
        order = expert.argsort(stable=True)
        token = torch.arange(N, device=z.device).repeat_interleave(k)[order]
        counts = torch.bincount(expert, minlength=E)
        inputs, outputs, start = z[token], [], 0
        for e, count in enumerate(counts.tolist()):
            if count:
                seg = inputs[start : start + count]
                outputs.append(F.relu(seg @ self.w_in[e].t()).square() @ self.w_out[e].t())
                start += count
        routed = torch.cat(outputs) * gates.reshape(-1)[order, None].to(inputs.dtype)
        y = torch.zeros(N, shape[-1], dtype=routed.dtype, device=z.device).index_add_(0, token, routed)

        load = counts.float() / (N * k)
        if self.training:
            self.pending_load = load.detach()
        share = (scores / (scores.sum(-1, keepdim=True) + 1e-9)).mean(0)
        balance_loss = E * (load * share).sum()
        return self.shared(x) + y.view(shape), balance_loss

    @torch.no_grad()
    def apply_balance(self) -> None:
        if self.pending_load is not None:
            self.balance_bias += self.balance_rate * torch.sign(1.0 / self.n_experts - self.pending_load)
            self.pending_load = None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.index = index
        self.engram = EngramGate(cfg.engram_dim, cfg.dim, cfg.n_streams) if index in cfg.engram_layers else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, cfg.window_of(index))
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.is_moe_layer(index):
            hidden = max(64, ((int(cfg.dim * cfg.moe_hidden_mult) + 63) // 64) * 64)
            shared_h = max(64, ((int(cfg.dim * cfg.moe_shared_mult) + 63) // 64) * 64)
            self.ffn: nn.Module = MoE(cfg.dim, cfg.moe_experts, cfg.moe_top_k, hidden, shared_h, cfg.ffn_type, cfg.moe_balance_rate)
        else:
            hidden = max(64, ((int(cfg.dim * cfg.ffn_mult) + 63) // 64) * 64)
            self.ffn = FeedForward(cfg.dim, hidden, cfg.ffn_type)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        memory: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre, post, res = maps
        a, f = 2 * self.index, 2 * self.index + 1
        if self.engram is not None and memory is not None:
            x = self.engram(x, memory)

        h_attn = self.attn_norm(torch.einsum("btnd,n->btd", x, pre[a]))
        attn_out = self.attn(h_attn, cos, sin, cache)
        x = torch.einsum("mn,btnd->btmd", res[a], x) + post[a][:, None] * attn_out[:, :, None, :]

        h_ffn = self.ffn_norm(torch.einsum("btnd,n->btd", x, pre[f]))
        ffn_res = self.ffn(h_ffn)
        ffn_out, bal = ffn_res if isinstance(ffn_res, tuple) else (ffn_res, x.new_zeros(()))
        x = torch.einsum("mn,btnd->btmd", res[f], x) + post[f][:, None] * ffn_out[:, :, None, :]
        return x, bal


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.embed_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.memory = None
        if cfg.engram_orders:
            self.memory = EngramMemory(cfg.vocab_size, cfg.engram_orders, cfg.engram_heads, cfg.engram_head_dim, cfg.engram_rows)
            per_order_dim = cfg.engram_heads * cfg.engram_head_dim
            self.memory_proj = nn.ModuleList(nn.Linear(per_order_dim, cfg.dim, bias=False) for _ in cfg.engram_orders)
            self.memory_norms = nn.ModuleList(RMSNorm(cfg.dim, cfg.norm_eps) for _ in cfg.engram_orders)

        self.entry = StreamEntry(cfg.dim, cfg.n_streams, cfg.stream_key_dim)
        self.hyper = HyperConnections(2 * cfg.n_layers, cfg.n_streams, cfg.sinkhorn_iters)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.exit = StreamExit(cfg.dim, cfg.stream_key_dim)
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.ndim == 1 and "norm" in name:
                continue
            leaf = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1]
            if name in ("embed.weight", "memory.table.weight"):
                nn.init.normal_(p, std=1.0)
            elif name == "head.weight":
                nn.init.normal_(p, std=0.001)
            elif leaf in ("wo", "w_down", "value") or ".w_out." in name:
                nn.init.zeros_(p)
            elif leaf in ("pre_logit", "post_logit", "res_logit", "query"):
                continue
            elif p.ndim >= 2:
                bound = math.sqrt(3.0 / p.size(-1))
                nn.init.uniform_(p, -bound, bound)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        state: DecodeState | None = None,
        return_logits: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        B, T = idx.shape
        pos = state.pos if state is not None else 0
        if pos + T > self.cfg.max_seq_len:
            raise ValueError(f"序列总长 {pos + T} 超出 max_seq_len={self.cfg.max_seq_len}")

        cos = self.rope_cos[pos : pos + T]
        sin = self.rope_sin[pos : pos + T]

        sources = [self.embed_norm(self.embed(idx))]
        memory = None
        if self.memory is not None:
            per_order = self.memory(idx, state.history if state is not None else None)
            for o, (proj, norm) in enumerate(zip(self.memory_proj, self.memory_norms)):
                sources.append(norm(proj(per_order[:, :, o])))
            memory = per_order.flatten(2)

        x = self.entry(torch.stack(sources, dim=2))
        maps = self.hyper.mappings(x.dtype)
        balance = x.new_zeros((), dtype=torch.float32)

        for i, block in enumerate(self.blocks):
            x, bal = block(x, cos, sin, maps, memory, state.layers[i] if state is not None else None)
            balance = balance + bal.float()

        hidden = self.final_norm(self.exit(x))
        want_logits = targets is None if return_logits is None else return_logits
        logits = self.head(hidden).float() if want_logits else None

        loss = None
        if targets is not None:
            bal_term = self.cfg.moe_balance_loss * balance
            if logits is None:
                loss = linear_cross_entropy(hidden, self.head.weight, targets) + bal_term
            else:
                loss = stable_cross_entropy(logits, targets) + bal_term

        if state is not None:
            state.pos = pos + T
            if self.memory is not None and self.memory.history_len:
                past = state.history if state.history is not None else idx.new_full((B, self.memory.history_len), self.memory.pad_id)
                state.history = torch.cat([past, idx], dim=1)[:, -self.memory.history_len :]

        return logits, loss

    def update_balance(self) -> None:
        for mod in self.modules():
            if isinstance(mod, MoE):
                mod.apply_balance()

    def new_state(self) -> DecodeState:
        return DecodeState(self.cfg.n_layers)

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def param_breakdown(self) -> dict[str, int]:
        """按部件统计参数量，方便报告模型构成。"""
        groups = {"词向量与输出层": 0, "Engram记忆表": 0, "多流残差mHC": 0, "注意力": 0, "前馈/MoE": 0, "其他": 0}
        seen = set()
        for name, p in self.named_parameters():
            if p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            n = p.numel()
            if "embed" in name or "head." in name:
                groups["词向量与输出层"] += n
            elif "memory." in name or ".engram." in name:
                groups["Engram记忆表"] += n
            elif "hyper." in name or "entry." in name or "exit." in name:
                groups["多流残差mHC"] += n
            elif ".attn." in name:
                groups["注意力"] += n
            elif ".ffn." in name:
                groups["前馈/MoE"] += n
            else:
                groups["其他"] += n
        return {k: v for k, v in groups.items() if v > 0}
