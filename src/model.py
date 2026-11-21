"""从零实现的现代 Decoder-Only Transformer 语言模型。

包含面试最常考的组件：
1. Pre-Norm + RMSNorm（去掉均值中心化，仅保留方差缩放，计算更省、训练更稳）
2. RoPE 旋转位置编码（用复数/二维旋转把绝对位置编码成相对位置内积）
3. GQA 分组查询注意力 + 可选 QK-Norm（显著降低 KV Cache 显存占用）
4. SwiGLU / ReLU² 前馈网络（可切换对比）
5. KV Cache 增量推理（支持 prefill + decode，且提供单元测试保证与全量前向严格等价）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 512
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 512
    ffn_mult: float = 8 / 3  # SwiGLU 经典缩放（约 2.67x，与 4x 标准 FFN 参数量对齐）
    ffn_type: str = "swiglu"  # "swiglu" 或 "relu2"
    qk_norm: bool = True
    rope_base: float = 10000.0
    tie_embeddings: bool = False
    norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim 必须能被 n_heads 整除"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        assert (self.dim // self.n_heads) % 2 == 0, "head_dim 必须是偶数（RoPE 成对旋转）"
        assert self.ffn_type in ("swiglu", "relu2")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


PRESETS: dict[str, dict] = {
    # M5 32GB 本机 2~10 分钟可跑完学习闭环
    "pico": dict(dim=192, n_layers=4, n_heads=6, n_kv_heads=2, max_seq_len=256),
    # 约 10M~15M 参数，适合在 Mac 上做消融对比与 RLVR 实验
    "nano": dict(dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512),
    # 约 50M 参数，适合单卡 GPU 或长时间 MPS 实验
    "small": dict(dim=512, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=1024),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 求均方根，防止半精度下平方溢出或精度损失
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算每个位置和维度对的 (cos, sin)，形状 (max_seq_len, head_dim // 2)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, head_dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, n_heads, T, head_dim) 的 Q/K 施加旋转位置编码。

    将相邻偶奇维看作二维向量 (x0, x1)，乘以旋转矩阵：
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos
    """
    cos = cos.to(x.dtype)[None, None, :, :]  # (1, 1, T, D/2)
    sin = sin.to(x.dtype)[None, None, :, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack((out0, out1), dim=-1).flatten(-2)


class KVCache:
    """单层注意力 KV 缓存。"""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

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

        if cache is not None:
            k, v = cache.update(k, v)

        # GQA：将 KV 头复制到与 Q 头同数（训练/短序列下语义最直观）
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 仅当 T > 1 且处于 prefill / 训练时使用因果遮罩；decode 单步 T=1 时无需掩码（可看全部历史）
        is_causal = cache is None or (T > 1 and k.size(2) == T)
        if cache is not None and T > 1 and k.size(2) != T:
            # 若带历史缓存同时送入多 token，显式构造下三角偏移掩码
            S = k.size(2)
            past = S - T
            mask = torch.ones((T, S), dtype=torch.bool, device=x.device).tril(diagonal=past)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ffn_type = cfg.ffn_type
        hidden = int(cfg.dim * cfg.ffn_mult)
        # 对齐到 64 的倍数对硬件更友好
        hidden = max(64, ((hidden + 63) // 64) * 64)
        if self.ffn_type == "swiglu":
            self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
            self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
            self.w_down = nn.Linear(hidden, cfg.dim, bias=False)
        else:
            self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
            self.w_down = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.w_down(F.relu(self.w_up(x)).square())


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

        # 残差分支输出投影缩放到 1/sqrt(2*L)，保证深层初始方差稳定
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"序列总长 {start_pos + T} 超出 max_seq_len={self.cfg.max_seq_len}")

        x = self.embed(idx)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.cfg.n_layers)]

    def count_parameters(self) -> int:
        # 去重（应对 weight tying）
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total
