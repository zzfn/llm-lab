"""手写 LoRA（Low-Rank Adaptation）注入、保存与无损合并。

核心原理（面试高频）：
1. 冻结原线性层权重 W，在旁路增加低秩分解矩阵 B @ A（其中 r << d）。
2. 初始化：A 用高斯 N(0, 0.02^2)，B 全零初始化，确保刚注入时 ΔW = B @ A = 0，完全不破坏原模型能力。
3. 推理零额外开销：训练完成后执行 merge_lora(model)，直接把 (alpha/r) * (B @ A) 加回 W 并还原成普通 nn.Linear。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if base.bias is not None:
            raise ValueError("LoRALinear 暂仅支持无 bias 的 nn.Linear")
        self.base = base
        self.base.weight.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Linear(base.in_features, rank, bias=False, device=dev, dtype=dt)
        self.B = nn.Linear(rank, base.out_features, bias=False, device=dev, dtype=dt)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        delta = (self.B.weight.float() @ self.A.weight.float()) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.base.weight.requires_grad_(True)
        return self.base


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    targets: tuple[str, ...] = ("wq", "wk", "wv", "wo", "w_up", "w_down", "w_gate"),
) -> tuple[int, int]:
    """冻结基座所有参数，仅对目标 Linear 层注入可训练 LoRA 旁路，返回 (可训练参数数, 总参数数)。"""
    for p in model.parameters():
        p.requires_grad_(False)

    for parent in list(model.modules()):
        for attr, child in list(parent.named_children()):
            if attr in targets and isinstance(child, nn.Linear):
                setattr(parent, attr, LoRALinear(child, rank=rank, alpha=alpha))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """将全部 LoRALinear 无损合并回标准 nn.Linear，消除推理额外开销。"""
    for parent in list(model.modules()):
        for attr, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, attr, child.merged_linear())
    return model


def save_lora_only(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items() if ".A.weight" in k or ".B.weight" in k}
    torch.save(state, path)
