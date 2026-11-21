"""防 Softmax 塌缩的稳定交叉熵损失（arXiv 2610.31642 v3）。

原理（麻雀虽小五脏俱全版）：
标准 F.cross_entropy 的梯度是 softmax(z) - onehot。
当模型对某词非常有把握时，间隔 m = z_t - LSE_{j!=t}(z_j) 超过精度临界值（FP32 ~16.6，BF16 ~5.5），
正确项 softmax 舍入为 1.0，梯度变成 0；但错误项仍保留极小正梯度，导致梯度之和不再为 0，
经过 Adam 的自适应二阶矩放大后容易引发 loss spike。
改写为数学等价的：
    m = -LSE_{j!=t}(z_j - z_t),   loss = softplus(-m)
其对 z_t 的导数为 -sigmoid(-m)，在 m 很大时依然精确非零且全词表梯度之和恒为 0。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FP32_CLIFF = 16.6


def _margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    t = targets.unsqueeze(-1)
    diff = logits - logits.gather(-1, t)
    others = diff.scatter(-1, t, float("-inf")).clamp(min=-1e4)
    return -torch.logsumexp(others, dim=-1)


def token_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """每个位置的 log p(target)，形状同 targets，饱和时梯度依然精确。"""
    return -F.softplus(-_margin(logits, targets))


def stable_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if not valid.any():
        return logits.new_zeros((), dtype=torch.float32)
    loss = -token_log_probs(logits, targets.clamp(min=0)) * valid
    total = loss.sum()
    return total / valid.sum() if reduction == "mean" else total


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk: int = 2048,
) -> torch.Tensor:
    """分块计算 Linear + 稳定交叉熵，避免整张 (B*T, V) logits 撑爆显存。"""
    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    valid_count = (targets != ignore_index).sum()
    if valid_count == 0:
        return hidden.new_zeros((), dtype=torch.float32)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        logits = F.linear(hidden[start : start + chunk], weight)
        total = total + stable_cross_entropy(logits, targets[start : start + chunk], ignore_index=ignore_index, reduction="sum")
    return total / valid_count


@torch.no_grad()
def margin_saturation(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    cliff: float = FP32_CLIFF,
) -> float:
    """统计领先间隔超过精度临界值的 token 占比（用于监控 softmax 塌缩风险）。"""
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if not valid.any():
        return 0.0
    m = _margin(logits[valid], targets[valid])
    return float((m > cliff).float().mean())
