"""极简 Muon 优化器（Momentum Orthogonalized by Newton-Schulz）。

面试核心要点：
- 为什么只给 >=2D 矩阵用 Muon，而 Embedding/Head/Norm 仍用 AdamW？
  因为 2D 线性层权重梯度的奇异值往往极度不均（少数大奇异值方向主导更新）。
  Muon 对动量矩阵做 Newton-Schulz 迭代近似正交化，把所有非零奇异值拉到 1 附近，
  相当于在谱范数约束下走最速下降；而 Embedding/Norm 是一维向量或查表，不具备这种矩阵结构。
- Newton-Schulz 五次迭代系数 (3.4445, -4.7750, 2.0315) 无需 SVD 即可在 GPU/MPS 上高效运行。
"""
from __future__ import annotations

import torch


def newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + 1e-7)
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.t()
    for _ in range(steps):
        s = x @ x.t()
        x = a * x + (b * s + c * (s @ s)) @ x
    return (x.t() if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon 仅适用于二维矩阵参数，收到 shape={tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.lerp_(g, 1.0 - mu)
                update = g.lerp(buf, mu) if group["nesterov"] else buf
                orth = newton_schulz_orthogonalize(update, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(orth, alpha=-lr * scale)
        return loss
