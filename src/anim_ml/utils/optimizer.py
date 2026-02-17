from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class DmlAdamW(Optimizer):
    """AdamW using mul_/add_ instead of lerp_ for DirectML compatibility."""

    def __init__(
        self,
        params: object,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)  # type: ignore[arg-type]

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: object = None) -> torch.Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()  # type: ignore[operator]

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.add_(exp_avg / denom, alpha=-step_size)

        return loss  # type: ignore[return-value]
