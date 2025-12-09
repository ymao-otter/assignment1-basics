import torch
import torch.nn as nn


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad
                state["exp_avg_sq"] = (
                    beta2 * state["exp_avg_sq"] + (1 - beta2) * grad**2
                )
                lr_t = (
                    lr
                    * (1 - beta2 ** state["step"]) ** 0.5
                    / (1 - beta1 ** state["step"])
                )
                p.data = p.data - lr_t * state["exp_avg"] / (
                    state["exp_avg_sq"].sqrt() + eps
                )
                p.data = p.data * (1 - lr * weight_decay)

        return loss
