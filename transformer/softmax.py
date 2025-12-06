import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
