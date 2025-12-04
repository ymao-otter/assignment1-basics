import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, torch.float32)
        result = (
            x
            * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
            * self.gain
        )
        return result.to(self.dtype)
