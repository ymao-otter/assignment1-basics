import torch.nn as nn
import torch
import einops


class RoPE(nn.Module):
    blocks: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer(
            "blocks", torch.empty(max_seq_len, d_k, 2, 2, device=device)
        )
        self.reset_blocks()

    def reset_blocks(self):
        for i in range(self.max_seq_len):
            for k in range(self.d_k):
                l_theta = i / self.theta ** (2 * k / self.d_k)
                self.blocks[i, k, 0, 0] = torch.cos(l_theta)
                self.blocks[i, k, 0, 1] = -torch.sin(l_theta)
                self.blocks[i, k, 1, 0] = torch.sin(l_theta)
                self.blocks[i, k, 1, 1] = torch.cos(l_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        *leading, d = x.shape
        x_blocks = x.view(*leading, d // 2, 2)
        val = einops.einsum(x_blocks, self.blocks[:seq_len], "... l, ... j l -> ... j")
        return val.view(x.shape)
