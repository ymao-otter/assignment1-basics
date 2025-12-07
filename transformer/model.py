import torch
import torch.nn as nn
import einops
from jaxtyping import Bool, Float
from torch import Tensor
from transformer.nn_utils import softmax


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        variance = 2 / (self.in_features + self.out_features)
        std = variance**0.5
        a = -3 * std
        b = 3 * std
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


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


class RoPE(nn.Module):
    blocks: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer(
            "blocks", torch.empty(max_seq_len, d_k // 2, 2, 2, device=device)
        )
        self.reset_blocks()

    def reset_blocks(self):
        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                l_theta = i / self.theta ** (2 * k / self.d_k)
                l_theta_tensor = torch.tensor(l_theta, device=self.device)
                self.blocks[i, k, 0, 0] = torch.cos(l_theta_tensor)
                self.blocks[i, k, 0, 1] = -torch.sin(l_theta_tensor)
                self.blocks[i, k, 1, 0] = torch.sin(l_theta_tensor)
                self.blocks[i, k, 1, 1] = torch.cos(l_theta_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        *leading, d = x.shape
        x_blocks = x.view(*leading, d // 2, 2)
        val = einops.einsum(x_blocks, self.blocks[:seq_len], "... l, ... j l -> ... j")
        return val.reshape(x.shape)


class FFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    SiLU (Swish) activation function.

    Args:
        in_features: Input features

    Returns:
        SiLU applied to input
    """
    return in_features * torch.sigmoid(in_features)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Scaled dot product attention.

    Args:
        Q: Query tensor
        K: Key tensor
        V: Values tensor
        mask: Optional mask tensor

    Returns:
        Output of scaled dot product attention
    """
    qk = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        inverse_mask = ~mask
        qk = qk.masked_fill(inverse_mask, float("-inf"))
    softmax_qk = softmax(qk / (K.shape[-1] ** 0.5))
    return einops.einsum(
        softmax_qk, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )
