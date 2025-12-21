import torch
from collections.abc import Iterable
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax implementation.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax over

    Returns:
        Softmax of input tensor along specified dimension
    """
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute average cross-entropy loss across examples.

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size)
        targets: Target class indices of shape (batch_size,)

    Returns:
        Average cross-entropy loss (scalar tensor)
    """
    batch_idx = torch.arange(inputs.shape[0])
    log_probs = torch.log_softmax(inputs, dim=-1)
    nll = log_probs[batch_idx, targets]  # negative log likelihood
    return -nll.mean()


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """
    Clip gradients to have maximum L2 norm.

    Args:
        parameters: Collection of trainable parameters
        max_l2_norm: Maximum L2 norm for gradients

    The gradients are modified in-place.
    """
    # Collect all gradients
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    # Compute total L2 norm across all gradients
    total_norm_squared = sum([(g**2).sum() for g in grads], start=torch.tensor(0.0))
    total_norm = torch.sqrt(total_norm_squared)

    # Scale gradients if norm exceeds threshold
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)
