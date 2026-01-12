import numpy as np
import math
import random
import torch


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    num_samples = len(x)
    # Fixed: Use np.random.randint instead of creating a huge list
    # This avoids allocating 32 GB of RAM for large datasets!
    starting_indices = np.random.randint(
        0, num_samples - context_length, size=batch_size
    )
    input_batch = torch.empty(batch_size, context_length, device=device)
    target_batch = torch.empty(batch_size, context_length, device=device)
    for i, starting_index in enumerate(starting_indices):
        input_batch[i] = torch.from_numpy(
            x[starting_index : starting_index + context_length]
        )
        target_batch[i] = torch.from_numpy(
            x[starting_index + 1 : starting_index + context_length + 1]
        )
    return input_batch, target_batch
