import torch


def data_loading(x, batch_size, context_length, device="cpu", offsets=None):
    """
    Sample `batch_size` number of sequences of length `context_length` from 
    1D token tensor `x`. Input and target sequences are returned, where targets are
    the input sequences shifted by one token.

    Uses pure torch ops so the CPU-to-GPU copy can be overlapped (non-blocking).
    """
    if context_length > x.shape[0]:
        raise RuntimeError("Insufficient raw text length for the given context_length.")

    # Reuse offsets if provided; otherwise create them on the same device as `x`.
    if offsets is None or offsets.numel() != context_length or offsets.device != x.device:
        offsets = torch.arange(context_length, device=x.device, dtype=torch.long)

    max_start = x.shape[0] - context_length - 1  # keep targets in-bounds
    if max_start <= 0:
        raise RuntimeError("context_length is too large for the provided data.")

    start_idx = torch.randint(0, max_start + 1, (batch_size,), device=x.device)
    idx = start_idx[:, None] + offsets

    inputs = x[idx]
    targets = x[idx + 1]

    non_blocking = x.device.type == "cpu" and device != "cpu"
    return (
        inputs.to(device=device, non_blocking=non_blocking),
        targets.to(device=device, non_blocking=non_blocking),
    )
