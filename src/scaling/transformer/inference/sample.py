import torch
import torch.nn.functional as F


def sample_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)[:, -1]


def fast_multinomial(p: torch.Tensor) -> torch.Tensor:
    """
    A slightly faster version of torch.multinomial(p, 1)
    See: https://github.com/pytorch/pytorch/issues/30968#issuecomment-859084590
    """
    return (p.cumsum(-1) >= torch.rand(p.shape[:-1]).to(p.device)[..., None]).byte().argmax(-1)


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    scaled_logits = logits / temperature
    token_probs = F.softmax(scaled_logits, dim=-1)
    return fast_multinomial(token_probs)[:, -1]


def top_k_transform(logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    top_k_logits = logits.clone()
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    top_k_logits[indices_to_remove] = -float("Inf")
    return top_k_logits


def top_p_transform(logits: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    # convert to 1D
    top_p_logits = logits.clone()
    sorted_logits, sorted_indices = torch.sort(top_p_logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > threshold
    # Shift the indices to the right to keep also the first token
    # above the threshold
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    top_p_logits[indices_to_remove] = -float("Inf")
    return top_p_logits
