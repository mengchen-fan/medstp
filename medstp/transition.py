"""Original token-grid transition scores; no raw-image processing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_SOBEL_KERNEL_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    score_min = scores.min()
    score_max = scores.max()
    return (scores - score_min) / (score_max - score_min + 1e-6)


def boundary_scores(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Original adjacent-difference fallback for grids smaller than 3 x 3."""
    h, w = grid_hw
    feat = tokens.float().reshape(h, w, -1)
    scores = torch.zeros(h, w, device=tokens.device, dtype=torch.float32)
    if w > 1:
        dx = (feat[:, 1:] - feat[:, :-1]).norm(dim=-1)
        scores[:, 1:] += dx
        scores[:, :-1] += dx
    if h > 1:
        dy = (feat[1:, :] - feat[:-1, :]).norm(dim=-1)
        scores[1:, :] += dy
        scores[:-1, :] += dy
    return normalize_scores(scores.flatten())


def sobel_boundary_scores(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    n_tokens = tokens.shape[0]
    if n_tokens == 0:
        return torch.empty(0, device=tokens.device)
    h, w = grid_hw
    if h < 3 or w < 3:
        return boundary_scores(tokens, grid_hw)

    feat = tokens.float().reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
    channels = feat.shape[1]
    cache_key = (tokens.device.type, tokens.device.index, feat.dtype, int(channels))
    cached_kernels = _SOBEL_KERNEL_CACHE.get(cache_key)
    if cached_kernels is None:
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=tokens.device,
            dtype=feat.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=tokens.device,
            dtype=feat.dtype,
        ).view(1, 1, 3, 3)
        cached_kernels = (
            kx.repeat(channels, 1, 1, 1).contiguous(),
            ky.repeat(channels, 1, 1, 1).contiguous(),
        )
        _SOBEL_KERNEL_CACHE[cache_key] = cached_kernels
    padded = F.pad(feat, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, cached_kernels[0], groups=channels)
    gy = F.conv2d(padded, cached_kernels[1], groups=channels)
    edge = torch.sqrt(gx.squeeze(0).square().sum(dim=0) + gy.squeeze(0).square().sum(dim=0) + 1e-6)
    return normalize_scores(edge.flatten())
