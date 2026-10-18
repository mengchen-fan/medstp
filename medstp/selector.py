"""Small public API around the unchanged fixed v1 selection rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .subspace import subspace_indices
from .transition import sobel_boundary_scores


@dataclass(frozen=True)
class MedSTPOutput:
    indices: torch.Tensor
    pruned_tokens: torch.Tensor


def resolve_grid(
    n_tokens: int,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    grid_hw: tuple[int, int] | None,
) -> tuple[int, int]:
    if grid_hw is not None and image_grid_thw is not None:
        raise ValueError("Pass image_grid_thw or grid_hw, not both.")
    if grid_hw is not None:
        if len(grid_hw) != 2 or any(int(v) != v or v <= 0 for v in grid_hw):
            raise ValueError("grid_hw must contain two positive integers.")
        h, w = (int(v) for v in grid_hw)
    else:
        if image_grid_thw is None or image_grid_thw.numel() != 3:
            raise ValueError("Provide one image's real image_grid_thw, or its known merged grid_hw.")
        raw = image_grid_thw.detach().reshape(-1).cpu().tolist()
        if any(int(v) != v or v <= 0 for v in raw):
            raise ValueError("image_grid_thw must contain three positive integers.")
        t, h, w = (int(v) for v in raw)
        if t != 1:
            raise ValueError("This selector accepts one 2D image, not a temporal/video grid.")
        if int(spatial_merge_size) != spatial_merge_size or spatial_merge_size < 1:
            raise ValueError("spatial_merge_size must be a positive integer.")
        merge = int(spatial_merge_size)
        # Match the verified resolver's merged-grid-first/direct-grid fallback.
        if (h // merge) * (w // merge) == n_tokens:
            h, w = h // merge, w // merge
        elif h * w != n_tokens:
            raise ValueError("image_grid_thw does not match the number of visual tokens.")
    if h * w != n_tokens:
        raise ValueError("The merged grid height times width must equal N.")
    return h, w


@torch.no_grad()
def select_medstp(
    visual_tokens: torch.Tensor,
    keep_ratio: float = 0.10,
    *,
    beta: float = 0.10,
    image_grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
    grid_hw: tuple[int, int] | None = None,
) -> MedSTPOutput:
    """Select K original visual tokens using one fixed MedSTP-v1 configuration.

    ``visual_tokens`` contains one image's post-merger patch embeddings [N, D].
    Supply real processor grid metadata or an explicit known merged ``grid_hw``.
    No model, data, labels, text embeddings, or trained selector is needed.
    """
    if visual_tokens.ndim != 2 or visual_tokens.shape[1] == 0:
        raise ValueError("visual_tokens must have shape [N, D] with D > 0.")
    if not visual_tokens.is_floating_point():
        raise TypeError("visual_tokens must be floating point.")
    if visual_tokens.device.type not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or CUDA; the original PCA backend is not configured for MPS.")
    if not math.isfinite(keep_ratio) or not 0 <= keep_ratio <= 1:
        raise ValueError("keep_ratio must be finite and in [0, 1].")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    n_tokens = int(visual_tokens.shape[0])
    k = int(keep_ratio * n_tokens)
    if k == 0 or k == n_tokens:
        indices = torch.arange(k, device=visual_tokens.device, dtype=torch.long)
        return MedSTPOutput(indices, visual_tokens[indices])
    grid = resolve_grid(n_tokens, image_grid_thw, spatial_merge_size, grid_hw)
    evidence = sobel_boundary_scores(visual_tokens, grid)
    indices = subspace_indices(visual_tokens, k, evidence, beta)
    return MedSTPOutput(indices, visual_tokens[indices])
