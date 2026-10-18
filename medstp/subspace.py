"""Fixed MedSTP-v1 PCA, residual ordering, and count-budget selection."""

from __future__ import annotations

import math

import torch

from .transition import normalize_scores


def pca_coordinates(tokens: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the original token-side centering, rank rule, and local PCA RNG."""
    devices: list[int] = []
    if tokens.is_cuda:
        index = tokens.device.index
        devices = [torch.cuda.current_device() if index is None else int(index)]
    q = max(int(k * 0.25), 4)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(0)
        standard = torch.sigmoid(tokens.float()).permute(1, 0)
        _, singular_values, loadings = torch.pca_lowrank(standard, q=q, center=True)
    sv_scale = singular_values.float().clamp_min(torch.finfo(torch.float32).eps).pow(1.0)
    return loadings, loadings.float() * sv_scale.unsqueeze(0)


def residual_order(
    coordinates: torch.Tensor,
    fallback_scores: torch.Tensor,
    residual_weights: torch.Tensor | None,
) -> list[int]:
    """Original v1 Gram-Schmidt pivot, stopping criterion, and fallback order."""
    n_rows = int(coordinates.shape[0])
    if n_rows == 0:
        return []
    residual = torch.nan_to_num(coordinates.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
    fallback = torch.nan_to_num(fallback_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if residual_weights is None:
        weights = torch.ones(n_rows, device=coordinates.device, dtype=residual.dtype)
    else:
        weights = torch.nan_to_num(
            residual_weights.to(device=coordinates.device, dtype=residual.dtype),
            nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(0.0)
    # Fixed original v1 loading_weight=0.0; retain its arithmetic unchanged.
    loading_factors = 1.0 + 0.0 * normalize_scores(fallback).to(
        device=coordinates.device, dtype=residual.dtype,
    )
    available = torch.ones(n_rows, device=coordinates.device, dtype=torch.bool)
    order: list[int] = []
    eps = torch.finfo(residual.dtype).eps
    for _ in range(n_rows):
        raw_residual_energy = residual.square().sum(dim=1)
        weighted_residual_gain = weights * loading_factors * raw_residual_energy
        weighted_residual_gain[~available] = -float("inf")
        best_gain, best_idx = torch.max(weighted_residual_gain, dim=0)
        best_raw_energy = raw_residual_energy[best_idx]
        if (
            not bool(torch.isfinite(best_gain).item())
            or float(best_gain.item()) <= float(eps)
            or float(best_raw_energy.item()) <= float(eps)
        ):
            remaining = torch.nonzero(available, as_tuple=False).flatten()
            if remaining.numel() > 0:
                fallback_order = remaining[torch.argsort(fallback[remaining], descending=True)]
                order.extend(int(idx.item()) for idx in fallback_order)
            break
        idx = int(best_idx.item())
        order.append(idx)
        available[idx] = False
        direction = residual[idx] / best_raw_energy.sqrt().clamp_min(eps)
        residual = residual - torch.outer(residual @ direction, direction)
        residual[idx] = 0
    return order


def subspace_indices(tokens: torch.Tensor, k: int, evidence: torch.Tensor, beta: float) -> torch.Tensor:
    """Specialize the original pca_residual_qr/count branch without NMS."""
    n_tokens = tokens.shape[0]
    if k <= 0:
        return torch.empty(0, device=tokens.device, dtype=torch.long)
    if n_tokens <= k:
        return torch.arange(n_tokens, device=tokens.device, dtype=torch.long)
    if n_tokens <= 16:
        return torch.topk(tokens.float().norm(dim=-1), k=k, largest=True).indices.sort().values

    signed_loadings, coordinates = pca_coordinates(tokens, k)
    projector_lengths = torch.abs(signed_loadings)
    belong = torch.argmax(projector_lengths, dim=1)
    residual_weights = None
    if beta != 0.0:
        evidence = normalize_scores(evidence.to(device=tokens.device, dtype=torch.float32))
        residual_weights = 1.0 + float(beta) * evidence
        projector_lengths = projector_lengths * residual_weights.unsqueeze(1)
    projector_scores = projector_lengths.clone()
    groups = projector_lengths.shape[1]
    projector_mask = belong.unsqueeze(1).repeat(1, groups)
    index_map = torch.arange(groups, device=tokens.device).unsqueeze(0).repeat(n_tokens, 1)
    mask = projector_mask != index_map
    projector_lengths[mask] = 0
    projector_scores[mask] = 0

    group_idxs: list[torch.Tensor] = []
    given_scores = torch.arange(n_tokens, 0, -1, device=tokens.device, dtype=projector_lengths.dtype)
    for g in range(groups):
        group_indices = torch.nonzero(belong == g, as_tuple=False).flatten()
        if group_indices.numel() == 0:
            group_idxs.append(group_indices)
            continue
        g_scores = projector_lengths[:, g][group_indices]
        keep_local = residual_order(
            coordinates[group_indices], g_scores,
            residual_weights[group_indices] if residual_weights is not None else None,
        )
        keep_indices = (
            group_indices[torch.tensor(keep_local, device=tokens.device, dtype=torch.long)]
            if keep_local else group_indices[:0]
        )
        if keep_indices.numel() > 0:
            projector_scores[keep_indices, g] = given_scores[:keep_indices.numel()]
        group_idxs.append(keep_indices)

    keep_counts = torch.tensor([int(v.numel()) for v in group_idxs], device=tokens.device, dtype=torch.float32)
    group_counts = (projector_mask == index_map).sum(dim=0).float()
    lower = torch.minimum(torch.ones_like(group_counts), group_counts)
    upper = torch.ones_like(group_counts) * 5 * math.ceil(k / 64)
    upper = torch.minimum(torch.minimum(upper, group_counts), keep_counts)
    while float(upper.sum().item()) < k and float(upper.sum().item()) < n_tokens:
        upper = torch.minimum(upper + 1, group_counts)
    other = max(0, int(k - lower.sum().item()))
    if float(keep_counts.sum().item()) <= 0:
        keep_counts = group_counts
    budget_weights = keep_counts
    norm_counts = budget_weights / budget_weights.sum().clamp_min(1.0)
    cumulative = torch.cumsum(norm_counts, dim=0)
    other_d = (cumulative * other).round().int()
    other_d = other_d - torch.cat([torch.zeros(1, device=tokens.device, dtype=torch.int32), other_d[:-1]])
    group_budget = torch.minimum(other_d.float() + lower, upper).int()
    order = torch.argsort(budget_weights, descending=True)
    target = int(other + int(lower.sum().item()))
    fill = 0
    while int(group_budget.sum().item()) < target and len(order) > 0:
        g = int(order[fill % len(order)].item())
        room = int(upper[g].item()) - int(group_budget[g].item())
        need = target - int(group_budget.sum().item())
        if room > 0:
            group_budget[g] += min(room, need)
        fill += 1
        if fill > len(order) * max(2, n_tokens):
            break

    sorted_by_group = torch.argsort(projector_scores, descending=True, dim=0)
    parts: list[torch.Tensor] = []
    for g in range(groups):
        budget = min(int(group_budget[g].item()), n_tokens)
        if budget > 0:
            parts.append(sorted_by_group[:, g][:budget])
    if not parts:
        return torch.topk(tokens.float().norm(dim=-1), k=k, largest=True).indices.sort().values
    selected = torch.cat(parts, dim=0)[:k]
    if selected.numel() > 150:
        selected = selected.sort().values
    selected = torch.unique(selected, sorted=True)
    if selected.numel() < k:
        remaining_mask = torch.zeros(n_tokens, device=tokens.device, dtype=torch.bool)
        remaining_mask[selected] = True
        remaining = torch.nonzero(~remaining_mask, as_tuple=False).flatten()
        fill_scores = tokens.float().norm(dim=-1)[remaining]
        selected = torch.cat([selected, remaining[torch.topk(fill_scores, k=k-selected.numel(), largest=True).indices]])
    elif selected.numel() > k:
        score = tokens.float().norm(dim=-1)[selected]
        selected = selected[torch.topk(score, k=k, largest=True).indices]
    return selected.sort().values
