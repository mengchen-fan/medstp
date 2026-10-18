"""Synthetic smoke test: no dataset, model download, or output files."""

from __future__ import annotations

import argparse

import torch

from .selector import select_medstp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu or install a CUDA PyTorch build.")
    torch.manual_seed(7)
    tokens = torch.randn(576, 128, device=device)
    result = select_medstp(tokens, image_grid_thw=torch.tensor([[1, 48, 48]]))
    assert len(result.indices) == int(0.1 * len(tokens))
    assert torch.equal(result.pruned_tokens, tokens[result.indices])
    print(f"MedSTP smoke test passed on {device}: {len(tokens)} -> {len(result.indices)} tokens.")


if __name__ == "__main__":
    main()
