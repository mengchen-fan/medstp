# MedSTP

**MedSTP** is a training-free visual-token pruning method for medical vision-language models. It selects a compact subset of the original visual embeddings before they reach the language model, using two complementary signals:

- **Subspace selection:** PCA coordinates and iterative residual-gain selection retain complementary directions in the visual representation.
- **Transition evidence:** Sobel-based gradients over the spatial token grid highlight tokens at structural transitions.

MedSTP needs no fine-tuning and no additional trained network. The selected tokens are the original embeddings, gathered unchanged and kept in their original order.

> This repository provides the standalone selector. It is not a turnkey inference or evaluation framework, and it does not include datasets, model weights, or experimental results.

## Requirements

- Python 3.10+
- PyTorch 2.2+

## Installation

```bash
git clone https://github.com/mengchen-fan/medstp.git
cd medstp
pip install -e .
```

## Quick Start

Run the synthetic smoke test. It downloads no data or models and writes no files.

```bash
python -m medstp                 # automatically selects CPU or CUDA
python -m medstp --device cuda   # explicitly use a GPU
```

## Usage

```python
from medstp import select_medstp

selection = select_medstp(
    visual_tokens,
    image_grid_thw=image_grid_thw,
    spatial_merge_size=spatial_merge_size,
)

kept_indices = selection.indices        # unique indices, in original order
kept_tokens = selection.pruned_tokens   # unchanged embeddings at those indices
```

The inputs are:

- **`visual_tokens`** is a floating-point tensor of shape `[N, D]` for a single image. It holds the visual embeddings *after* the vision-language merger, as they would be passed to the language model.
- **`image_grid_thw`** is the image's grid metadata as returned by the model's processor. Pass the real values; MedSTP does not guess a square grid.
- **`spatial_merge_size`** is the merge factor from the backbone's `vision_config`.

If you already know the merged spatial grid, you can pass it directly as `grid_hw` instead of `image_grid_thw`.

Inputs may be on CPU or CUDA.

## Integration Notes

- **Visual tokens only.** Pass only the image's visual embeddings. Exclude text tokens and special delimiters such as vision start/end markers.
- **Keep alignment.** Apply the same `selection.indices` to the visual position IDs and to any other token-aligned features or masks. This keeps the retained tokens at their original positions.
- **One image at a time.** For inputs with multiple images, run the selector separately on each image's tokens and its own grid metadata.

## Repository Structure

```text
medstp/
├── selector.py     # public API and grid validation
├── subspace.py     # PCA-based subspace selection
└── transition.py   # Sobel-based transition evidence
tests/
└── test_medstp.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```
