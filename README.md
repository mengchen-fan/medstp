# MedSTP

A minimal, training-free visual-token selector for frozen medical VQA models.
The implementation keeps the verified MedSTP-v1 selection rule: token-side PCA,
component grouping, Sobel-weighted residual-QR ordering, count-based group budgets,
and exact-K completion. It does **not** use similarity NMS.

This repository contains only the method, a synthetic smoke test, and unit tests.
No datasets, images, model weights, predictions, experiment results, cluster
paths, or evaluation pipelines are included.

## Install

Requires Python 3.10+ and PyTorch 2.2+. Install the appropriate PyTorch build for
your CPU/CUDA environment, then install this package:

```bash
pip install -e .
python -m medstp
python -m unittest discover -s tests -v
```

The smoke test uses generated token embeddings; it downloads nothing and writes
no files. To run it on a GPU:

```bash
python -m medstp --device cuda
```

## Use

```python
import torch
from medstp import select_medstp

# Replace these synthetic tensors with one image's post-merger visual embeddings
# and its processor-provided image_grid_thw.
tokens = torch.randn(576, 128)              # [N, D]
image_grid_thw = torch.tensor([[1, 48, 48]])

selection = select_medstp(
    tokens,
    image_grid_thw=image_grid_thw,
    spatial_merge_size=2,
)
selected_indices = selection.indices
selected_tokens = selection.pruned_tokens
```

The grid above is a synthetic example, not an inferred grid. The merged grid is
24 x 24, matching the 576 visual embeddings. Pass the real processor metadata
and the model's `vision_config.spatial_merge_size` in an actual VLM integration.
If the actual post-merger grid is already known, pass `grid_hw=(height, width)`
instead. Missing or inconsistent grids raise an error; the package does not
silently reshape tokens into a guessed square.

`select_medstp` accepts a single image's `[N, D]` floating-point tensor, on CPU
or CUDA. Returned indices are unique, sorted in original sequence order, and on
the input device. Returned embeddings are gathered unchanged from the input,
preserving its dtype. The retained count is `K = floor(keep_ratio * N)`.

### One fixed default configuration

| Setting | Default |
|---|---|
| Keep ratio | 0.10 |
| Sobel multiplier beta | 0.10 |
| PCA components | `max(floor(K / 4), 4)` |
| PCA random seed | 0, inside a local RNG context |
| PCA preprocessing | `sigmoid(tokens.float()).T`, `center=True` |
| PCA coordinates | signed loadings multiplied by singular values |
| Group assignment | maximum absolute PCA loading |
| Within-group ordering | Sobel-weighted residual-QR, then loading fallback |
| Group allocation | original count-based lower/upper-cap allocation |
| Sobel padding | replicate |

The public API allows changing the token budget and `beta`, but there are no
dataset-specific presets, v1/v2 switches, search scripts, or hidden environment
variable overrides. This is one fixed release configuration, **not a claim that
one configuration reproduces every historical paper-table row**.

As in the source implementation, PCA requires `max(floor(K / 4), 4)` not to
exceed `min(N, D)` when `N > 16`. The rank is not silently clamped. Repeated
calls are deterministic within the same backend; different PyTorch versions,
devices, and numerical tie behavior can change selected indices.

### VLM integration boundary

Apply the selector after the vision-language merger and before LLM insertion.
Pass only image patch embeddings: exclude image delimiters, text, and other
special tokens. Gather the corresponding original visual-position IDs with
`selected_indices`; do not assign a new spatial grid to the retained sequence.
Backbone-specific aligned features (for example Qwen3-VL DeepStack embeddings)
must be gathered with the same indices, and the sequence mask must be shortened
consistently. For multiple images, call the selector separately for each image.

The package returns a token subset; it does not patch a Transformers model or
implement decoding, dataset loading, answer scoring, or benchmarking.

## Source and attribution

The extraction preserves the verified Delta MedSTP-v1 branch, rather than the
older NMS-based `MedSTP-clean` prototype. PCA grouping and count-based budget
allocation follow the [PruneSID](https://github.com/ZhengyaoFang/PruneSID) lineage;
the within-group selection is residual-QR with the existing Sobel multiplier.
See [SOURCE.md](SOURCE.md) for the exact extraction boundary and source hash.
