# Implementation provenance

The release is a fixed-configuration extraction of the `MedSTP-v1` branch in
`qwen_vqa_pruning_multichoice_compare_pmc.py`, verified against Delta on
2026-10-18. The source SHA-256 is:

```text
bfc100b5d53e7bfe6edb998618dd365259f1b09cfad0454f6b76546ff4e2a9fa
```

The retained branch consists of:

- `deterministic_token_side_pca`: token-side sigmoid preprocessing, local
  seed 0, and `torch.pca_lowrank(..., center=True)`.
- `_pca_residual_greedy_order_legacy`: weighted residual-energy pivoting,
  float32 epsilon stopping, and original loading-score fallback.
- `prunesid_indices` with `dedup_mode="pca_residual_qr"`, rank ratio .25,
  floor rounding, singular-value power 1, loading weight 0, and count budgets.
- `sobel_boundary_scores`: depthwise Sobel on visual embeddings, replicate
  padding, channel-aggregated magnitude, and min-max normalization with 1e-6.
- The original group caps, cumulative rounded allocation, exact-K completion,
  and restoration of original token order.

Unselected experimental branches, baselines, global environment switches,
logging, dataset loaders, model patches, metric computation, and cluster paths
were removed. The QR/count branch uses the original recurrence, not a new
optimizer or similarity-threshold rule. The grouped-GPU speed variant is not
part of this minimal release.

The public wrapper adds input validation and real-grid resolution. For a real
model grid it preserves the merged-grid-first/direct-grid resolver convention.
Unknown/mismatched grids are rejected instead of using the experiment runner's
guessed-grid fallback. Zero/full budgets return the corresponding original
token slice directly. Neither validation nor these early returns changes the
selection for supported inputs with valid grid metadata.

The chosen default beta is .1 and keep ratio is .1. This release intentionally
does not consolidate the historically different dataset-tuned configurations
into a claim of one universal paper-reproduction setting. No paper results or
datasets are distributed here.

## Extraction validation

The extracted selector was checked against the source functions on CPU in
942 synthetic cases spanning token counts, grids, keep ratios, beta values,
dtypes, and degenerate/repeated embeddings. Retained indices and gathered
embeddings matched exactly; Sobel scores and PCA intermediate tensors were
also checked. Validation used Python 3.12 and PyTorch 2.14.1. The included
unit suite also checks budget, ordering, grid handling, input/RNG preservation,
and the absence of environment-variable overrides. CUDA execution has a
conditional unit test but was not executed in this local validation.
