# GNN-SKAN source review and resulting improvements

Reviewed on 27 October 2026 at upstream commit
`5a352a170dadb8a0fa3f66eb92b6be0975ef15df` of
[Lirain21/GNN-SKAN](https://github.com/Lirain21/GNN-SKAN/tree/5a352a170dadb8a0fa3f66eb92b6be0975ef15df).
The repository was downloaded to `/tmp/gnn-skan-reference` for reading. Its Python
code and training scripts were not imported or executed.

## Source and reuse boundary

Inspected the README, `kan/SKAN_layer.py` and `KanConv_layer.py`, including the
RBF primitive, SKAN feature layer, convolution placement and classifier setup.
No LICENSE/COPYING/NOTICE file was found in the inspected tracked tree. No upstream
source files were copied, vendored or introduced as runtime dependencies. The
changes here are independently written extensions of our existing coefficient
expansion and Gaussian basis, guided by the architecture and mathematical
behavior. The upstream project remains the credited research reference.

Stable source links:

- [RBF and SKAN layers](https://github.com/Lirain21/GNN-SKAN/blob/5a352a170dadb8a0fa3f66eb92b6be0975ef15df/kan/SKAN_layer.py)
- [Graph layers and prediction heads](https://github.com/Lirain21/GNN-SKAN/blob/5a352a170dadb8a0fa3f66eb92b6be0975ef15df/KanConv_layer.py)

## Findings and selected improvements

The reference separates normalized RBF features from a raw-input SiLU base
branch. Its center grid and bandwidth have separate optimization flags. The
reference Gaussian uses a denominator d in `exp(-((x-center)/d)^2)`, initially
based on grid spacing. Our Gaussian convention uses standard deviation sigma in
`exp(-0.5*((x-center)/sigma)^2)`, so matching kernels requires `sigma=d/sqrt(2)`.
This conversion is tested explicitly rather than treating the conventions as
interchangeable.

Implemented:

1. `TransformConfig.basis_normalization`: `none` (default) or trainable `layernorm`.
   Only the KAN basis input is normalized. The base branch still consumes raw x.
2. `coefficient_init`: `normal` (default) or `truncated_normal`, with optional
   finite positive `coefficient_init_std`. The default fan/basis-scaled normal
   initialization is unchanged. Truncation uses PyTorch's absolute [-2,2] bounds.
3. RBF `trainable_centers` and `trainable_widths` flags; both remain True by default.
   Frozen values stay in state dictionaries but receive no gradients.
4. Optional initial Gaussian `width` and `shared_width`. Widths are softplus
   constrained plus epsilon, avoiding zero or negative bandwidths during training.
   The original per-basis raw-zero width initialization is unchanged by default.
5. Exported `skan_transform(...)` preset: normalized RBF branch, shared positive
   width initialized from grid spacing, std-0.1 truncated coefficients and optional
   raw SiLU base branch. It returns an ordinary serializable TransformConfig.
6. Added an example using identity GCN messages, post-aggregation SKAN-style
   updates and an optional SKAN-style classifier. Classifier grid settings can
   differ from encoder settings through separate configs.

`skan_transform` defaults its optimization flags to True, matching this
framework's existing adaptive-RBF behavior. The upstream feature-layer constructor
and experiment flags have their own defaults; users must set matching options
explicitly for comparisons. With base_residual=False, the preset also disables
output bias, matching a pure basis contraction.

## Deliberate differences and deferred work

The reference's GCN applies its SKAN after neighborhood aggregation. Our update
stage already supports this: select an identity message transform and a KAN
update. The generic graph layer retains its own learnable aggregate bias, so
this configuration is not an identical upstream layer parameterization.

We retain the tested PyG target-grouped attention normalization and incoming-degree
GCN convention. The inspected upstream code groups GAT logits by the source index
row and computes GCN degrees from that row. We did not copy these choices into
our general directed-graph API; numerical parity with standard PyG operators
remains the baseline contract. Likewise, source-specific categorical molecular
embeddings are not substituted for the framework's dense-feature edge API.

The upstream empirical grid-update method was reviewed but not adopted. Changing
centers during training also changes the represented function; coefficient
transfer and optimizer-state policies need their own design. A docstring and the
interpolation expression in the inspected code also assign opposite meanings to
the grid interpolation weight. We avoid exposing that ambiguity here.

Not implemented: exact full-model reproduction, dataset/scaffold pipelines,
GIN, jumping-knowledge combinations, advanced pooling, pruning, masked molecular
multilabel losses, grid refinement or source checkpoint imports. LayerNorm is
optional for every basis family, including extensions, but it is not necessarily
a good fit for every stage. In particular, scalar input normalization removes
scalar variation from the basis branch. No accuracy or speed improvement is
claimed from the toy checks.

## Compatibility and tests

Old configs omit the new fields and receive their original defaults. With
normalization disabled, the Identity module adds no state keys; original RBF
parameter names and shapes are unchanged. Old config/state strict-load behavior
is explicitly tested. New options, learned norm weights, width shape and frozen
parameter flags survive JSON configuration and model checkpoint reconstruction.

Added tests for every freezing/shared-width combination, numerical derivatives,
initial-width values and positivity under extreme raw parameters, manual
normalized-RBF equations, raw base-input preservation, initialization scale,
invalid options, four-backbone gradients/checkpoints, old config/state loading,
and exact repeated CPU training. Existing numerical PyG parity, transform package,
and executable documentation suites are rerun as regression checks.

The user guide now includes an executable SKAN-style example and the new config
fields. CI runs the new training example. Existing implementation/future-work
documents describe their earlier state; this review records the newer optional
feature-layer improvements, while exact paper architectures remain future work.

### Verification results

- Combined framework/transform suites: **214 passed, 3 CUDA-only checks skipped**.
- Combined source line coverage: **95.34%**, above the 90% gate.
- All **17 user-guide Python examples** executed as part of the suite.
- `ruff check KAN_framwork mathematical_transforms`: passed.
- SKAN-style graph example, seed 7, 30 epochs: toy objective decreased from
  approximately `0.709291` to `0.000600`; training-set evaluation loss approximately
  `0.000488`. This demonstrates execution, not generalization performance.
- Existing checkpoint/config tests and PyG operator parity checks passed.
- CUDA and the remote GitHub Actions matrix were not executed on this CPU host.
