# Future steps

This plan separates paper reproduction, broader functionality and deployment.
The current implementation and its limits are recorded in `IMPLEMENTATION.md`.
None of the items below are claimed to be complete.

## Priority 1: reproducible paper-specific architectures

### 1. Exact residual gated ReLU-KAN recommendation model

Use the supplied Kou/Xu KAN-GNN PDF, especially sections 3.3–3.5.

1. Implement a dedicated `ResGatedReLUKANConv` operator with
   `ReLUKAN(h_j-h_i)` difference messages.
2. Add `sigmoid(W_g [h_i || h_j] + b_g)` vector gates, separate self/residual
   mappings, LayerNorm and paper-consistent dropout ordering.
3. Implement the paper's full decoder module with its gated/self/residual paths;
   the present concatenation KAN head is only a generic learned pair scorer.
4. Construct a mashup–API bipartite training graph with explicit node types,
   reproducible description-feature inputs and candidate-pair constraints.
5. Implement balanced negative sampling that excludes all known positives and
   rejects invalid type combinations. Separate evaluation negatives from training.
6. Implement query-grouped Recall@K, Precision@K, NDCG@K, MAP and AUC with
   well-defined candidate pools and leakage-free edge splits.
7. Recover paper hyperparameters, source availability and dataset licensing;
   report multiple seeds with confidence intervals.

Acceptance: independently coded equation-level gate/update tests, no held-out
edges in encoder training topology, deterministic negative-sampling fixtures,
ranking metric reference tests, and a documented reproducibility report. Do not
call this architecture a reproduction until dataset and training settings match.

### 2. Exact SKAN and SKAN+ variants

Use the supplied GNN-SKAN PDF, especially Algorithm 1 and equations 7–11.

1. Add trainable epsilon and update `SKAN((1+epsilon)*h_i + m_i)` as a dedicated
   update policy, with operator-specific aggregation adapters.
2. Add per-edge/per-feature RBF center and bandwidth parameterization where
   required by the original SKAN implementation; compare it with the current
   shared-basis design rather than conflating them.
3. Match the exact base residual, RBF count, initialization and normalizations.
4. Provide SKAN with MLP classifier and SKAN+ with the specified two-layer SKAN
   classifier, as separately named presets.
5. Add molecular feature pipelines, scaffold splits and the paper's classification,
   regression and few-shot evaluation protocols.

Acceptance: manual equation checks and parameter counts; matched MLP vs SKAN vs
SKAN+ ablations; deterministic scaffold-split fixtures; clear mean/std metrics.

### 3. Exact Fourier KA-GCN and KA-GAT

Use `Lecture 1.pdf` (Nature Machine Intelligence 2025), not its generic filename
as an indication of content.

1. Implement node initialization from atom features and aggregated bond features.
2. Implement edge initialization from bonds plus endpoint atom features.
3. Add persistent, iteratively updated edge states and edge-derived attention
   matching the KA-GAT equations and supplement.
4. Match Fourier frequency count, scaling, residual structure and readout.
5. Compare exact KA-GCN/KA-GAT presets with the current generic Fourier hybrids.

Acceptance: molecular featurization fixtures, equation-level tests for edge-state
updates, matching parameter counts and a documented benchmark protocol.

## Priority 2: richer KAN tools

- Adaptive spline grid refinement with controlled coefficient transfer and
  optimizer-state handling. Test function preservation before/after refinement.
- A full layer-factory registry for completely custom KAN equations; the current
  registry extends bases only. Require serialization identifiers and validated
  input/output dimensions.
- Optional independent center/width parameters per input or input-output pair.
  Measure memory vs expressivity relative to the current shared centers.
- Additional bases such as Chebyshev, Legendre, rational and wavelet families,
  with explicit domain/scaling contracts and numerical derivative tests.
- Basis smoothness/entropy/sparsity penalties, symbolic fitting and pruning with
  separate tests and explicit objectives; keep supervised loss selection separate.
- Visualize learned edge functions, basis support, gradients and extrapolation.
- Chunked/fused basis contraction to reduce memory; verify gradient/output parity
  against the current reference implementation before optimizing.

## Priority 3: training and data pipelines

- A mini-batch trainer accepting PyG loaders and neighbor-sampled batches.
- Validation-based early stopping, best-checkpoint selection, schedulers,
  gradient clipping and training curves. Never choose models on the test split.
- Resume checkpoints with optimizer, scheduler, preprocessing, config and RNG
  state; test resumed training against uninterrupted training.
- Graph-level stratified/scaffold splitting and leakage-safe link splitting.
- Streaming preprocessing statistics; categorical encoders and missing-value
  policies fitted only on training data.
- Regression MAE/RMSE, binary/multilabel metrics, calibration and ranking metrics.
- Loss configuration dataclasses supporting kwargs, device-aware buffers,
  composite objectives and explicitly optimized trainable custom objectives.
- Explicit output/target validation to reject regression broadcasting and
  categorical dtype mistakes before optimization.
- Experiment metadata including data hashes, library versions, hardware, seeds,
  configurations and parameter counts. Add compatible dependency lockfiles for
  supported Python versions and use them in reproduction environments.

Acceptance: offline integration tests with tiny loaders, exact CPU resume
comparisons and separate train/validation/test fixtures.

## Priority 4: broader graph support and performance

- `SparseTensor`/sparse COO inputs and large-graph graph normalization caching.
- Bipartite tuple inputs and heterogeneous relation-specific layers.
- Configurable attention head concatenation, returned attention weights and
  explicit independent source/root transform configurations.
- Native residual/norm/activation policies rather than a fixed model ReLU.
- Graph edge-state update modules and canonical weighted variants where valid.
- Empty-graph batching policy and more extensive shape/device error reporting.
- GPU tests for documented deterministic-supported paths; separately characterize
  speed-focused nondeterministic CUDA runs.
- Benchmarks for time, peak memory and parameter count across MLP vs every KAN
  family, keeping widths, data splits and training budgets comparable.
- Evaluate torch.compile/export compatibility with numerical equivalence checks.

Acceptance: public compatibility matrix; no claim that GPU outputs are bitwise
identical across devices or framework versions.

## Priority 5: graphical interface

Build a local graphical configuration application on top of the current API:

1. Dataset upload and schema mapping for node, edge, batch and target tensors.
2. Task selection: node, graph or link; objective and split validation.
3. Visual stage controls for MLP/KAN/identity with independent basis/width options.
4. Backbone selection per layer and controls for attention heads and edge inputs.
5. Custom trusted Python extension loading with clear local-code execution consent;
   do not evaluate arbitrary uploaded code on a shared server.
6. Architecture preview, parameter/memory estimate, training progress, metrics and
   learned function plots.
7. JSON import/export and checkpoint downloads compatible with the Python API.

Acceptance: end-to-end tests covering stage changes, invalid combinations,
configuration roundtrip and a complete offline training example. Choose a UI
stack after clarifying whether the intended deployment is desktop, local web or
multi-user service.

## Priority 6: release engineering

- Run and observe the GitHub matrix after the repository is hosted/pushed.
- Expand CI across the stated minimum and current dependency versions.
- Pin action revisions and dependency locks as part of a maintenance policy.
- Install the built wheel in a clean environment and smoke-test it independently
  of the editable source tree in CI.
- Add versioned release notes, compatibility policy and documentation hosting.
- Choose a license before public distribution; the current repository does not
  assume a licensing choice for the user or the source papers.
- Configure PyPI trusted publishing only when package naming and publication are
  authorized. Current CI delivers artifacts but does not publish a package.
- Keep PR test workflows read-only; do not expose release credentials to fork PRs.

Acceptance: a tagged, reviewed release with green tests, validated wheel/sdist,
reproducible installation instructions, selected license and explicit publication
authorization where required.

## Update after reference source review

The review of `Lirain21/GNN-SKAN` has now supplied optional normalized RBF feature
layers, independently frozen centers/widths, shared positive bandwidths and
coefficient initialization controls through `skan_transform`. See
[SOURCE_REVIEW_GNN_SKAN.md](SOURCE_REVIEW_GNN_SKAN.md). These complete part of the
feature-layer configuration work above, not the full paper-model reproduction.
Future architecture work should distinguish source behavior from paper equations
and retain explicit graph direction/attention normalization contracts. Grid
adaptation still requires coefficient and optimizer-state policies before adoption.
