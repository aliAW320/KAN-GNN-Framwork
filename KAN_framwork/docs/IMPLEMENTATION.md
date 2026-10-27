# Implementation details and work log

Date: 27 October 2026. Package: `kangnn`, version `0.1.0`.

## 1. Scope and interpretation

The request is implemented as a reusable PyTorch package with a user-facing
Python configuration API and an offline JSON CLI. “Deciding” means the output
prediction head, and “instead of nlp” is interpreted as choosing KAN instead of
MLP, consistent with the explicit MLP/KAN requirement. No language model is
required. A graphical application is future work.

Required graph backbones: GCN, GraphSAGE, GAT, GATv2. Required function families:
B-splines, Fourier series, ReLU, RBF. Losses are selectable independently from
bases; custom basis modules and custom scalar loss callables are supported.
Node, graph, and link prediction share the same encoder and stage configuration.
The implementation supports homogeneous graphs with dense node features and
COO edge indices through PyTorch Geometric. It is not an implementation of every
possible GNN operation or dataset pipeline.

## 2. Repository inspection and paper review

The repository contained an empty `KAN_framwork` directory and research PDFs;
there was no package, test suite, or applicable repository AGENTS.md. Existing
PDFs and the `recommendation_Generation` directory were left intact.

All four root papers were converted to text with `pdftotext` for inspection.
Text extracts were placed under `/tmp`, not committed as duplicate paper copies.
The API-recommendation PDF emitted PDF parser warnings but yielded readable
architecture sections. Relevant methods and equations were inspected directly.

| Supplied PDF | Identity | Design contribution | Difference from this package |
|---|---|---|---|
| `original .pdf` | Liu et al., *KAN: Kolmogorov–Arnold Networks*, arXiv:2404.19756v5 | Trainable univariate edge functions and spline expansion | Fixed open uniform spline grids; no adaptive grid refinement, symbolic discovery, pruning, or exact original training procedure |
| `Lecture 1.pdf` | Li et al., *Kolmogorov–Arnold graph neural networks for molecular property prediction*, Nature Machine Intelligence, DOI 10.1038/s42256-025-01087-7 | Fourier KANs in node embedding, message passing, and readout | Generic graph features; no chemistry featurizer, molecular neighbor-bond initializer, iterative edge-state updates, or exact KA-GAT edge-attention architecture |
| `3711896.3736984.pdf` | Li et al., *GNN-SKAN: Advancing Molecular Representation Learning with SwallowKAN*, KDD 2025, DOI 10.1145/3711896.3736984 | Adaptive Gaussian RBF centers/bandwidths; SKAN update and optional SKAN classifier | Learnable RBF family and independent update/classifier choices supplied; exact `(1+epsilon)h + m` SKAN residual update and paper benchmark pipeline are not implemented |
| `KAN-GNN_ Kolmogorov-Arnold Network Inspired Graph Neural Networks for Personalized API Recommendation.pdf` | Kou and Xu, *KAN-GNN*, DOI 10.26599/TST.2026.9010002 | ReLU-derived basis, nonlinear graph encoder and learned pair decoder | ReLU basis and learned concatenation decoder supplied; difference messages, sigmoid gates, LayerNorm, ResGated decoder paths, bipartite sampler, and dataset-specific features are not reproduced |

The papers motivate the modular framework; selecting a similar basis does not
establish numerical reproduction of a paper. In particular, `residual=True` adds
a projected skip connection and is not the recommendation paper's gated layer.
The RBF basis shares centers and widths across feature edges; coefficients remain
independent on each input-output edge. Per-edge adaptive centers are future work.

The external compatibility reference was the official PyG implementation and
[operator overview](https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html),
with [GraphSAGE source](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/sage_conv.py)
and [GATv2 source](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py).
Automated numerical comparisons against the installed PyG operators verify the
implemented orientation, self-loop, normalization, and attention conventions.

## 3. Architecture and stage boundaries

```text
PyG Data or Batch
  x ---- optional training-fitted standardizer
    -> configurable preprocessing transform: input -> hidden
    -> one or more GraphLayer stages:
         source message projection: MLP or KAN
         optional edge projection: MLP or KAN
         GAT/GATv2 target and attention transforms
         invariant graph aggregation
         optional SAGE root contribution
         bias -> configurable update transform
         optional projected skip connection
         ReLU -> model dropout
    -> graph task only: mean/sum/max pool by batch ID
    -> configurable readout transform
    -> link task only: concatenate source and target query embeddings
    -> configurable decision transform
    -> raw scores/logits
    -> independent supervised loss + optional KAN coefficient L1 penalty
```

Graph-layer aggregation remains a mathematical reduction. Replacing a sum with
an arbitrary dense transform over neighbor ordering would break permutation
equivariance. The framework exposes MLP/KAN choices around this reduction instead.
Preprocessing has two distinct pieces: statistical feature preparation and the
learned embedding transform. Only the learned embedding is MLP/KAN-selectable.

`TransformConfig` configures type, basis, basis kwargs, hidden widths, bias and
SiLU base residual. `LayerConfig` chooses the backbone and its stage transforms,
heads, attention dropout, self loops, edge dimensions, and projected residual.
`ModelConfig` supplies widths, per-layer configs, preprocessing, readout,
decision, task, graph pooling, and inter-layer dropout. JSON dictionaries are
converted into these typed config objects; unknown fields are rejected by the
dataclass constructor. `to_dict()` supports JSON round trips.

The public API is exported from `kangnn/__init__.py`. Registries reject duplicate
names so user extensions cannot silently replace a built-in. Model construction
uses only named trusted Python factories; no JSON code evaluation is performed.

## 4. KAN equations and implementation

For input x with features i and outputs o:

```text
y[o] = sum_i sum_k C[o,i,k] B_k(x[i]) + W_base[o,:] SiLU(x) + bias[o]
```

`C` is a learned `[out_channels, in_channels, num_basis]` tensor. Bases produce
`[..., in_channels, num_basis]`, contracted with coefficients through `einsum`.
Base residual and bias can be disabled independently. Coefficients initialize
with normal standard deviation `1/sqrt(in_channels*num_basis)`. The base branch
uses PyTorch Linear initialization. Multiple hidden widths stack KAN layers;
MLP hidden layers use ReLU, with no output activation.

### B-splines

`BSplineBasis` builds an open uniform knot vector with repeated endpoints and
uses Cox–de Boor recursion. For degree p and n basis functions, require n > p >= 0.
Zero-width repeated-knot intervals have zero contribution, with safe denominators
to avoid NaN gradients. Inputs clamp to `[low, high]`; the exact right endpoint
explicitly activates the final basis. Tests check nonnegativity, partition of
unity, compact support, endpoint behavior and numerical derivatives.

These are fixed knots, not the original KAN paper's adaptive grid updates.
Clamping yields zero spline derivative beyond the interval; choose an appropriate
domain or retain the SiLU base branch.

### Fourier series

For odd basis count `2H+1`, the basis contains:

```text
1, sin(2*pi*h*x/period) for h=1..H, cos(2*pi*h*x/period) for h=1..H
```

Frequencies are buffers, not parameters. Harmonic coefficients are learned by the
KAN layer. Period defaults to `2*pi`. Tests verify periodicity and gradients.
The basis is globally supported and periodic; normalization and period matter.

### ReLU ramps

```text
B_k(x) = ReLU(x - center_k) - ReLU(x - center_k - softplus(raw_width_k))
```

Centers and widths are trainable. This matches a shifted/scaled ReLU-ramp family
motivated by the recommendation paper's localized piecewise-linear basis. It is
not an ordinary MLP ReLU activation. Its coefficient expansion lives on KAN
feature edges, and it does not imply the paper's full residual gated architecture.

### Gaussian RBFs

```text
B_k(x) = exp(-0.5 * ((x-center_k)/(softplus(raw_width_k)+1e-6))**2)
```

Centers initialize uniformly over the domain and widths initialize from raw zero.
Both are trainable. Positive widths avoid invalid Gaussian scales. The domain is
an initialization range, not a clipping interval. Tests exercise center and width
gradients as well as coefficient gradients.

Custom bases implement the same tensor contract and may add trainable parameters
or buffers. The factory validates module type and advertised basis count; the
KAN forward validates output shape. Users are responsible for finite values,
differentiability and function semantics. The runnable polynomial example shows
registration, custom training and ordinary autograd integration.

## 5. Graph equations and conventions

Edges `edge_index[:,e] = [j,i]` send information from source j to target i. Graph
layers subclass PyG `MessagePassing` with `node_dim=0`; node tensors have shape
`[N,H,C]` internally. `propagate(..., size=(N,N))` preserves isolated nodes.

### GCN

```text
A_hat = A + I (when configured; only missing loops are added)
z = message_transform(x)
out_i = sum_j A_hat[j,i] / sqrt(degree[j]*degree[i]) * z_j
```

The implementation uses PyG `gcn_norm` with source-to-target flow and incoming
weighted degrees, matching `GCNConv`, including on directed graphs. Nonnegative
finite edge weights are optional. For directed data, these are PyG's conventions;
undirected graphs require both directions in the edge list. With nonlinear
messages this is a KAN-enhanced GCN variant; a nonlinear transform does not
commute with aggregation.

### GraphSAGE

```text
out_i = mean_{j in neighbors(i)} message_transform(x_j) + root_transform(x_i)
```

The root and neighbor modules have independent parameters but use the same
transform config. No loops are added automatically because the root path already
represents the central node. Neighbor means on isolated nodes are zero.
Layer bias is applied after aggregation. With a bias-free linear message config
and identity update, this matches PyG's mean `SAGEConv`.

### GAT

```text
z_i = source_transform(x_i)
logit_ji,h = LeakyReLU(att_src_h(z_j,h) + att_dst_h(z_i,h), slope=0.2)
alpha_ji,h = softmax over incoming j for each destination i and head h
out_i,h = sum_j dropout(alpha_ji,h) * z_j,h
```

Scorers are independent per head and may be scalar-output MLPs or KANs. GAT
shares the source/target feature projection, as in the standard homogeneous
operator. Extra attention scorer bias or nonlinear transforms are configurable
extensions, not baseline GAT equivalence settings.

### GATv2

```text
z_j = source_transform(x_j), q_i = target_transform(x_i)
logit_ji,h = att_h(LeakyReLU(z_j,h + q_i,h, slope=0.2))
alpha and aggregation follow GAT
```

The order of nonlinearity and scoring is deliberately different from GAT.
Feature projections for source and target are independent. In both attention
operators, each head produces C channels and outputs average across heads, as in
PyG `concat=False`. Attention normalization is the PyG grouped softmax. Dropout
acts on normalized weights during training, so dropped weights need not sum to 1.

### Common post-processing and edge features

Head outputs are averaged. SAGE adds its root. The layer then applies a learned
output bias, the update transform, and optionally a bias-free linear skip from
original input. The end-to-end model adds ReLU and model dropout after every
message-passing layer. Standalone `GraphLayer` omits that outer ReLU/dropout so
numerical parity with PyG operators is directly testable.

Optional edge attributes use a configurable encoder and add to source values.
For attention, these modified values also influence scores. New attention loops
get zero feature vectors. GCN's internal loop normalization can reorder/change
loop entries, so edge-attribute use requires `self_loops=False` and explicit
aligned loops. Edge weights are rejected for non-GCN backbones rather than
silently ignored. SparseTensor, bipartite tuple inputs, heterogeneous graphs, and
ragged feature dimensions are not part of the current operator contract.

## 6. Training, evaluation, splits and checkpoints

`seed_everything` seeds Python, NumPy, CPU PyTorch and available CUDA generators,
uses one PyTorch CPU thread, enables strict deterministic algorithms and sets the
cuBLAS workspace configuration. Call it before constructing data-dependent random
objects and models. Repeated CPU runs in the same environment are the guarantee;
there is no cross-version or cross-device bitwise guarantee.

`node_split` uses its own seeded torch.Generator and returns disjoint exhaustive
boolean masks. Each requested split must contain at least one node.
`FeatureStandardizer` learns mean and population standard deviation from training
rows only, protects constant columns with a minimum scale, and stores statistics
as module buffers. Transforming data does not mutate it. Graph-level splits and
leakage-free link splits remain the caller's responsibility.

`fit` runs full-batch Adam with configurable epochs, learning rate, weight decay
and KAN coefficient penalty. A label mask is applied before the objective.
Empty/invalid masks, non-scalar custom losses and non-finite losses fail clearly.
K-fold CV, minibatch loaders, early stopping, schedulers, metrics beyond accuracy
and loss, mixed precision and optimizer restart are future work.

Built-in objectives wrap PyTorch CE, BCEWithLogits, MSE, L1, SmoothL1 and NLL. The
trainer applies log-softmax for the named NLL path; all model outputs stay raw.
Callables and registered loss factories are accepted. `make_loss` exposes kwargs
such as label smoothing or class weights; the caller places loss buffers on the
correct device. Custom loss parameters are not added to Adam automatically.
The optional `kan_l1` penalty sums mean absolute coefficient values over KAN
layers; it is not an implementation of the original KAN entropy/pruning penalty.

`evaluate` runs under no-grad, restores the original model training mode, and
returns objective value plus multiclass accuracy for integer one-dimensional
labels. Evaluation excludes coefficient regularization and any training dropout.
It does not select an optimal checkpoint automatically.

Checkpoints contain a format version, serializable model config and model state
dict. Loading uses `torch.load(..., weights_only=True)`. Custom Python factories
must already be registered. Optimizer, scaler, graph data, loss and RNG states
are excluded; persist these explicitly when required. Loading returns a newly
constructed model in training mode, so call `.eval()` for prediction.

## 7. File-by-file work completed

1. Created `pyproject.toml` for setuptools packaging, dependencies, test extras,
   lint configuration and the `kangnn` entry point.
2. Created `src/kangnn/basis.py`: B-splines, Fourier, ReLU ramps, RBFs and custom
   registration, with domain/count checks and trainable basis parameters.
3. Created `transforms.py`: common transform config, KAN coefficient contraction,
   optional SiLU residual, MLP/identity factories and deep width support.
4. Created `graph.py`: four MessagePassing backbones, independent attention
   scorers per head, graph normalization, loops, edge values, validation and skips.
5. Created `model.py`: per-layer composition, node/graph/link outputs,
   permutation-invariant pooling, configurable readout and ordered pair decoding.
6. Created `data.py`: training-fitted standardizer, deterministic local-RNG node
   splits and a fixed offline graph fixture.
7. Created `training.py`: supervised loss registry, seed control, Adam trainer,
   masks, coefficient regularization and mode-preserving evaluation.
8. Created `checkpoint.py`: versioned config/state persistence and loading.
9. Exported the public API in `__init__.py`; created `cli.py` for CLI/JSON demos.
10. Added mixed configuration and custom-extension examples.
11. Added basis, operator and full-pipeline tests with an autouse seed fixture.
12. Added GitHub Actions for CPU tests, lint, coverage, examples, source/wheel
    packages and artifact upload; no publishing credentials required.
13. Wrote the README, this implementation log, a separate future-work plan and
    source-distribution manifest. Added ignore rules for environments and outputs.
14. Installed a workspace-local virtual environment. The initial sandbox download
    could not resolve the package host; approved network execution successfully
    installed CPU PyTorch, PyG and testing/build dependencies.
15. Ran the numerical and integration tests, lint, examples and package build.
    Changed attention scoring to independent head modules, added multihead parity
    tests and stage-isolation checks. Added CLI in-process coverage and packaging
    metadata after the initial successful test run.

## 8. Automated verification design

All tests use deterministic synthetic tensors/graphs and a seed fixture. There
are no dataset downloads or external service calls during tests.

| Capability | Automated checks |
|---|---|
| Four bases | Shapes, finite outputs, gradcheck, state roundtrip; spline partition/support/endpoints, Fourier periodicity, adaptive RBF gradients |
| KAN formula | Manual coefficient contraction, base residual on/off, coefficient/input gradients |
| MLP and KAN depth | Multiple hidden widths and identity shape constraints |
| Custom basis | Exact polynomial-like output, registration and duplicate rejection |
| Four backbones | Numerical forward parity with installed PyG baseline, with/without added loops, directed degrees and isolated nodes |
| Weighted GCN | Weighted normalization parity |
| Attention | GAT/GATv2 distinction, multiple-head parity, KAN scorers across every basis, dropout modes/reseeding |
| Message/update choices | All four backbones crossed with MLP/four KAN bases, gradients and permutation equivariance |
| Edge features | Each backbone, aligned empty edges, gradients, attention loop handling and configurable KAN edge transforms |
| All task types | 4 backbones x 5 transform choices x 3 tasks: training, shapes, evaluation and config roundtrip |
| Independent stages | One KAN stage at a time among MLPs for all four bases across preprocessing, message, update, attention, edge, readout, decision |
| Graph batching/pooling | Mean/sum/max, batched vs independent output parity, graph permutation invariance |
| Losses | Every built-in objective trains; custom callable/module/registered objective; invalid and nonfinite objective rejection |
| Deterministic training | Two 40-epoch KAN runs produce exactly equal histories and parameters; toy loss must fall substantially |
| Leakage protection | Held-out label changes cannot affect masked training; held-out feature changes cannot affect fitted scaler statistics |
| Persistence | All four bases have exact CPU prediction roundtrips; fitted scaler state roundtrip |
| CLI | Subprocess smoke run, MLP/NLL/Fourier/JSON code paths, saved model, invalid task rejection |

This is systematic coverage of documented features, not enumeration of every
possible user-provided module, hyperparameter combination, graph or floating
point backend. Numerical comparisons use small tolerances where summation order
can vary. Deterministic repeated-run comparisons use exact equality.

## 9. CI/CD and validation record

The workflow `.github/workflows/ci.yml` runs on pushes, PRs and manual dispatch.
Python 3.10 and 3.12 jobs install CPU PyTorch 2.5.1 and PyG 2.6.1, lint the package,
run deterministic tests with a 90% coverage gate, and execute examples. A package
job depends on both test jobs, builds a source distribution and universal wheel,
checks metadata with twine, and uploads downloadable artifacts. Coverage XML is
also retained. Workflow permissions are read-only and no production deployment
or package publication is configured. Artifact delivery is the current CD scope.

Local tests ran using Python 3.14.7, torch 2.14.1+cpu and torch-geometric 2.8.1.
PyG emitted Python 3.14 annotation deprecation warnings; Torch emitted a JIT
support warning. These are upstream warnings, not test failures. CI uses the
older pinned backend pair to check another supported environment, but that
remote matrix has not been executed in this session. Dependency ranges are not
a complete transitive lockfile; reproduce exact experiments with a frozen
working environment and hardware record.

Initial verification: 143 tests passed, 94.66% line coverage; lint passed; both
examples executed; source and wheel builds succeeded. Final verification results
are appended below after all additions and documentation are complete.

## 10. Current boundaries

The framework is usable for small/medium homogeneous graph experiments and as
building blocks for normal PyG training loops. It provides a high-level API and
CLI, not a browser UI. It does not claim benchmark accuracy, exact paper
reproduction, production-scale throughput, GPU repeatability, or compatibility
with every PyG operator option. Basis evaluation materializes feature-by-basis
activations, so large edge batches may consume substantial memory.

Paper-specific gated difference messages, exact SKAN updates, molecular edge-state
updates, adaptive spline grids, explainability plots and real dataset pipelines
are explicitly planned in the separate future-work file. The root source PDFs
remain the authoritative references for those specialized architectures.

### Final local verification

- `pytest KAN_framwork/tests -q --cov=kangnn --cov-report=term-missing
  --cov-report=xml:/tmp/kangnn-coverage.xml --cov-fail-under=90`:
  **146 passed**, **94.66% line coverage**; approximately 6.4 seconds locally.
- `ruff check KAN_framwork`: **passed**.
- Default RBF-KAN GATv2 CLI, 60 epochs, seed 7: toy training-set accuracy 1.0,
  evaluation CE approximately `0.00003606`.
- Mixed JSON CLI, 60 epochs, seed 7: toy training-set accuracy 1.0. CE reached
  floating-point zero on this tiny fixture; this is not held-out evidence.
- Custom polynomial basis and registered custom CE example: executed successfully;
  objective decreased from approximately `148.708817` to `0.236872` in 20 epochs.
- `python -m build KAN_framwork --no-isolation`: source archive and wheel built
  successfully. Confirmed the source archive includes docs and JSON examples.
- Installed the wheel with `--no-deps --target /tmp/kangnn-wheel-validation` for
  an import/CLI smoke check using the existing tested dependency environment.
- GitHub-hosted jobs and their Python 3.10/3.12 matrix are configured, not remotely
  executed. Twine metadata validation is part of CI, not the local verification.

Local coverage/build logs and wheel validation directories are disposable;
workspace `.venv`, build directories and distribution artifacts are ignored by
version control. Test warnings originate from upstream Python 3.14 compatibility
paths and remain visible instead of being hidden globally.

## Detailed user-guide addition

Added `docs/USER_GUIDE.md` in response to the request for detailed correct usage.
Reviewed the current model, graph, transform, training, preprocessing, checkpoint
and optional numerical-transform implementations before writing examples.

The guide contains 20 sections and 16 independent Python examples covering
installation, tensor schemas, normalization and label masks, per-stage MLP/KAN
choices, all four backbones, basis selection, objective/target conventions,
regression, graph minibatches, explicit leakage-safe link fixtures, edge
attributes/weights, GPU setup, custom losses/bases, Z/Laplace integration,
checkpoint/scaler roundtrips and JSON configuration. Configuration tables and
troubleshooting document the implemented behavior, including checkpoint dtype
limitations, full-batch optimizer resets and directed pair decoding.

Added `tests/test_user_guide.py` to extract each Python block and execute it in an
independent subprocess with a temporary working directory. Documentation checks
target CPU, avoid dataset downloads and skip optional transform snippets if that
separate package is not installed. Existing workflows discover this test; the
transform workflow installs both packages and therefore exercises all snippets.
Linked the guide from the root and both package READMEs.

Verification: **16 documentation examples passed** in approximately 44 seconds;
`ruff check KAN_framwork mathematical_transforms` passed; guide code fences were
balanced and local guide links resolved. No framework runtime code was changed.

## Reference-source-informed SKAN options

Reviewed the user-supplied `Lirain21/GNN-SKAN` source at commit
`5a352a170dadb8a0fa3f66eb92b6be0975ef15df` without executing upstream code.
The detailed comparison, attribution, reuse boundary, selected improvements and
remaining architecture differences are in [SOURCE_REVIEW_GNN_SKAN.md](SOURCE_REVIEW_GNN_SKAN.md).

Added optional basis-only LayerNorm and coefficient initializer controls to
TransformConfig/KANLinear. Extended the existing RBF with independent center/width
training flags, positive explicit initial widths and shared-width support.
Exported `skan_transform()` for convenient SKAN-style feature configuration;
existing defaults and unnormalized state-dict keys remain compatible. Added
`examples/skan_style.py`, tests for equations/gradients/config/checkpoint behavior
and repeated training, a runnable user-guide section and updated CI example runs.
No upstream files were copied or added as dependencies. Exact molecular-model
reproduction and grid adaptation remain deferred.
