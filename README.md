# KAN-GNN: configurable graph neural networks

Build PyTorch Geometric networks with **MLP or KAN transforms selected independently
at each learned stage**. Combine GCN, GraphSAGE, GAT and GATv2 layers for node,
graph or link prediction, choose a basis and loss, or register your own functions.

A separate package supplies differentiable numerical Z/Laplace transforms and
transform-inspired KAN bases. The interface is a Python API and JSON-driven CLI.

**Start here:** [Detailed user guide with runnable examples](KAN_framwork/docs/USER_GUIDE.md).

## Contents

- [Capabilities](#capabilities)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture and stage selection](#architecture-and-stage-selection)
- [Graph operators](#graph-operators)
- [KAN bases and SKAN options](#kan-bases-and-skan-options)
- [Tasks and data contracts](#tasks-and-data-contracts)
- [Losses and training](#losses-and-training)
- [Custom bases and losses](#custom-bases-and-losses)
- [Z and Laplace transforms](#z-and-laplace-transforms)
- [GPU and determinism](#gpu-and-determinism)
- [Saving models and configurations](#saving-models-and-configurations)
- [Command-line interface](#command-line-interface)
- [Tests and GitHub Actions](#tests-and-github-actions)
- [Repository layout and documentation](#repository-layout-and-documentation)
- [Research references and limitations](#research-references-and-limitations)

## Capabilities

| Area | Implemented capabilities |
|---|---|
| Graph backbones | GCN, mean GraphSAGE, GAT, GATv2; mix operators across layers |
| Learned stages | MLP/KAN preprocessing, messages, node updates, edge encoding, attention scoring, readout and prediction |
| Transform composition | Multiple hidden widths, dimension-preserving identity, optional KAN SiLU base branch and biases |
| Built-in KAN bases | B-spline, Fourier series, learnable ReLU ramps, adaptive Gaussian RBF |
| SKAN-style options | Basis-only LayerNorm, independent RBF center/width freezing, shared bandwidths, coefficient initialization |
| Prediction tasks | Node classification/regression, graph classification/regression, ordered link scoring |
| Graph features | Directed COO edges, isolated nodes, optional edge attributes, nonnegative GCN edge weights |
| Attention | Multiple independently scored heads, averaged output, attention dropout |
| Readout | Graph mean/sum/max pooling and configurable learned readout transform |
| Training | Full-batch Adam, label masks, weight decay, KAN coefficient L1 regularization, custom training loops |
| Objectives | CE, BCEWithLogits, NLL, MSE, L1, SmoothL1, custom loss factories/callables |
| Data preparation | Training-fitted feature standardization and seeded node splits |
| Persistence | JSON-compatible model configs and versioned model config/state checkpoints |
| Numerical transforms | Finite Z sums and finite-interval sampled Laplace integration, real/complex autograd |
| Extensions | Custom basis registration, custom loss registration, optional Z/Laplace KAN bases |
| Verification | Deterministic CPU tests, numerical PyG parity, gradient checks, executable documentation, conditional CUDA tests |
| Automation | Python 3.10/3.12 GitHub Actions, lint, coverage gates, examples, wheel/source builds and downloadable artifacts |

## Installation

Requires Python **3.10+**. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './KAN_framwork[test]'
# Optional numerical transforms and additional KAN bases:
python -m pip install -e './mathematical_transforms[test]'
```

If the environment already exists, activate it without recreating it. The core
package depends on PyTorch, PyTorch Geometric and NumPy; the transform package
needs only PyTorch unless using its optional graph-framework integration.
The `[test]` extras install pytest, coverage and Ruff. Install without those extras
for runtime-only use.

The command above installs CPU PyTorch. For CUDA, install a CUDA-enabled PyTorch
build appropriate to your system instead. Then install the packages as above.

Directory names and import names differ:

| Directory | Distribution | Python import |
|---|---|---|
| `KAN_framwork/` | `kangnn` | `kangnn` |
| `mathematical_transforms/` | `kan-transforms` | `kan_transforms` |

## Quick start

Run the offline CLI demo:

```bash
kangnn --operator gatv2 --kind kan --basis rbf --epochs 60
```

Or train through Python with explicit train/validation/test labels:

```python
import torch
from kangnn import (
    FeatureStandardizer, GraphKAN, LayerConfig, ModelConfig, TrainingConfig,
    TransformConfig, evaluate, fit, node_split, seed_everything,
)
from kangnn.data import toy_graph

seed_everything(7)
data = toy_graph()
train, validation, test = node_split(6, train=0.5, validation=0.25, seed=7)
scaler = FeatureStandardizer(data.num_features).fit(data.x, train)
data.x = scaler(data.x)

kan = TransformConfig(kind="kan", basis="rbf", basis_kwargs={"num_basis": 8})
model = GraphKAN(ModelConfig(
    in_channels=2, hidden_channels=8, out_channels=2,
    preprocessing=kan,
    layers=[LayerConfig(operator="gcn", message=kan, update=kan)],
    decision=TransformConfig(kind="mlp"),
    task="node",
))
history = fit(model, data, loss="cross_entropy",
              config=TrainingConfig(epochs=60, lr=0.01, kan_l1=1e-4), mask=train)
print("Training objective:", history[0], "->", history[-1])
print("Validation:", evaluate(model, data, mask=validation))
print("Test:", evaluate(model, data, mask=test))
model.eval()
with torch.no_grad():
    predicted_classes = model(data).argmax(-1)
    embeddings = model.encode(data)
print("Predictions:", predicted_classes.tolist())
assert embeddings.shape == (6, 8)
```

The fixture is deliberately tiny; its metrics are not evidence of benchmark
quality. In real experiments, select settings on validation data and evaluate
test data only after model selection. A node mask restricts labels, not topology;
this example is transductive. For inductive work, construct appropriate subgraphs.

## Architecture and stage selection

```text
Numerical node features
  -> learned preprocessing: MLP / KAN
  -> graph layers:
       message projection + optional edge encoder
       attention scoring where applicable
       neighborhood aggregation
       node update + optional projected residual
       ReLU + activation dropout
  -> graph pooling for graph tasks
  -> readout transform
  -> endpoint concatenation for link tasks
  -> decision head
  -> raw predictions -> supervised objective
```

| Stage | Configuration |
|---|---|
| Learned input embedding | `ModelConfig.preprocessing` |
| Source message projection | `LayerConfig.message` |
| Post-aggregation node transform | `LayerConfig.update` |
| Edge-feature embedding | `LayerConfig.edge`, with `edge_dim` |
| Scalar attention scorers | `LayerConfig.attention` |
| Learned readout | `ModelConfig.readout` |
| Prediction head | `ModelConfig.decision` |

Use a separate `TransformConfig` for each stage. For example:

```python
from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig
from kangnn.data import toy_graph

mlp = TransformConfig(kind="mlp", hidden=(16,))
rbf = TransformConfig(kind="kan", basis="rbf")
fourier = TransformConfig(kind="kan", basis="fourier", basis_kwargs={"num_basis": 9})
model = GraphKAN(ModelConfig(
    2, 8, 2, preprocessing=rbf,
    layers=[
        LayerConfig(operator="sage", message=mlp, update=rbf),
        LayerConfig(operator="gatv2", heads=2, message=fourier,
                    attention=TransformConfig(kind="mlp", bias=False),
                    update=mlp, residual=True),
    ],
    readout=rbf, decision=mlp,
))
assert model(toy_graph()).shape == (6, 2)
```

A default MLP is a linear map; `hidden=(32,16)` adds hidden ReLU layers. KAN hidden
widths stack KAN feature layers. `kind="identity"` requires equal input/output
widths and no hidden widths; updates and readout default to identity.

Aggregation/pooling remain permutation-invariant reductions. Splitting, graph
construction and statistical standardization are also algorithmic operations.
They are not replaced with arbitrary dense MLP/KAN transforms. Graph-layer output
ReLU is currently fixed in the high-level model.

## Graph operators

| Operator | Neighborhood rule | Configuration details |
|---|---|---|
| `gcn` | Degree-normalized sum | Optional nonnegative weights; adds missing self loops by default |
| `sage` | Neighbor mean plus independent root path | No automatic loops; root/message modules share config, not parameters |
| `gat` | Attention-weighted incoming sum | Shared source/target value projection; independent scorers per head |
| `gatv2` | Attention after source/target vector interaction | Independent source/target projections |

Heads are averaged, equivalent to the output-width convention of `concat=False`.
`hidden_channels` therefore stays constant when increasing head count. GCN/SAGE
require `heads=1`. Attention-weight dropout is `LayerConfig.dropout`; activation
dropout after each graph layer is `ModelConfig.dropout`.

`GraphLayer` subclasses PyG `MessagePassing` and can be used separately from
`GraphKAN`. It accepts feature tensors and COO edges directly. Standalone layers
omit the high-level model's outer ReLU/dropout.

## KAN bases and SKAN options

| Basis name | Default | Behavior |
|---|---|---|
| `bspline` | 8 bases, degree 3, domain `(-1,1)` | Fixed open uniform knots; inputs clamp outside domain |
| `fourier` | 9 bases, period `2*pi` | Constant plus sine/cosine harmonics; odd count >=3 |
| `relu` | 8 bases, domain `(-1,1)` | Learnable shifted finite-width ReLU ramps |
| `rbf` | 8 bases, domain `(-1,1)` | Gaussian kernels; learnable centers and positive widths |
| `laplace` (extension) | 8 bases, time range `(0.1,4)` | Real positive-query Laplace kernel expansion |
| `z_transform` (extension) | 8 bases | Real positive-query finite inverse-power expansion |

Basis options go into `basis_kwargs`. B-splines require `num_basis > degree`.
Standardization does not guarantee inputs fall in a spline interval. RBF/ReLU
`domain` initializes locations rather than clipping inputs; Fourier uses `period`.

KAN controls in `TransformConfig`:

| Option | Default | Purpose |
|---|---|---|
| `base_residual` | `True` | Learned mapping of raw-input SiLU alongside basis features |
| `bias` | `True` | Feature-layer output bias |
| `basis_normalization` | `"none"` | Optional `"layernorm"` on the basis branch only |
| `coefficient_init` | `"normal"` | Also supports `"truncated_normal"` |
| `coefficient_init_std` | `None` | Explicit positive std, otherwise fan/basis-scaled |

RBF additionally accepts `width`, `shared_width`, `trainable_centers` and
`trainable_widths`. Width is Gaussian standard deviation and remains positive
through softplus. Default RBF widths remain per-basis and trainable.

Use `skan_transform()` to combine SKAN-style feature-layer options:

```python
from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig, skan_transform
from kangnn.data import toy_graph

update = skan_transform(num_basis=8, domain=(-2., 2.),
                        trainable_centers=True, trainable_widths=False)
model = GraphKAN(ModelConfig(
    2, 8, 2,
    layers=[LayerConfig(operator="gcn", message=TransformConfig(kind="identity"),
                        update=update)],
    decision=skan_transform(num_basis=4, domain=(-4., 8.)),
))
assert model(toy_graph()).shape == (6, 2)
```

The preset normalizes RBF inputs, initializes shared bandwidth from grid spacing,
uses truncated-normal coefficient std 0.1 and retains a raw-input SiLU branch.
Identity messages put the nonlinear KAN in the update after aggregation.
This is not a complete reproduction of the upstream molecular model.
See the [source comparison](KAN_framwork/docs/SOURCE_REVIEW_GNN_SKAN.md).

## Tasks and data contracts

All tasks consume PyG `Data` or `Batch` with floating `x: [N,F]` and int64
`edge_index: [2,E]`. Row 0 is source, row 1 is destination. Use contiguous node
IDs and both edge directions for undirected relationships. Supply finite numerical
features; this package does not generate text embeddings.

| Task | Model setting | Output | Data/target preparation |
|---|---|---|---|
| Node classification | `task="node"`, output C classes | `[N,C]` | Long class labels `[N]`, optional boolean masks |
| Node regression | `task="node"`, output T targets | `[N,T]` | Float targets `[N,T]` |
| Graph classification | `task="graph"`, output C classes | `[G,C]` | PyG Batch; one long label per graph |
| Graph regression | `task="graph"`, output T targets | `[G,T]` | Per-graph float `y: [1,T]` before batching |
| Binary link prediction | `task="link"`, output 1 | `[Q,1]` | Query pairs `[2,Q]`, float labels `[Q,1]` |

Graph tasks pool with `mean`, `sum` or `max`. Unbatched graph data is one graph.
Link tasks concatenate ordered endpoint embeddings and use a learned head;
`edge_label_index` is separate from message-passing `edge_index`. Query pairs
can also be supplied directly as `model(data, edge_label_index=pairs)`.
The decoder is directed; symmetric scoring requires an explicit policy.

Remove held-out positive links, including reverse directions for undirected
splits, from training message edges. Construct valid negatives yourself. The
framework does not automatically provide negative sampling or ranking metrics.

Optional edge fields:

- `edge_attr: [E,edge_dim]`: configure matching `edge_dim` in every layer receiving
  it. Encoded edges add to source values and also affect attention.
- `edge_weight: [E]`: finite nonnegative GCN-only weights. A mixed model forwards
  this field to all layers, so non-GCN layers reject it.
- GCN with edge attributes requires `self_loops=False`; prepare aligned loops and
  features yourself if needed. Attention replaces loops and assigns new loops
  zero raw edge features. An encoder bias can turn those zeros into nonzero values.

See the user guide for complete [regression](KAN_framwork/docs/USER_GUIDE.md#9-node-regression),
[graph minibatch](KAN_framwork/docs/USER_GUIDE.md#10-graph-classification-and-minibatches),
[link](KAN_framwork/docs/USER_GUIDE.md#11-link-prediction-without-held-out-edge-leakage),
and [edge-feature](KAN_framwork/docs/USER_GUIDE.md#12-edge-attributes-and-weights) examples.

## Losses and training

| Loss string | Inputs/targets |
|---|---|
| `cross_entropy` | Raw logits `[P,C]`, long class IDs `[P]` |
| `bce_logits` | Raw logits and float labels with identical shape `[P,T]` |
| `nll` | Helper applies log-softmax to logits; long class IDs `[P]` |
| `mse`, `l1`, `smooth_l1` | Predictions and float targets of identical shape `[P,T]` |

P is nodes, graphs or query pairs. For one regression/binary target, keep `[P,1]`
rather than `[P]`. Do not apply softmax before CE or sigmoid before BCEWithLogits.
Apply them only for interpreting predictions.

`fit(model,data,loss,config,mask)` performs full-batch Adam updates and returns a
loss history. `TrainingConfig` provides `epochs=100`, `lr=0.01`,
`weight_decay=0.0` and `kan_l1=0.0`. The KAN penalty sums mean absolute coefficient
values across KAN layers. The helper rejects empty masks, nonscalar losses and
nonfinite objectives. A fresh optimizer is created on each call.

`evaluate` disables gradients, temporarily switches to evaluation mode and
restores the previous mode. It returns supervised loss, plus multiclass accuracy
for one-dimensional long targets. Binary, regression and ranking metrics beyond
loss must be computed separately. Evaluation excludes coefficient regularization.

`FeatureStandardizer` fits mean/scale using training rows only and persists its
statistics as buffers. `node_split` returns disjoint seeded CPU train/validation/
test masks. Choose graph-level or leakage-safe edge splits separately.

Use normal PyTorch/PyG loader loops for minibatch training, custom optimizers,
schedulers or trainable loss parameters. The model integrates with those loops;
the convenience `fit` helper does not take a loader or optimize loss parameters.

## Custom bases and losses

Extend the framework without changing its source:

```python
import torch
from torch import nn
from kangnn import (GraphKAN, ModelConfig, TrainingConfig, TransformConfig,
                    fit, make_loss, register_basis, register_loss, seed_everything)
from kangnn.data import toy_graph

class PolynomialBasis(nn.Module):
    def __init__(self, num_basis=4):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("powers", torch.arange(num_basis))

    def forward(self, x):
        return x.tanh().unsqueeze(-1).pow(self.powers)

class SmoothedCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        return nn.functional.cross_entropy(predictions, targets,
                                            label_smoothing=self.smoothing)

register_basis("readme_polynomial", PolynomialBasis)
register_loss("readme_ce", SmoothedCE)
seed_everything(7)
custom = TransformConfig(kind="kan", basis="readme_polynomial")
model = GraphKAN(ModelConfig(2, 8, 2, preprocessing=custom, decision=custom))
objective = make_loss("readme_ce", smoothing=0.05)
fit(model, toy_graph(), loss=objective, config=TrainingConfig(epochs=3))
```

A basis maps `[...,F]` to `[...,F,num_basis]`. It can contain trainable parameters;
store constants as registered buffers for device/state handling. Custom losses
must return a differentiable scalar. `fit` also accepts a direct loss callable.
`make_loss` accepts PyTorch options such as class weights or label smoothing.

Registration names must be unique. Register custom classes at startup and before
checkpoint reconstruction. The registry extends basis functions; entirely custom
KAN equations require composing/subclassing modules. JSON never evaluates code.
Direct `make_loss("nll")` expects log probabilities; automatic conversion occurs
only in the named `loss="nll"` helper path.

## Z and Laplace transforms

The independent `mathematical_transforms/` package has numerical tools and KAN
bases. A signal transform and a per-feature KAN basis are distinct operations.

```python
import torch
from kan_transforms import laplace_transform, z_transform

sequence = torch.tensor([1., 2., 3.], dtype=torch.float64)
z = torch.tensor([1., 2., 1j], dtype=torch.complex128)
print(z_transform(sequence, z)) # [6, 2.75, -2-2j]
times = torch.linspace(0, 10, 10001, dtype=torch.float64)
signal = torch.exp(-2 * times)
s = torch.tensor([0., 1., 1.+1j], dtype=torch.complex128)
print(laplace_transform(signal, times, s))
```

- Z evaluates a finite indexed sum; `start_index` supports shifted indices.
  `z=0` is rejected. No infinite-sequence convergence region is inferred.
- Laplace uses trapezoidal quadrature over the supplied finite interval.
  Times must be real, nonnegative, strictly increasing and contain at least two
  points. Missing tails and quadrature error remain the caller's responsibility.
- Both accept batch dimensions, a sample `dim` and scalar/vector queries.
  Supported numerical dtypes are float32/64 and complex64/128, with normal dtype
  promotion. All tensors must share a device. Extreme queries may overflow.
- Autograd supports signal/query differentiation. Use real scalar objectives for
  complex outputs, for example `output.abs().square().mean()`.

To use the extension bases in any KAN stage:

```python
from kan_transforms import register_kan_bases
from kangnn import GraphKAN, ModelConfig, TransformConfig
from kangnn.data import toy_graph

register_kan_bases()
model = GraphKAN(ModelConfig(
    2, 8, 2,
    preprocessing=TransformConfig(kind="kan", basis="laplace"),
    decision=TransformConfig(kind="kan", basis="z_transform"),
))
assert model(toy_graph()).shape == (6, 2)
```

`laplace` uses `exp(-softplus(x)*t_k)`; `z_transform` uses
`(1+softplus(x))**(-k)`. Their fixed nodes are buffers and outer KAN coefficients
are trainable. Registration is explicit and idempotent for these same classes;
normal imports do not modify the graph framework registry.

## GPU and determinism

Modules and tensor operations support device moves. Install CUDA-enabled PyTorch
and move model, graph data, masks, preprocessing buffers and objective buffers to
the same device. Move the model before constructing a manual optimizer.

```python
import torch
from kangnn import GraphKAN, ModelConfig, TrainingConfig, fit, seed_everything
from kangnn.data import toy_graph

seed_everything(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphKAN(ModelConfig(2, 8, 2)).to(device)
data = toy_graph().to(device)
fit(model, data, config=TrainingConfig(epochs=3))
print("Device:", device)
```

Call `seed_everything` before constructing models. It seeds Python, NumPy and
PyTorch, uses one CPU thread and enables strict deterministic algorithms.
Repeatability is tested on the same CPU environment; bitwise identity across
versions/hardware is not guaranteed. Some CUDA graph operations may reject strict
mode. Use CPU or deliberately opt out with
`torch.use_deterministic_algorithms(False)` when necessary, accepting weaker
repeatability. CUDA execution remains unverified on the local CPU-only host;
conditional GPU parity/backward tests skip when hardware is unavailable.

## Saving models and configurations

`ModelConfig.to_dict()` serializes stage settings; `ModelConfig(**dictionary)`
reconstructs nested configs. Use ordinary JSON import/export for architecture
settings. `save_model` stores format version, config and model state;
`load_model` uses `weights_only=True` and reconstructs the model.

```python
from pathlib import Path
import torch
from kangnn import FeatureStandardizer, GraphKAN, ModelConfig, load_model, save_model
from kangnn.data import toy_graph

raw = toy_graph()
scaler = FeatureStandardizer(2).fit(raw.x)
data = raw.clone()
data.x = scaler(raw.x)
model = GraphKAN(ModelConfig(2, 8, 2)).eval()
output = Path("artifacts")
output.mkdir(exist_ok=True)
save_model(model, output / "model.pt")
torch.save(scaler.state_dict(), output / "scaler.pt")
restored = load_model(output / "model.pt", map_location="cpu").eval()
restored_scaler = FeatureStandardizer(2)
restored_scaler.load_state_dict(torch.load(output / "scaler.pt", weights_only=True))
fresh = raw.clone()
fresh.x = restored_scaler(raw.x)
with torch.no_grad():
    torch.testing.assert_close(model(data), restored(fresh), atol=0, rtol=0)
```

This is a persistence demonstration; fit the scaler on training rows in a real
experiment. Apply it once at inference and preserve feature order. Register custom
bases before loading and call `.eval()` afterward. Checkpoints omit optimizer,
RNG state, loss code and preprocessing state; save the scaler separately. They
are prediction checkpoints, not exact training-resume snapshots. Model dtype is
not stored in config; preserve float64 by explicit dtype-aware reconstruction.

## Command-line interface

```bash
kangnn --help
kangnn --operator sage --kind mlp --epochs 30
kangnn --operator gatv2 --kind kan --basis fourier --loss nll --epochs 30
kangnn --config KAN_framwork/examples/mixed.json --epochs 60 --save /tmp/mixed.pt
python KAN_framwork/examples/custom_extensions.py
python KAN_framwork/examples/skan_style.py
python mathematical_transforms/examples/numerical.py
python mathematical_transforms/examples/kan_integration.py
```

| Flag | Choices/default |
|---|---|
| `--operator` | `gcn` (default), `sage`, `gat`, `gatv2` |
| `--kind` | `kan` (default), `mlp` |
| `--basis` | `rbf` (default), `bspline`, `fourier`, `relu` |
| `--loss` | `cross_entropy` (default), `nll` |
| `--epochs` | 60 |
| `--seed` | 7 |
| `--config` | Optional ModelConfig JSON file |
| `--save` | Optional model checkpoint destination |

A config file replaces architecture flags; training/seed/save flags remain active.
The CLI uses the fixed offline node-classification dataset and requires
`in_channels=2`, `out_channels=2`, `task="node"`. Its evaluation is on training
data. Use Python for real datasets, other objectives and graph/link tasks.
Optional/custom bases require registration in a Python entry point before config
construction; they are not added to the built-in CLI `--basis` choices.

## Tests and GitHub Actions

```bash
pytest KAN_framwork/tests mathematical_transforms/tests -q
pytest KAN_framwork/tests --cov=kangnn --cov-fail-under=90
pytest mathematical_transforms/tests --cov=kan_transforms --cov-fail-under=90
ruff check KAN_framwork mathematical_transforms
```

Tests use offline synthetic fixtures. They cover basis values/gradients, manual
KAN equations, custom registration, stage isolation, every backbone, numerical
PyG baseline parity, multihead attention, weights/attributes, permutation
properties, graph batching, all tasks/losses, deterministic training, masking,
serialization, CLI behavior, numerical transforms and executable guide examples.
Conditional CUDA checks run only with available hardware.

Recorded local verification after SKAN additions: **214 passed, 3 CUDA checks
skipped**, with **95.34% combined line coverage**. This is a dated local result,
not a live CI badge. GitHub-hosted execution has not been verified in this session.

| Workflow | Actions |
|---|---|
| [KAN-GNN CI](.github/workflows/ci.yml) | Python 3.10/3.12 CPU tests, Ruff, 90% coverage gate, examples, wheel/sdist build, twine check, artifact upload |
| [Transform CI](.github/workflows/transforms-ci.yml) | Both-package integration, transform coverage gate, examples, framework regression tests, package build and artifact upload |

Workflows trigger on pushes, PRs and manual dispatch. Their CPU backend pair is
pinned to PyTorch 2.5.1/PyG 2.6.1; dependencies are not a complete transitive lock.
Artifacts are delivered through GitHub Actions. No PyPI publication or production
deployment is configured.

## Repository layout and documentation

```text
KAN_framwork/
  src/kangnn/                 Graph model, layers, bases, transforms, training, data, CLI
  examples/                  Mixed JSON, custom extensions, SKAN-style example
  tests/                     Deterministic numerical/integration/documentation checks
  docs/                      User guide, implementation, source review, future steps
  pyproject.toml             Independent graph-package installation
mathematical_transforms/
  src/kan_transforms/         Numerical transforms and optional KAN bases
  examples/                  Signal transforms and graph integration
  tests/                     Numerical, autograd, graph and conditional CUDA checks
  docs/                      Implementation and future steps
  pyproject.toml             Independent transform-package installation
.github/workflows/           Test/build/artifact automation
```

| Document | Use |
|---|---|
| [Detailed user guide](KAN_framwork/docs/USER_GUIDE.md) | Step-by-step usage, runnable task examples, complete config tables and troubleshooting |
| [Graph package README](KAN_framwork/README.md) | Package-specific installation, integration and extension contracts |
| [Graph implementation log](KAN_framwork/docs/IMPLEMENTATION.md) | Equations, design choices, paper mapping, work log and verification |
| [GNN-SKAN source review](KAN_framwork/docs/SOURCE_REVIEW_GNN_SKAN.md) | Pinned source revision, resulting improvements and deliberate differences |
| [Graph roadmap](KAN_framwork/docs/FUTURE_STEPS.md) | Paper reproduction, data pipelines, performance and UI work |
| [Transform README](mathematical_transforms/README.md) | Numerical conventions, shape/device contracts and KAN integration |
| [Transform implementation log](mathematical_transforms/docs/IMPLEMENTATION.md) | Design and verification details |
| [Transform roadmap](mathematical_transforms/docs/FUTURE_STEPS.md) | Additional transforms, inverses, trainable kernels and GPU CI |

## Research references and limitations

The original KAN, Fourier KA-GNN, GNN-SKAN and residual gated ReLU-KAN papers
provided for this project motivate its modular design. Their mapping to this code
is recorded in the implementation log. Source-informed SKAN improvements refer
to [Lirain21/GNN-SKAN](https://github.com/Lirain21/GNN-SKAN) at commit
`5a352a170dadb8a0fa3f66eb92b6be0975ef15df`. No upstream source files were vendored;
the source review records the independent implementation and reuse boundary.

The current framework supports dense numerical features on homogeneous COO graphs.
It does not yet supply SparseTensor/bipartite/heterogeneous operators, categorical
molecular featurization, scaffold splitting, automatic negative sampling, ranking
metrics, adaptive spline grids, grid-transfer policies, symbolic fitting/pruning,
a graphical UI or numerical inverse transforms. Exact paper-model and benchmark
reproductions are not claimed. KAN basis activations can consume substantial
memory on wide layers or large edge batches.

Use ordinary PyTorch loops for advanced training policies, loaders and metrics.
Model activation placement and attention head averaging follow the current
implementation; not every option of PyG's original operators is exposed. Select
a license before public package distribution; none is assumed for this project.
