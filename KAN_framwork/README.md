# KAN-GNN framework

A Python configuration API and JSON CLI for building PyTorch Geometric graph
networks with independently selected MLP or KAN transforms. Supports **GCN,
GraphSAGE, GAT, GATv2**, node classification/regression, graph
classification/regression, and learned link scoring.

The code uses the existing directory spelling `KAN_framwork`; the import name is
`kangnn`. Python 3.10+ is required. This is a research framework with an offline
example, not a graphical application or a claim of reproduced paper benchmarks.

For a step-by-step tutorial with independently runnable examples, see the
[detailed user guide](docs/USER_GUIDE.md).

## Install and run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './KAN_framwork[test]'
kangnn --operator gatv2 --kind kan --basis rbf --epochs 60
kangnn --config KAN_framwork/examples/mixed.json --epochs 60 --save /tmp/mixed.pt
python KAN_framwork/examples/custom_extensions.py
pytest KAN_framwork/tests --cov=kangnn --cov-fail-under=90
ruff check KAN_framwork
```

The CLI uses a fixed six-node dataset with no downloads. Its reported evaluation
is on that same dataset and is a smoke check, not a held-out performance estimate.
Use the Python interface with your own PyG `Data`, `Batch`, loaders, and splits
for real experiments. The CLI supports CE/NLL for its two-class example; all
other objectives are available through Python.

## Build a model

```python
from kangnn import (
    GraphKAN, ModelConfig, LayerConfig, TransformConfig,
    TrainingConfig, FeatureStandardizer, node_split, fit, evaluate,
    seed_everything, save_model, load_model,
)
from kangnn.data import toy_graph

seed_everything(7)  # before constructing the model
mlp = TransformConfig(kind="mlp", hidden=(16,))
kan = TransformConfig(kind="kan", basis="bspline",
                      basis_kwargs={"num_basis": 8, "degree": 3, "domain": (-2, 2)})
rbf = TransformConfig(kind="kan", basis="rbf", basis_kwargs={"num_basis": 8})
model = GraphKAN(ModelConfig(
    in_channels=2, hidden_channels=8, out_channels=2,
    preprocessing=kan,
    layers=[
        LayerConfig(operator="gcn", message=mlp, update=rbf),
        LayerConfig(operator="gatv2", heads=2, message=kan,
                    attention=TransformConfig(kind="mlp", bias=False), update=mlp),
    ],
    readout=rbf,
    decision=mlp,
    task="node",
))
data = toy_graph()
train, validation, test = node_split(data.num_nodes, train=0.5, validation=0.25, seed=7)
scaler = FeatureStandardizer(data.num_features).fit(data.x, train)
data.x = scaler(data.x)
history = fit(model, data, "cross_entropy", TrainingConfig(epochs=50), mask=train)
print(evaluate(model, data, mask=validation))
save_model(model, "/tmp/model.pt")
restored = load_model("/tmp/model.pt")
restored.eval()
```

This example illustrates the API on very small splits; it is not a statistically
meaningful evaluation. In transductive node prediction, training can use the full
unlabeled graph topology while the mask restricts supervised labels. In inductive
experiments, construct training subgraphs separately.

## Configuration choices

| Stage | Configuration | Behavior |
|---|---|---|
| Feature preparation | `FeatureStandardizer.fit(x, train_mask)` | Training-only statistics; separate state dict |
| Node embedding | `ModelConfig.preprocessing` | Input features to hidden width |
| Message projection | `LayerConfig.message` | Source values; also SAGE root and GATv2 target projections |
| Edge embedding | `LayerConfig.edge`, `edge_dim` | Optional edge features added to values |
| Attention scoring | `LayerConfig.attention` | Independent scorer per attention head |
| Aggregation | `operator` | GCN normalized sum, SAGE mean, or attention-weighted sum |
| Node update | `LayerConfig.update` | Transform aggregated hidden representations |
| Graph pooling | `pooling="mean"/"sum"/"max"` | Permutation-invariant graph reduction |
| Readout transform | `ModelConfig.readout` | Transform pooled graphs or node embeddings |
| Prediction/decision | `ModelConfig.decision` | Output logits, regression values, or pair scores |

Each trainable transform accepts `kind="mlp"` or `kind="kan"`. `identity` is also
available when input and output widths match. A zero-hidden-layer MLP is a linear
map; set `hidden=(32, 16)` for a deeper MLP. KAN hidden widths stack KAN layers
without inserting fixed MLP activations. Graph layers are followed by ReLU and
configurable model dropout. Updates and readout default to identity so baseline
operators are readily expressible.

Aggregation/pooling are fixed mathematical reductions, not feature MLPs. Choose
KAN on messages, updates, and readout to learn transformations around those
reductions while preserving permutation symmetry. Structural preprocessing
(splits, graph construction, sampling) also remains algorithmic.

| Basis | Name | Options and definition |
|---|---|---|
| B-spline | `bspline` | `num_basis=8`, `degree=3`, `domain=(-1,1)`; clamped open knots |
| Fourier series | `fourier` | Odd `num_basis=9`, `period=2*pi`; constant, sine and cosine harmonics |
| ReLU ramps | `relu` | `num_basis=8`, `domain=(-1,1)`; learnable shifted, finite-width ramps |
| Gaussian RBF | `rbf` | `num_basis=8`, `domain=(-1,1)`; learnable centers and positive widths |

B-splines clamp outside their domain. Choose a sensible domain after inspecting
training-only feature statistics; the optional SiLU residual also carries signal
outside spline support. Fourier layers are periodic; RBF widths use softplus.
Fourier's default is 9 bases; the other built-ins default to 8. Basis choices are
independent of supervised loss choices.

## SKAN-style feature layers

`skan_transform()` configures normalized RBF features, a shared grid-spacing
bandwidth and truncated-normal coefficient initialization. Use it for `update`
and optionally `decision` to explore SKAN-style graph models. RBF centers and
widths can be frozen independently. Existing defaults remain unchanged.
See the [user guide](docs/USER_GUIDE.md#skan-style-settings-from-the-reference-implementation)
and [source review](docs/SOURCE_REVIEW_GNN_SKAN.md) for options and differences
from the reference. Run `python KAN_framwork/examples/skan_style.py` for an example.

## Graph and link tasks

For graph tasks, use `task="graph"`, `pooling="mean"` and a PyG `Batch` with graph
labels shaped `[num_graphs]` for CE, or `[num_graphs, num_targets]` for regression.
Unbatched `Data` is treated as one graph.

For link tasks:

```python
import torch
from kangnn import GraphKAN, ModelConfig, TransformConfig, TrainingConfig, fit
from kangnn.data import toy_graph

data = toy_graph()
data.edge_label_index = torch.tensor([[0, 0, 3], [1, 4, 5]])
data.y = torch.tensor([[1.], [0.], [1.]])
model = GraphKAN(ModelConfig(2, 8, 1, task="link",
                            decision=TransformConfig(kind="kan", basis="relu")))
fit(model, data, "bce_logits", TrainingConfig(epochs=30))
model.eval()
with torch.no_grad():
    probabilities = model(data).sigmoid()
```

Query edges are separate from `edge_index`, which controls message passing. You
must split positive edges, exclude validation/test positives from training
message edges, and construct negative examples correctly for your problem. The
framework does not perform negative sampling automatically. The concatenation
pair decoder is ordered; for undirected predictions, average scores for both
orders or supply a symmetric decoder in a subclass. See the future-work document
for dataset-specific bipartite recommendation support.

## Losses and extensions

Built-in supervised losses: `cross_entropy`, `bce_logits`, `mse`, `l1`,
`smooth_l1`, and `nll`. CE/NLL targets are integer class IDs `[N]`. BCE targets
are float tensors with the same shape as logits; for binary tasks use
`out_channels=1` and `[N,1]`. Regression targets match prediction shapes.
`fit(..., loss="nll")` explicitly applies log-softmax. `make_loss("nll")` returns
raw PyTorch NLLLoss, so direct callers must provide log probabilities.

Pass kwargs or custom functions without editing framework code:

```python
from torch import nn
from kangnn import make_loss, register_loss
weighted_ce = make_loss("cross_entropy", label_smoothing=0.1)
# fit(model, data, loss=weighted_ce)

def objective(predictions, targets):
    return nn.functional.cross_entropy(predictions, targets)
# fit(model, data, loss=objective)

register_loss("my_ce", lambda smoothing=0.05:
              nn.CrossEntropyLoss(label_smoothing=smoothing))
custom_loss = make_loss("my_ce", smoothing=0.1)
```

Custom basis contract:

```python
import torch
from torch import nn
from kangnn import register_basis, TransformConfig

class PolynomialBasis(nn.Module):
    def __init__(self, num_basis=4):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("powers", torch.arange(num_basis))

    def forward(self, x):
        return x.unsqueeze(-1) ** self.powers

register_basis("polynomial", PolynomialBasis)
custom = TransformConfig(kind="kan", basis="polynomial",
                         basis_kwargs={"num_basis": 4})
```

A basis is an `nn.Module` mapping `[..., input_features]` to
`[..., input_features, num_basis]`. It may contain trainable parameters and
buffers, so autograd, device moves, and state dictionaries work normally. The
KAN learns separate coefficients per input-output edge. See
[`examples/custom_extensions.py`](examples/custom_extensions.py) for a runnable
custom basis/loss example. Register custom names before JSON construction or
checkpoint loading. JSON contains configuration only; it never evaluates Python.
For completely different KAN equations, subclass `KANLinear`/`GraphKAN` in Python
and compose your own modules; the built-in config factory registers basis
families, not arbitrary full-layer classes.

## Data and persistence contracts

- `x`: float `[num_nodes, in_channels]`; `edge_index`: int64 `[2, num_edges]`,
  source row then target row. Use contiguous node IDs. Features and indices must
  share a device. Supply finite feature values.
- `edge_weight`: finite nonnegative `[num_edges]`, GCN only.
- `edge_attr`: float `[num_edges, edge_dim]`. Set matching `edge_dim` in every
  model layer when supplying edge features. Move the entire `Data` to the device.
- SAGE does not add self loops; its root transform represents the central node.
  GCN adds missing loops; attention removes old loops and adds one loop per node.
  Newly created attention loops receive zero edge features. For GCN edge
  features, set `self_loops=False` and prepare aligned loops/features yourself.
- Attention heads each produce `hidden_channels`; the outputs are averaged.
  Source/target value projections and attention scorers can use different
  transform kinds. GATv2 has independent source/target projections.
- `save_model` stores versioned config and model weights. `load_model` uses
  `weights_only=True`. Save the feature standardizer separately with its
  `state_dict`; model checkpoints do not include optimizer, scaler, data,
  registries, or RNG state. Call `.eval()` after loading for inference.
- `fit` is a full-batch convenience trainer. To use neighbor sampling or large
  datasets, place the model inside a normal PyTorch/PyG training loop. Trainable
  parameters of custom loss modules are not optimized by this helper.

## Documentation and verification

- [Implementation details and work log](docs/IMPLEMENTATION.md): equations,
  paper mapping, design decisions, files, verification, and limits.
- [Future steps](docs/FUTURE_STEPS.md): prioritized research and engineering work.
- [GitHub Actions](../.github/workflows/ci.yml): deterministic CPU test matrix,
  coverage gate, examples, package build, and downloadable artifacts.

Repeatability means repeated runs in the same CPU software/hardware environment.
We do not promise bitwise identity across PyTorch releases, devices, or CUDA
scatter implementations. Strict deterministic mode can reject unsupported GPU
operations. Automated tests use small offline fixtures and do not download data.
