# User guide: building and training KAN graph networks

This guide describes the code currently implemented in this repository. Every
Python code block is a complete example that can run independently after
installation. Examples use small offline graphs so you can learn the API without
downloading datasets. Their metrics illustrate execution, not benchmark quality.

## Contents

1. [Installation and first run](#1-installation-and-first-run)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Preparing graph data](#3-preparing-graph-data)
4. [Complete node-classification example](#4-complete-node-classification-example)
5. [Choosing MLP or KAN independently](#5-choosing-mlp-or-kan-independently)
6. [Choosing a graph operator](#6-choosing-a-graph-operator)
7. [Choosing a KAN basis](#7-choosing-a-kan-basis)
8. [Losses and target shapes](#8-losses-and-target-shapes)
9. [Node regression](#9-node-regression)
10. [Graph classification and minibatches](#10-graph-classification-and-minibatches)
11. [Link prediction without held-out-edge leakage](#11-link-prediction-without-held-out-edge-leakage)
12. [Edge attributes and weights](#12-edge-attributes-and-weights)
13. [GPU use and reproducibility](#13-gpu-use-and-reproducibility)
14. [Custom losses](#14-custom-losses)
15. [Custom KAN bases](#15-custom-kan-bases)
16. [Z and Laplace extensions](#16-z-and-laplace-extensions)
17. [Saving and loading](#17-saving-and-loading)
18. [JSON configuration and CLI](#18-json-configuration-and-cli)
19. [Configuration reference](#19-configuration-reference)
20. [Troubleshooting and experiment checklist](#20-troubleshooting-and-experiment-checklist)

## 1. Installation and first run

Run shell commands from the repository root, the directory containing
`KAN_framwork/` and `mathematical_transforms/`. Python 3.10 or newer is required.
If `.venv` already exists, activate it rather than recreating it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './KAN_framwork[test]'
python -m pip install -e './mathematical_transforms[test]'
```

The final command is needed for the optional Z/Laplace examples. For GPU use,
install a CUDA-enabled PyTorch build appropriate to your system instead of the
CPU build above. This guide does not prescribe a CUDA version for your machine.

Check the installation and run a complete offline demo:

```bash
python -c "import torch, torch_geometric, kangnn; print(torch.__version__); print(torch_geometric.__version__)"
kangnn --operator gatv2 --kind kan --basis rbf --epochs 60
```

The CLI prints JSON with initial loss, final training loss, evaluation loss,
accuracy and seed. Its evaluation uses the same six-node graph it trains on.
Use the split-based Python examples below for train/validation/test workflows.

## 2. How the pipeline works

```text
Data.x
  -> learned preprocessing/embedding
  -> graph layer 1: message -> aggregation -> update
  -> ReLU and model dropout
  -> further graph layers, each followed by ReLU/dropout
  -> graph pooling, for graph tasks only
  -> readout transform
  -> concatenate query endpoint embeddings, for link tasks only
  -> decision/output head
  -> raw logits or regression values
```

Each learned stage has a `TransformConfig`. Select `kind="mlp"`, `kind="kan"`,
or, where dimensions allow, `kind="identity"`. A KAN adds basis functions and
learned coefficients on feature connections. This feature expansion is separate
from graph edges, which define which nodes exchange messages.

The graph operator determines the invariant neighborhood reduction: normalized
sum for GCN, mean plus a root path for GraphSAGE, or attention-weighted sum for
GAT/GATv2. Reduction, graph construction and dataset splitting are algorithms;
they are not replaced by an MLP or KAN.

`GraphKAN` returns raw predictions. Apply softmax/sigmoid only when interpreting
classification predictions, not before CE/BCEWithLogits training. Node and graph
regression outputs have no final activation. `encode(data)` returns node
embeddings before pooling, readout and decision.

## 3. Preparing graph data

Use a PyTorch Geometric `Data` object. Let N be nodes, E edges, F input features,
C classes and T regression targets.

| Field | Shape and dtype | Meaning |
|---|---|---|
| `x` | `[N,F]`, floating point | Node features, normally float32 |
| `edge_index` | `[2,E]`, `torch.long` | Row 0 sends to row 1 |
| `y` for node CE | `[N]`, `torch.long` | Class IDs from 0 to C-1 |
| `y` for graph CE | `[num_graphs]`, `torch.long` after batching | One class per graph |
| `y` for regression | `[num_predictions,T]`, float | Must match the prediction shape |
| `y` for binary BCE | `[num_predictions,1]`, float | Values 0 or 1 |
| `edge_attr` | `[E,edge_dim]`, float | Optional edge features |
| `edge_weight` | `[E]`, nonnegative float | Optional GCN weights |
| `edge_label_index` | `[2,num_queries]`, `torch.long` | Link pairs to score |
| `batch` | `[N]`, `torch.long` | Graph membership, created by PyG batching |

External node IDs must be mapped to contiguous integers 0 through N-1. An
undirected relationship normally requires both edge directions. Avoid accidental
duplicate edges: they contribute multiple messages. Features should be finite.
Do not store text directly in `x`; supply numerical features or external
embeddings. This framework does not generate NLP embeddings.

```python
import torch
from torch_geometric.data import Data

x = torch.tensor([[1., 0.], [0.5, 0.2], [-1., 0.], [-0.5, 0.1]])
# Undirected pairs 0--1 and 2--3, represented in both directions.
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)
assert data.num_nodes == 4 and data.num_features == 2
print(data)
```

For real datasets, load features/edges/labels with your normal data tooling, then
construct the same object. Splits and labels depend on the task. Do not apply
`node_split` to a collection of graphs or to link pairs and assume it creates the
correct graph/link evaluation protocol.

## 4. Complete node-classification example

This example fits normalization on training rows, trains only on training labels,
reports validation results and evaluates the test split after training. For real
experiments, choose hyperparameters using validation data and reserve test data
for the final report. The tiny fixture's split contains too few examples for a
meaningful performance estimate.

```python
import torch
from kangnn import (
    FeatureStandardizer, GraphKAN, LayerConfig, ModelConfig, TrainingConfig,
    TransformConfig, evaluate, fit, node_split, seed_everything,
)
from kangnn.data import toy_graph

seed_everything(7)  # Call before model construction.
data = toy_graph()
train, validation, test = node_split(
    data.num_nodes, train=0.5, validation=0.25, seed=7,
)

# Learn statistics only from training features. Never refit on validation/test.
scaler = FeatureStandardizer(data.num_features).fit(data.x, train)
data.x = scaler(data.x)

kan = TransformConfig(kind="kan", basis="rbf", basis_kwargs={"num_basis": 8})
model = GraphKAN(ModelConfig(
    in_channels=data.num_features,
    hidden_channels=8,
    out_channels=2,
    preprocessing=kan,
    layers=[LayerConfig(operator="gcn", message=kan, update=kan)],
    decision=TransformConfig(kind="mlp"),
    task="node",
))
history = fit(
    model, data, loss="cross_entropy",
    config=TrainingConfig(epochs=60, lr=0.01, weight_decay=0.0, kan_l1=1e-4),
    mask=train,
)
print("First/last training objective:", history[0], history[-1])
print("Validation:", evaluate(model, data, mask=validation))
print("Test:", evaluate(model, data, mask=test))

model.eval()
with torch.no_grad():
    logits = model(data)                  # [6,2]
    probabilities = logits.softmax(-1)    # [6,2]
    predicted_classes = logits.argmax(-1) # [6]
    embeddings = model.encode(data)      # [6,8]
assert embeddings.shape == (6, 8)
print("Predicted classes:", predicted_classes.tolist())
print("Class probabilities:", probabilities)
```

`fit` returns one scalar objective value per epoch. It uses full-batch Adam and
constructs a new optimizer on every call. Repeated calls do not resume Adam's
momentum. The training objective includes configured coefficient regularization;
`evaluate` reports supervised loss without that penalty.

A label mask restricts supervision, not topology. Using the whole unlabeled graph
is a transductive setup. An inductive setup needs separately constructed training
subgraphs and an appropriate evaluation graph.

## 5. Choosing MLP or KAN independently

This example uses a different transform in each stage. Set a stage to an MLP to
replace only that stage; the other choices remain independent.

```python
import torch
from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig
from kangnn.data import toy_graph

mlp = TransformConfig(kind="mlp", hidden=(16,))
spline = TransformConfig(kind="kan", basis="bspline",
                         basis_kwargs={"num_basis": 8, "degree": 3, "domain": (-3., 3.)})
rbf = TransformConfig(kind="kan", basis="rbf")
fourier = TransformConfig(kind="kan", basis="fourier",
                          basis_kwargs={"num_basis": 9})
model = GraphKAN(ModelConfig(
    2, 8, 2,
    preprocessing=spline,
    layers=[
        LayerConfig(operator="sage", message=mlp, update=rbf),
        LayerConfig(operator="gatv2", heads=2, message=fourier,
                    attention=TransformConfig(kind="mlp", bias=False),
                    update=mlp, residual=True),
    ],
    readout=rbf,
    decision=mlp,
))
model.eval()
with torch.no_grad():
    assert model(toy_graph()).shape == (6, 2)
```

A default MLP has no hidden widths and is a linear map. Use `hidden=(32,16)` for
additional layers. MLP hidden layers use ReLU; stacked KAN layers do not insert
those MLP activations. The enclosing graph model still applies ReLU after every
graph layer.

Identity requires equal input/output widths and no hidden layers. It is useful
for `update` or `readout`. Preprocessing identity requires
`in_channels == hidden_channels`. Decision identity requires the input width to
match the output width; link decisions receive twice the hidden width.

## 6. Choosing a graph operator

| Operator string | Aggregation | Heads | Notes |
|---|---|---|---|
| `gcn` | Degree-normalized weighted sum | Must be 1 | Optional nonnegative edge weights |
| `sage` | Neighbor mean plus independent root transform | Must be 1 | No automatic self loops |
| `gat` | Incoming-edge attention | 1 or more | Shared value projection for source/target |
| `gatv2` | Dynamic attention after vector nonlinearity | 1 or more | Independent source/target projections |

```python
import torch
from kangnn import GraphKAN, LayerConfig, ModelConfig
from kangnn.data import toy_graph

for operator in ("gcn", "sage", "gat", "gatv2"):
    layer = LayerConfig(operator=operator, heads=2 if operator in {"gat", "gatv2"} else 1)
    model = GraphKAN(ModelConfig(2, 8, 2, layers=[layer]))
    model.eval()
    with torch.no_grad():
        assert model(toy_graph()).shape == (6, 2)
    print(operator, "works")
```

Heads are averaged, so increasing head count does not multiply the output width.
`LayerConfig.dropout` affects attention weights during training;
`ModelConfig.dropout` affects node activations after graph layers.
`residual=True` adds a learned projected skip; it is separate from the KAN SiLU
base branch and from the gated architecture described in the supplied papers.

For lower-level composition, `GraphLayer(in_channels,out_channels,config)` accepts
`x`, `edge_index`, optional `edge_weight` and `edge_attr`. It subclasses PyG
`MessagePassing`. It returns graph-layer outputs without the enclosing model's
ReLU/dropout. SparseTensor, heterogeneous and bipartite tuple inputs are not part
of its current public contract.

## 7. Choosing a KAN basis

| Name | Typical kwargs | Important behavior |
|---|---|---|
| `bspline` | `num_basis=8, degree=3, domain=(-1,1)` | Requires count > degree; clamps outside domain |
| `fourier` | `num_basis=9, period=6.283185307179586` | Odd count >=3; periodic harmonics |
| `relu` | `num_basis=8, domain=(-1,1)` | Learnable centers and positive widths for ReLU ramps |
| `rbf` | `num_basis=8, domain=(-1,1)` | Gaussian kernels with learned centers/widths |

```python
import torch
from kangnn import TransformConfig, make_transform

configs = [
    TransformConfig(kind="kan", basis="bspline",
                    basis_kwargs={"num_basis": 10, "degree": 3, "domain": (-3., 3.)}),
    TransformConfig(kind="kan", basis="fourier",
                    basis_kwargs={"num_basis": 9, "period": 6.283185307179586}),
    TransformConfig(kind="kan", basis="relu", basis_kwargs={"num_basis": 8}),
    TransformConfig(kind="kan", basis="rbf", basis_kwargs={"num_basis": 8}),
]
for config in configs:
    transform = make_transform(2, 4, config)
    assert transform(torch.tensor([[0.2, -0.4]])).shape == (1, 4)
```

`base_residual=True` adds a learned linear mapping of SiLU(input) alongside the
basis expansion. Set it to `False` for a pure basis expansion. `bias` controls
transform output bias; graph layers also have their own output bias.

Standardization does not guarantee values stay in `[-1,1]`. Inspect training-only
feature statistics and choose the spline interval accordingly. Hidden-layer
feature distributions can differ from input distributions; a narrow spline
interval can saturate. Fourier/RBF/ReLU `domain` settings have different meanings:
Fourier uses `period`, while RBF/ReLU domains initialize their basis locations.
They do not implement spline-style clipping.

### SKAN-style settings from the reference implementation

The [GNN-SKAN source review](SOURCE_REVIEW_GNN_SKAN.md) motivated optional
LayerNorm on basis inputs, independent RBF freezing, explicit positive bandwidths
and coefficient initialization controls. Existing configurations keep their old
defaults. A convenience factory exposes these settings together:

```python
from kangnn import (GraphKAN, LayerConfig, ModelConfig, TransformConfig, skan_transform)
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

Using identity messages places the nonlinear KAN transform after normalized
aggregation. `skan_transform` normalizes only the basis branch; the optional SiLU
base branch receives the raw feature input. It initializes a shared positive RBF
width from grid spacing and coefficients with truncated-normal std 0.1. Its
center/width training defaults are True; set either False for controlled ablations.

For individual options, use `TransformConfig(basis_normalization="layernorm",
coefficient_init="truncated_normal", coefficient_init_std=0.1, kind="kan",
basis="rbf", basis_kwargs={...})`. RBF kwargs include `width` (Gaussian standard
deviation), `shared_width`, `trainable_centers`, and `trainable_widths`.
LayerNorm across a single feature has a constant normalized result; retain the
base branch or use a wider representation if you need input dependence there.

This is a feature-layer preset, not exact reproduction of the upstream molecular
model. It does not add molecular categorical embeddings, scaffold splits,
jumping-knowledge readout or periodic grid adaptation. Run
`python KAN_framwork/examples/skan_style.py` for a complete graph training example.

## 8. Losses and target shapes

| Loss name | Output shape | Target shape/dtype | Prediction interpretation |
|---|---|---|---|
| `cross_entropy` | `[P,C]` | `[P]`, long class IDs | `softmax(-1)` or `argmax(-1)` |
| `nll` via `fit/evaluate` string | `[P,C]` raw logits | `[P]`, long | Helpers apply log-softmax |
| `bce_logits` | `[P,1]` binary or `[P,T]` multilabel | Same shape, float | `sigmoid()` |
| `mse` | `[P,T]` | Same shape, float | Raw regression values |
| `l1` | `[P,T]` | Same shape, float | Raw regression values |
| `smooth_l1` | `[P,T]` | Same shape, float | Raw regression values |

P is nodes, graphs or link queries depending on the task. In particular, a
single regression target is `[P,1]`, not `[P]`. Matching shapes avoids unintended
broadcasting in PyTorch regression losses.

Use `make_loss(name, **kwargs)` for options such as label smoothing, class
weights or SmoothL1 beta, then pass the returned callable to `fit`. For a
weighted CE module, move its class-weight buffer to the model's device.

`make_loss("nll")` directly returns PyTorch NLLLoss and expects log probabilities.
Only the named `loss="nll"` trainer/evaluator path automatically applies
log-softmax to model logits. A custom callable must do its own conversion.

For binary tasks, the built-in evaluator returns loss but not binary accuracy.
Compute `(logits.sigmoid() >= 0.5)` and compare against float targets explicitly.

## 9. Node regression

```python
import torch
from kangnn import (GraphKAN, ModelConfig, TrainingConfig, TransformConfig,
                    evaluate, fit, seed_everything)
from kangnn.data import toy_graph

seed_everything(7)
data = toy_graph()
data.y = (2 * data.x[:, :1] - data.x[:, 1:2]).float()  # [6,1]
model = GraphKAN(ModelConfig(
    2, 8, 1, task="node",
    preprocessing=TransformConfig(kind="kan", basis="rbf"),
))
fit(model, data, loss="mse", config=TrainingConfig(epochs=40))
print(evaluate(model, data, loss="mse"))
model.eval()
with torch.no_grad():
    assert model(data).shape == data.y.shape
```

This snippet trains/evaluates on the fixture to demonstrate regression shape
handling. In an experiment, use training, validation and test masks as in section
4, and pass `loss="mse"` to both training and evaluation.

## 10. Graph classification and minibatches

Each graph has its own node/edge tensors and one graph label. PyG's loader
combines disconnected graphs into a `Batch`, adjusts node indices, and supplies
`batch`. Set `task="graph"` to pool before readout/decision.

```python
import torch
from torch_geometric.loader import DataLoader
from kangnn import (GraphKAN, ModelConfig, TransformConfig, evaluate, seed_everything)
from kangnn.data import toy_graph

seed_everything(7)
graphs = []
for label in (0, 1, 0, 1, 0, 1):
    graph = toy_graph()
    graph.x = graph.x + float(label)
    graph.y = torch.tensor([label], dtype=torch.long) # One label per graph.
    graphs.append(graph)
train_graphs, validation_graphs, test_graphs = graphs[:4], graphs[4:5], graphs[5:]
loader = DataLoader(train_graphs, batch_size=2, shuffle=False)
model = GraphKAN(ModelConfig(
    2, 8, 2, task="graph", pooling="mean",
    decision=TransformConfig(kind="kan", basis="rbf"),
))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    model.train()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch) # [number of graphs in batch,2]
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
validation = next(iter(DataLoader(validation_graphs, batch_size=1)))
test = next(iter(DataLoader(test_graphs, batch_size=1)))
print("Validation:", evaluate(model, validation))
print("Test:", evaluate(model, test))
```

Use `pooling="mean"`, `"sum"` or `"max"`. A single unbatched `Data` is treated
as one graph. For graph regression, choose `out_channels=T`, float per-graph
`y` shaped `[1,T]` before batching, and a regression objective.

The helper `fit` takes one `Data/Batch`, not a loader. Use a manual loop as above
for minibatches. It also allows normal PyTorch schedulers and gradient clipping.
For GPU minibatches, move each batch to the model device in the loop. For real
graph datasets, split graphs before computing normalization statistics and fit
statistics using training graphs only.

## 11. Link prediction without held-out-edge leakage

`edge_index` is the encoder's message graph. `edge_label_index` lists candidate
pairs scored by the decoder. They serve different purposes. Binary link labels
must be float `[number_of_queries,1]` when `out_channels=1`.

The following fixture removes both directions of held-out undirected positives
from the encoder graph. Explicit negatives are absent from the complete known
positive set. This is an example split, not a general negative sampler.

```python
import torch
from torch_geometric.data import Data
from kangnn import (GraphKAN, ModelConfig, TrainingConfig, TransformConfig,
                    evaluate, fit, seed_everything)
from kangnn.data import toy_graph

seed_everything(7)
x = toy_graph().x
train_positive = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
# Held out positives: (0,2) for validation and (3,5) for test.
message_edges = torch.cat([train_positive, train_positive.flip(0)], dim=1)
train_negative = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 4]], dtype=torch.long)
train = Data(
    x=x, edge_index=message_edges,
    edge_label_index=torch.cat([train_positive, train_negative], dim=1),
    y=torch.tensor([[1.]] * 4 + [[0.]] * 4),
)
validation = Data(
    x=x, edge_index=message_edges,
    edge_label_index=torch.tensor([[0, 0], [2, 5]], dtype=torch.long),
    y=torch.tensor([[1.], [0.]]),
)
test = Data(
    x=x, edge_index=message_edges,
    edge_label_index=torch.tensor([[3, 2], [5, 3]], dtype=torch.long),
    y=torch.tensor([[1.], [0.]]),
)
model = GraphKAN(ModelConfig(
    2, 8, 1, task="link",
    decision=TransformConfig(kind="kan", basis="relu"),
))
fit(model, train, "bce_logits", TrainingConfig(epochs=40))
print("Validation:", evaluate(model, validation, "bce_logits"))
print("Test:", evaluate(model, test, "bce_logits"))
model.eval()
with torch.no_grad():
    probabilities = model(test).sigmoid()
    # Optional explicit query argument, instead of data.edge_label_index:
    reverse_logits = model(test, edge_label_index=test.edge_label_index.flip(0))
    symmetric_scores = (model(test) + reverse_logits) / 2
print("Test probabilities:", probabilities.flatten().tolist())
assert symmetric_scores.shape == (2, 1)
```

The default decoder concatenates ordered endpoint embeddings; it is directed.
For undirected scoring, the example averages both orders' logits before applying
sigmoid. Use a consistent policy during training/evaluation if symmetry is part
of your task. For bipartite recommendation, construct valid user/item candidates
and negatives yourself. The framework does not automatically perform negative
sampling, edge splitting, or ranking evaluation.

## 12. Edge attributes and weights

Configure `edge_dim` in every graph layer if `Data.edge_attr` is present; the
model forwards that same attribute tensor to each layer. Edge encoders may be
MLPs or KANs. In the current design, encoded edges add to source message values.

```python
import torch
from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig
from kangnn.data import toy_graph

# Attention with a two-feature edge vector.
data = toy_graph()
data.edge_attr = torch.ones(data.edge_index.size(1), 2)
edge_kan = TransformConfig(kind="kan", basis="rbf")
model = GraphKAN(ModelConfig(2, 8, 2, layers=[
    LayerConfig(operator="gatv2", heads=2, edge_dim=2, edge=edge_kan),
    LayerConfig(operator="sage", edge_dim=2, edge=edge_kan),
]))
assert model(data).shape == (6, 2)

# Weighted GCN: a separate graph without edge_attr.
weighted = toy_graph()
weighted.edge_weight = torch.linspace(0.5, 1.5, weighted.edge_index.size(1))
gcn = GraphKAN(ModelConfig(2, 8, 2, layers=[LayerConfig(operator="gcn")]))
assert gcn(weighted).shape == (6, 2)

# GCN with edge attributes requires explicit self-loop policy.
edge_gcn = GraphKAN(ModelConfig(2, 8, 2, layers=[
    LayerConfig(operator="gcn", edge_dim=2, edge=edge_kan, self_loops=False),
]))
assert edge_gcn(data).shape == (6, 2)
```

GCN edge attributes require `self_loops=False`. If you need self loops, add them
and aligned edge attributes/weights during graph preparation. GCN internally
normalizes weights. GAT/GATv2 replace existing loops with one loop per node;
new raw loop features are zero. An edge encoder with bias may map those zeros to
nonzero embeddings. GraphSAGE uses its root path and adds no loops.

`edge_weight` is GCN-only. A mixed model containing SAGE/GAT/GATv2 will reject a
global `edge_weight`; remove it or write an explicit per-layer forwarding policy.
A mixed edge-attribute model must also respect each layer's loop requirements.

## 13. GPU use and reproducibility

Install CUDA-enabled PyTorch first. Check availability with
`python -c "import torch; print(torch.cuda.is_available())"`. Move the model,
data, masks and loss buffers to the same device. Move the model before creating
a manual optimizer. Custom basis constants should be parameters or registered
buffers so `.to(device)` moves them too.

```python
import torch
from kangnn import (GraphKAN, ModelConfig, TrainingConfig, evaluate, fit,
                    node_split, seed_everything)
from kangnn.data import toy_graph

seed_everything(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = toy_graph()
train, validation, test = node_split(6, train=0.5, validation=0.25, seed=7)
data = data.to(device)
train, validation, test = [mask.to(device) for mask in (train, validation, test)]
model = GraphKAN(ModelConfig(2, 8, 2)).to(device)
fit(model, data, config=TrainingConfig(epochs=5), mask=train)
print("Device:", device)
print(evaluate(model, data, mask=validation))
```

`seed_everything` enables strict deterministic algorithms, seeds Python/NumPy/
PyTorch and uses one CPU thread. The tested repeatability guarantee is repeated
CPU runs in the same software/hardware environment. GPU scatter operations can
raise an error if no deterministic implementation is available. If your GPU run
fails specifically for that reason, either run on CPU or deliberately opt out
with `torch.use_deterministic_algorithms(False)` after seeding. Disabling it
relaxes repeatability; it is not necessary for every GPU workload.

GPU execution has not been validated on the current CPU-only machine. The
transform package includes conditional GPU tests; CPU CI does not run them.
The example above automatically uses CPU when CUDA is unavailable.

## 14. Custom losses

A custom loss receives predictions and targets and returns a differentiable
scalar tensor. It must handle the model's raw output convention. This example
shows a configured built-in, a registered module and a direct callable.

```python
import torch
from torch import nn
from kangnn import (GraphKAN, ModelConfig, TrainingConfig, evaluate, fit,
                    make_loss, register_loss, seed_everything)
from kangnn.data import toy_graph

class SmoothedCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        return nn.functional.cross_entropy(
            predictions, targets, label_smoothing=self.smoothing,
        )

def custom_objective(predictions, targets):
    ce = nn.functional.cross_entropy(predictions, targets)
    return ce + 1e-4 * predictions.square().mean()

register_loss("guide_smoothed_ce", SmoothedCE)
objectives = [
    make_loss("cross_entropy", weight=torch.tensor([1., 1.5]), label_smoothing=0.05),
    make_loss("guide_smoothed_ce", smoothing=0.1),
    custom_objective,
]
for objective in objectives:
    seed_everything(7)
    model = GraphKAN(ModelConfig(2, 8, 2))
    data = toy_graph()
    fit(model, data, loss=objective, config=TrainingConfig(epochs=3))
    print(evaluate(model, data, loss=objective))
```

Registration names must be new; repeating a normal `register_loss` call with the
same name raises an error. Register once at application startup. Pass the same
objective to evaluation if you want comparable supervised values.

`fit` optimizes model parameters only. If a loss module has trainable parameters,
use a manual optimizer that explicitly includes them. Class weights are buffers,
not trainable parameters, but must still be moved to the device. If you need a
loss involving graph structure, a closure can capture that structure or a custom
training loop can compute it directly.

## 15. Custom KAN bases

Your basis must be an `nn.Module`, expose integer `num_basis`, and map
`[...,F] -> [...,F,num_basis]`. Keep input dependence in differentiable PyTorch
operations. Register persistent constants as buffers and trainable quantities as
`nn.Parameter`. The outer KAN learns input-output coefficients automatically.

```python
import torch
from torch import nn
from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig, TransformConfig,
                    fit, register_basis, seed_everything)
from kangnn.data import toy_graph

class BoundedPolynomialBasis(nn.Module):
    def __init__(self, num_basis=4):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("powers", torch.arange(num_basis))

    def forward(self, x):
        return x.tanh().unsqueeze(-1).pow(self.powers)

register_basis("guide_polynomial", BoundedPolynomialBasis)
seed_everything(7)
custom = TransformConfig(kind="kan", basis="guide_polynomial",
                         basis_kwargs={"num_basis": 4})
model = GraphKAN(ModelConfig(
    2, 8, 2, preprocessing=custom, decision=custom,
    layers=[LayerConfig(operator="sage", message=custom, update=custom)],
))
history = fit(model, toy_graph(), config=TrainingConfig(epochs=10))
print("Custom basis objective:", history[-1])
```

The tanh here bounds polynomial inputs; it is a deliberate custom-basis choice.
To use another function, implement the same shape contract. For a completely
different full-layer KAN equation, compose your own PyTorch modules or subclass
`GraphKAN`; the config registry currently extends basis families, not arbitrary
complete layer classes. To load a checkpoint containing a custom basis in a new
process, import its class and register its name first.

## 16. Z and Laplace extensions

Install `mathematical_transforms` using section 1. It offers two distinct tools:
numerical transforms of sampled signals and transform-inspired real KAN bases.

### Use numerical transforms

```python
import torch
from kan_transforms import laplace_transform, z_transform

signal = torch.tensor([1., 2., 3.], dtype=torch.float64)
z = torch.tensor([1., 2., 1j], dtype=torch.complex128)
Z = z_transform(signal, z) # Finite sum: 1 + 2/z + 3/z**2
print("Z values:", Z)
torch.testing.assert_close(Z, torch.tensor([6., 2.75, -2.-2j], dtype=torch.complex128))

times = torch.linspace(0, 10, 10001, dtype=torch.float64)
samples = torch.exp(-2 * times)
s = torch.tensor([0., 1., 1.+1j], dtype=torch.complex128)
L = laplace_transform(samples, times, s)
print("Finite-interval Laplace:", L)
# Exact integral on [0,10], compared with numerical trapezoidal quadrature.
reference = (1 - torch.exp(-(s + 2) * times[-1])) / (s + 2)
torch.testing.assert_close(L, reference, atol=1e-6, rtol=1e-5)
```

Z queries must be nonzero. Laplace times must be nonnegative, real and strictly
increasing. Laplace integrates only over the given interval, not to infinity.
Both APIs support a sample `dim`, batch dimensions, scalar/vector queries and
real/complex float32/64 precision. Move all tensors to the same device. They do
not implement symbolic transforms or numerical inverses.

### Use the new functions inside KAN

```python
from kan_transforms import register_kan_bases
from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig,
                    TransformConfig, fit, seed_everything)
from kangnn.data import toy_graph

register_kan_bases() # Adds laplace and z_transform to kangnn's registry.
seed_everything(7)
laplace = TransformConfig(kind="kan", basis="laplace",
                          basis_kwargs={"num_basis": 8, "time_range": (0.1, 4.)})
z = TransformConfig(kind="kan", basis="z_transform", basis_kwargs={"num_basis": 8})
model = GraphKAN(ModelConfig(
    2, 8, 2, preprocessing=laplace, readout=laplace, decision=z,
    layers=[LayerConfig(operator="gatv2", message=z, update=laplace)],
))
print(fit(model, toy_graph(), config=TrainingConfig(epochs=10))[-1])
```

The bases evaluate `exp(-softplus(x)*t_k)` and `(1+softplus(x))**(-k)` per feature.
Their positive query maps avoid singularities for ordinary real features. These
are basis expansions, not automatic Laplace/Z transforms of your node sequence.
They work in attention/edge encoders as well as the shown stages. Register before
reconstructing configs or checkpoints. Unlike normal custom registration,
`register_kan_bases()` is explicitly idempotent for these same classes.

## 17. Saving and loading

Save the model config/weights and fitted preprocessing statistics separately.
Keep raw input features available so inference applies the scaler exactly once.

```python
from pathlib import Path
import torch
from kangnn import (FeatureStandardizer, GraphKAN, ModelConfig, TrainingConfig,
                    fit, load_model, node_split, save_model, seed_everything)
from kangnn.data import toy_graph

seed_everything(7)
raw = toy_graph()
train, _, _ = node_split(6, train=0.5, validation=0.25, seed=7)
scaler = FeatureStandardizer(2).fit(raw.x, train)
data = raw.clone()
data.x = scaler(raw.x)
model = GraphKAN(ModelConfig(2, 8, 2))
fit(model, data, config=TrainingConfig(epochs=5), mask=train)
model.eval()
with torch.no_grad():
    expected = model(data)

output = Path("artifacts")
output.mkdir(exist_ok=True)
save_model(model, output / "model.pt")
torch.save(scaler.state_dict(), output / "scaler.pt")

restored = load_model(output / "model.pt", map_location="cpu")
restored.eval()
restored_scaler = FeatureStandardizer(2)
restored_scaler.load_state_dict(torch.load(output / "scaler.pt", weights_only=True))
fresh = raw.clone()
fresh.x = restored_scaler(raw.x)
with torch.no_grad():
    actual = restored(fresh)
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
print("Checkpoint prediction roundtrip passed")
```

This writes `artifacts/model.pt` and `artifacts/scaler.pt` relative to the working
directory. Register any custom basis names before `load_model`. Loading creates
a new model in training mode; call `.eval()` for inference.

Checkpoints do not contain data, optimizer state, RNG state, loss code or
preprocessing statistics. They support prediction reconstruction, not exact
training resumption. `load_model` currently expects `map_location` usable as a
model `.to(...)` destination, such as `"cpu"`, `"cuda"` or a `torch.device`.
For double-precision models, dtype conversion is not recorded in `ModelConfig`;
reconstruct the config, call `.double()`, and load the saved state dict explicitly
when preserving float64 weights is required.

## 18. JSON configuration and CLI

A `ModelConfig` can roundtrip through a normal JSON dictionary:

```python
import json
from pathlib import Path
from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig
from kangnn.data import toy_graph

config = ModelConfig(
    2, 8, 2,
    preprocessing=TransformConfig(kind="kan", basis="rbf"),
    layers=[LayerConfig(operator="gatv2", heads=2)],
)
path = Path("model_config.json")
path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
loaded_config = ModelConfig(**json.loads(path.read_text(encoding="utf-8")))
model = GraphKAN(loaded_config)
assert model(toy_graph()).shape == (6, 2)
```

From the repository root, try the supplied mixed configuration:

```bash
kangnn --help
kangnn --operator sage --kind mlp --epochs 30
kangnn --operator gatv2 --kind kan --basis fourier --loss nll --epochs 30
kangnn --config KAN_framwork/examples/mixed.json --epochs 60 --save /tmp/mixed.pt
```

`--config` supplies the whole model config; `--operator`, `--kind` and `--basis`
apply only when no config is supplied. Epochs, seed, loss and checkpoint path
remain CLI options. The CLI dataset requires `task="node"`, `in_channels=2` and
`out_channels=2`; use Python for your own data, graph/link tasks and other losses.

The CLI's built-in `--basis` choices do not include optional/custom registered
names. To use an extension JSON file, run a Python entry point that registers
extensions before constructing `ModelConfig`/`GraphKAN`, or before calling
`kangnn.cli.main()`. JSON is configuration, not executable custom-function code.

## 19. Configuration reference

### `TransformConfig`

| Field | Default | Meaning |
|---|---|---|
| `kind` | `"mlp"` | `mlp`, `kan`, or dimension-preserving `identity` |
| `basis` | `"bspline"` | Registered family used when kind is KAN |
| `basis_kwargs` | `{}` | Options passed to the basis constructor |
| `hidden` | `()` | Intermediate widths; final width is set by the stage |
| `bias` | `True` | Bias in each feature transform |
| `base_residual` | `True` | KAN-only SiLU base branch |
| `basis_normalization` | `"none"` | KAN basis branch: none or layernorm; raw base input unchanged |
| `coefficient_init` | `"normal"` | KAN coefficient initialization: normal or truncated_normal |
| `coefficient_init_std` | `None` | KAN coefficient std; None uses 1/sqrt(input_width*basis_count) |

### `LayerConfig`

| Field | Default | Meaning |
|---|---|---|
| `operator` | `"gcn"` | `gcn`, `sage`, `gat`, `gatv2` |
| `message` | Bias-free linear MLP | Source transform; also configures independent SAGE root/GATv2 target modules |
| `update` | Identity | Applied after aggregation and graph output bias |
| `attention` | Bias-free linear MLP | Scalar scorer per head, attention operators only |
| `heads` | `1` | Attention heads, averaged output; nonattention requires 1 |
| `dropout` | `0.0` | Attention-weight dropout probability |
| `self_loops` | `True` | GCN/attention loop handling; SAGE adds no loops |
| `residual` | `False` | Learned projected skip from layer input |
| `edge_dim` | `None` | Required edge feature width when attributes are provided |
| `edge` | Linear MLP with bias | Optional edge-feature transform |

### `ModelConfig`

| Field | Default | Meaning |
|---|---|---|
| `in_channels` | Required | Width of data.x |
| `hidden_channels` | Required | Shared encoder width |
| `out_channels` | Required | Classes or output target count |
| `layers` | Two default GCN layers | Ordered list of per-layer configs |
| `preprocessing` | Linear MLP | Learned input-to-hidden embedding |
| `readout` | Identity | After graph pooling or directly on nodes |
| `decision` | Linear MLP | Hidden-to-output, or 2*hidden-to-output for links |
| `task` | `"node"` | `node`, `graph`, or `link` |
| `pooling` | `"mean"` | Graph task: mean, sum, max |
| `dropout` | `0.0` | Activation dropout after graph-layer ReLU |

### `TrainingConfig`

| Field | Default | Meaning |
|---|---|---|
| `epochs` | `100` | Positive number of full-batch updates |
| `lr` | `0.01` | Positive Adam learning rate |
| `weight_decay` | `0.0` | Nonnegative Adam weight decay |
| `kan_l1` | `0.0` | Penalty on mean absolute KAN coefficients, summed across KAN layers |

Configs select architecture when the model is constructed. Changing a config
object afterward does not rebuild the modules and can make runtime settings
inconsistent with their dimensions. Construct a new model for architecture changes.

## 20. Troubleshooting and experiment checklist

| Symptom | Likely cause and correction |
|---|---|
| `ModuleNotFoundError: kangnn` | Activate the correct environment and install the sibling package in editable mode |
| Matrix multiplication dimension error | `in_channels` must equal `data.x.size(1)`; use matching feature width/order at inference |
| CE target dtype error | Use long integer class IDs, not one-hot or floating labels |
| BCE/regression shape warning | Use `[P,1]` for one target, matching output shape exactly |
| Unknown basis | Check spelling and register optional/custom factories before construction/loading |
| Fourier count error | Select an odd `num_basis` of at least 3 |
| Identity dimension error | Match stage input/output widths and remove identity hidden widths |
| Edge attribute configuration error | Provide attributes exactly when edge_dim is configured, in every graph layer |
| GCN edge-feature loop error | Set self_loops=False and construct any desired aligned loops yourself |
| Edge weights rejected | Use GCN for every receiving layer or implement explicit selective forwarding |
| Device mismatch | Move data, masks, model, scaler and any loss buffers to the same device |
| Nonfinite objective | Check finite features, target shape, learning rate and custom basis/loss numerics |
| Training loss differs from evaluation | Dropout/regularization differ; training mode and label masks may also differ |
| Different predictions after load | Use eval mode, the same fitted scaler and feature order; register custom bases first |
| GPU deterministic error | Use a supported CPU path or deliberately relax strict deterministic mode |
| Unexpectedly excellent test scores | Check graph/edge/feature/label leakage and whether evaluation reused training data |

A practical experiment sequence:

1. Specify the task, label meaning, prediction shape and metric before choosing
   transforms. Establish an MLP baseline with the same graph split.
2. Separate train/validation/test correctly for nodes, graphs or links. Fit
   feature preprocessing on training data only.
3. Verify a forward pass and objective on a tiny batch. Check edge orientation,
   target shapes and device placement before a long training run.
4. Change one stage from MLP to KAN at a time. Use the validation split to choose
   basis count, domain/period, hidden widths and training settings.
5. Run multiple seeds for scientific comparisons. Keep data splits fixed when
   comparing architecture choices and record configuration/dependency versions.
6. Save the selected model and its fitted preprocessing. Evaluate the held-out
   test split once for the final report; do not tune based on its score.

Automated checks from the repository root:

```bash
pytest KAN_framwork/tests mathematical_transforms/tests -q
pytest KAN_framwork/tests/test_user_guide.py -q
ruff check KAN_framwork mathematical_transforms
```

The documentation test runs every Python block in this guide in a fresh process
and temporary working directory. CPU examples are executable; CUDA execution
still depends on hardware and supported deterministic operations. Dataset-specific
performance, exact paper reproduction and a graphical UI remain outside these
examples. For architectural equations see [IMPLEMENTATION.md](IMPLEMENTATION.md);
for planned features see [FUTURE_STEPS.md](FUTURE_STEPS.md).
