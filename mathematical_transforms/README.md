# Z and Laplace transforms for PyTorch and KAN

This separate package provides both numerical signal transforms and real-valued
transform-inspired KAN bases. Import name: `kan_transforms`. Numerical tools
require only PyTorch; KAN integration additionally uses the sibling `kangnn`
package. No modifications to the original KAN basis registry happen on import.

For an end-to-end graph-framework tutorial, see the
[KAN-GNN user guide](../KAN_framwork/docs/USER_GUIDE.md), including Z/Laplace integration.

## Installation

From the repository root, using the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -e './mathematical_transforms[test]'
# Also install the graph framework if using KAN integration:
python -m pip install -e ./KAN_framwork
python mathematical_transforms/examples/numerical.py
python mathematical_transforms/examples/kan_integration.py
```

## Numerical transforms

```python
import torch
from kan_transforms import z_transform, laplace_transform

sequence = torch.tensor([1., 2., 3.], dtype=torch.float64)
z = torch.tensor([1., 2., 1j], dtype=torch.complex128)
X = z_transform(sequence, z)  # 1 + 2/z + 3/z**2

times = torch.linspace(0, 10, 10001, dtype=torch.float64)
signal = torch.exp(-2 * times)
s = torch.tensor([0., 1., 1.+1j], dtype=torch.complex128)
F = laplace_transform(signal, times, s)  # approximately 1/(s+2), small omitted tail
```

| Function | Definition | Scope |
|---|---|---|
| `z_transform(signal, z, start_index=0, dim=-1)` | `sum_k signal[k] * z**(-(k+start_index))` | Exact finite sum up to floating-point error; default truncated unilateral sequence |
| `laplace_transform(signal, times, s, dim=-1)` | Trapezoidal integration of `signal(t)*exp(-s*t)` | Approximation over the provided finite time interval |

`start_index` is an integer. Negative values represent a finite sequence that
includes negative indices; this does not reconstruct unknown samples.
For a finite sequence, Z evaluation on `exp(2*pi*i*k/N)` matches the forward DFT
convention used by `torch.fft.fft`.

Signals may have batch dimensions, with `dim` selecting the sample axis. A vector
of Q query points yields the batch shape followed by `[Q]`; a scalar query yields
only the batch shape. Queries are scalars or nonempty one-dimensional tensors.
All tensors must share a device. Supported dtypes: float32, float64, complex64,
complex128, with normal PyTorch promotion. Integer tensors must be converted to
floating point explicitly.

Laplace times must be real, nonnegative, strictly increasing, and contain at least
two points; nonuniform spacing is supported. The function integrates between the
first and last times. It does not assume unobserved values before or after that
interval, derive a symbolic expression, or infer a region of convergence.
`z=0`, nonfinite inputs and malformed shapes are rejected. Large negative real
parts of `s`, tiny nonzero `z`, or large sequence indices may overflow; choose
queries and precision appropriate for your data.

Operations remain in PyTorch's autograd graph. For complex outputs, train with a
real objective, for example `output.abs().square().mean()`. The mathematical
conventions follow the [UBC unilateral Z-transform notes](https://www.math.ubc.ca/~feldman/m267/genfnl.pdf).
Complex differentiation follows [PyTorch's autograd convention](https://docs.pytorch.org/docs/stable/notes/autograd).

## Transform-inspired KAN bases

A numerical transform maps a sequence or sampled function to query values.
A KAN basis maps each scalar feature to basis activations. These are different
operations. The two supplied KAN bases use the transform kernels, with a positive
feature-to-query map that avoids singularities and keeps activations bounded:

| Basis | Configuration name | Definition |
|---|---|---|
| `LaplaceBasis` | `laplace` | `B_k(x) = exp(-softplus(x)*t_k)`; fixed positive time nodes |
| `ZTransformBasis` | `z_transform` | `B_k(x) = (1+softplus(x))**(-k)`; k starts at zero |

Both return `[..., input_features, num_basis]` for real input features and work
with the existing KAN coefficient contraction. Time/index nodes are buffers,
not learned parameters; the outer KAN coefficients and optional SiLU branch are
trainable. These bases are design extensions, not paper-reproduction claims.

```python
from kan_transforms import register_kan_bases
from kangnn import GraphKAN, ModelConfig, LayerConfig, TransformConfig

register_kan_bases()
laplace = TransformConfig(kind="kan", basis="laplace",
                          basis_kwargs={"num_basis": 8, "time_range": (0.1, 4.0)})
z = TransformConfig(kind="kan", basis="z_transform",
                    basis_kwargs={"num_basis": 8})
model = GraphKAN(ModelConfig(
    2, 8, 2,
    preprocessing=laplace,
    layers=[LayerConfig(operator="gatv2", message=z, update=laplace)],
    readout=laplace,
    decision=z,
))
```

These names work in any configurable KAN stage, including attention and edge
encoders, and with GCN, GraphSAGE, GAT and GATv2. Register them before constructing
models from JSON or loading checkpoints. The existing demonstration CLI's
`--basis` choices are unchanged; use Python or its JSON config path after calling
`register_kan_bases()`. Repeated registration of these same classes is safe;
collisions with another factory fail before either entry is changed.

You can also use `LaplaceBasis`/`ZTransformBasis` directly without installing PyG
or `kangnn`. Extend the numerical module with ordinary PyTorch functions, or
supply additional `nn.Module` basis classes through `kangnn.register_basis`.

## GPU and verification

Move signal, queries, times and models to the same device:

```python
if torch.cuda.is_available():
    X_gpu = z_transform(sequence.cuda(), z.cuda())
    F_gpu = laplace_transform(signal.cuda(), times.cuda(), s.cuda())
    model = model.cuda()
```

The implementation has no CPU-specific numerical operations. CUDA parity and
backward tests run automatically when a CUDA device is available; otherwise they
are explicitly skipped. CPU tests include analytical references, finite-domain
quadrature error checks, DFT equivalence, complex autograd gradchecks, shape/batch
behavior, every graph backbone, repeated training and checkpoint roundtrips.

```bash
pytest mathematical_transforms/tests --cov=kan_transforms --cov-fail-under=90
ruff check mathematical_transforms
```

See [implementation notes](docs/IMPLEMENTATION.md), [future steps](docs/FUTURE_STEPS.md)
and [the dedicated workflow](../.github/workflows/transforms-ci.yml).
