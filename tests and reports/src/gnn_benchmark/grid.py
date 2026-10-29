"""Explicit candidate enumeration; no-KAN configurations are never trained."""
from dataclasses import asdict, dataclass
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from kangnn import GraphKAN, LayerConfig, ModelConfig, TransformConfig
from kangnn.basis import BASES, register_basis

BASE_NAMES = ("fourier", "relu", "bspline", "rbf", "sigmoid", "tanh")
OPERATORS = ("gcn", "gat", "gatv2", "sage")
PREPROCESS = ("none", "mlp", "kan2", "kan4")
HEADS = ("mlp", "kan1", "kan2", "kan4")


class SmoothBasis(nn.Module):
    """Shifted sigmoid/tanh features; not an ordinary MLP activation switch."""

    def __init__(self, num_basis=8, function="sigmoid"):
        super().__init__()
        if num_basis < 1 or function not in {"sigmoid", "tanh"}:
            raise ValueError("Invalid smooth-basis configuration")
        self.num_basis, self.function = num_basis, function
        self.centers = nn.Parameter(torch.linspace(-2., 2., num_basis))
        self.raw_scales = nn.Parameter(torch.zeros(num_basis))

    def forward(self, x):
        z = (x.unsqueeze(-1) - self.centers) * (F.softplus(self.raw_scales) + 1e-6)
        return z.sigmoid() if self.function == "sigmoid" else z.tanh()


class SigmoidBasis(SmoothBasis):
    def __init__(self, num_basis=8):
        super().__init__(num_basis, "sigmoid")


class TanhBasis(SmoothBasis):
    def __init__(self, num_basis=8):
        super().__init__(num_basis, "tanh")


def register_bases():
    for name, factory in {"sigmoid": SigmoidBasis, "tanh": TanhBasis}.items():
        if name in BASES and BASES[name] is not factory:
            raise ValueError(f"Basis collision: {name}")
        if name not in BASES:
            register_basis(name, factory)


@dataclass(frozen=True)
class Candidate:
    preprocessing: str
    operator: str
    message: str
    update: str
    decision: str
    basis: str

    @property
    def has_kan(self):
        return (self.preprocessing.startswith("kan") or self.message == "kan"
                or self.update == "kan" or self.decision.startswith("kan"))

    @property
    def id(self):
        return (f"pre-{self.preprocessing}__{self.operator}__msg-{self.message}"
                f"__upd-{self.update}__head-{self.decision}__{self.basis}")

    def to_dict(self):
        return {"model_id": self.id, **asdict(self), "has_kan": self.has_kan}


def enumerate_grid(independent=False):
    modes = list(product(("mlp", "kan"), repeat=2)) if independent else [("mlp", "mlp"),
                                                                                     ("kan", "kan")]
    return [Candidate(pre, op, msg, upd, head, basis)
            for pre, op, (msg, upd), head, basis in product(PREPROCESS, OPERATORS, modes,
                                                          HEADS, BASE_NAMES)]


def transform(kind, basis, width, depth=1):
    options = {"num_basis": 9 if basis == "fourier" else 8}
    return TransformConfig(kind=kind, basis=basis, basis_kwargs=options,
                           hidden=(width,) * (depth - 1))


def build_model(candidate, in_channels, out_channels, width=32, graph_layers=2):
    register_bases()
    if not candidate.has_kan:
        raise ValueError("All-MLP candidates are excluded from training")
    if width < 1 or graph_layers < 1:
        raise ValueError("Widths and graph layer count must be positive")
    # No preprocessing means true identity; do not hide a learned projection here.
    hidden = in_channels if candidate.preprocessing == "none" else width
    if candidate.preprocessing == "none":
        pre = TransformConfig(kind="identity")
    elif candidate.preprocessing == "mlp":
        pre = transform("mlp", candidate.basis, hidden)
    else:
        pre = transform("kan", candidate.basis, hidden, int(candidate.preprocessing[3:]))
    head_kind = "mlp" if candidate.decision == "mlp" else "kan"
    head_depth = 1 if head_kind == "mlp" else int(candidate.decision[3:])
    return GraphKAN(ModelConfig(
        in_channels, hidden, out_channels, preprocessing=pre, task="graph",
        layers=[LayerConfig(operator=candidate.operator,
                            message=transform(candidate.message, candidate.basis, hidden),
                            update=transform(candidate.update, candidate.basis, hidden),
                            attention=TransformConfig(bias=False)) for _ in range(graph_layers)],
        decision=transform(head_kind, candidate.basis, hidden, head_depth),
    ))
