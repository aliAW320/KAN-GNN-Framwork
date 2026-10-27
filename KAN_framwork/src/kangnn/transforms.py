"""Uniform feature-transform configuration for every trainable stage."""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from .basis import make_basis


@dataclass
class TransformConfig:
    kind: str = "mlp"
    basis: str = "bspline"
    basis_kwargs: dict = field(default_factory=dict)
    hidden: tuple[int, ...] = ()
    bias: bool = True
    base_residual: bool = True
    basis_normalization: str = "none"
    coefficient_init: str = "normal"
    coefficient_init_std: float | None = None

    def __post_init__(self):
        if self.kind not in {"mlp", "kan", "identity"}:
            raise ValueError("kind must be mlp, kan, or identity")
        if self.basis_normalization not in {"none", "layernorm"}:
            raise ValueError("basis_normalization must be none or layernorm")
        if self.coefficient_init not in {"normal", "truncated_normal"}:
            raise ValueError("coefficient_init must be normal or truncated_normal")
        if (self.coefficient_init_std is not None and
                (not math.isfinite(self.coefficient_init_std) or self.coefficient_init_std <= 0)):
            raise ValueError("coefficient_init_std must be finite and positive")
        if any(w <= 0 for w in self.hidden):
            raise ValueError("hidden widths must be positive")


class KANLinear(nn.Module):
    """y_o = sum_i,k c_oik B_k(x_i) + W_base SiLU(x) + bias.

    Coefficients are independent per input-output edge. Basis centers/widths
    are shared across edges in this implementation to bound memory use.
    """

    def __init__(self, in_channels, out_channels, config=None):
        super().__init__()
        config = config or TransformConfig(kind="kan")
        self.basis = make_basis(config.basis, **config.basis_kwargs)
        self.basis_norm = (nn.LayerNorm(in_channels) if config.basis_normalization == "layernorm"
                           else nn.Identity())
        self.coefficients = nn.Parameter(torch.empty(out_channels, in_channels,
                                                     self.basis.num_basis))
        std = (config.coefficient_init_std if config.coefficient_init_std is not None
               else (in_channels * self.basis.num_basis) ** -0.5)
        initializer = (nn.init.trunc_normal_ if config.coefficient_init == "truncated_normal"
                       else nn.init.normal_)
        initializer(self.coefficients, std=std)
        self.base = nn.Linear(in_channels, out_channels, bias=False) if config.base_residual else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if config.bias else None

    def forward(self, x):
        b = self.basis(self.basis_norm(x))
        if b.shape != (*x.shape, self.basis.num_basis):
            raise ValueError("Custom basis must return shape [*x.shape, num_basis]")
        y = torch.einsum("...ik,oik->...o", b, self.coefficients)
        if self.base is not None:
            y = y + self.base(F.silu(x))
        return y if self.bias is None else y + self.bias


def make_transform(in_channels, out_channels, config=None):
    config = config or TransformConfig()
    if config.kind == "identity":
        if in_channels != out_channels or config.hidden:
            raise ValueError("Identity requires equal dimensions and no hidden layers")
        return nn.Identity()
    widths = [in_channels, *config.hidden, out_channels]
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(KANLinear(a, b, config) if config.kind == "kan"
                      else nn.Linear(a, b, bias=config.bias))
        if i < len(widths) - 2 and config.kind == "mlp":
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def skan_transform(*, num_basis=8, domain=(-2.0, 2.0), trainable_centers=True,
                   trainable_widths=True, base_residual=True, hidden=()):
    """Configure an SKAN-style feature layer, not a full paper-model reproduction.

    Uses normalized RBF input, raw SiLU base input, one shared positive bandwidth,
    grid-spacing Gaussian denominator, and truncated-normal coefficients (std .1).
    """
    if (isinstance(num_basis, bool) or not isinstance(num_basis, int) or num_basis < 2
            or len(domain) != 2 or not all(math.isfinite(v) for v in domain)
            or domain[0] >= domain[1]):
        raise ValueError("SKAN initialization requires >=2 bases and finite increasing domain")
    width = (domain[1] - domain[0]) / (num_basis - 1) / math.sqrt(2)
    if width <= 1e-6:
        raise ValueError("SKAN domain spacing is too small")
    return TransformConfig(
        kind="kan", basis="rbf", hidden=hidden, bias=base_residual,
        base_residual=base_residual, basis_normalization="layernorm",
        coefficient_init="truncated_normal", coefficient_init_std=0.1,
        basis_kwargs={"num_basis": num_basis, "domain": domain, "width": width,
                      "shared_width": True, "trainable_centers": trainable_centers,
                      "trainable_widths": trainable_widths},
    )
