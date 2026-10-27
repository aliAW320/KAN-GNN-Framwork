"""Real-valued KAN kernels inspired by Z and Laplace transform integrands.

These are feature expansions, not transforms of an entire signal. Positive query
maps keep kernels bounded for arbitrary finite real neural-network features.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LaplaceBasis(nn.Module):
    """B_k(x)=exp(-softplus(x)*t_k), with fixed positive time nodes."""

    def __init__(self, num_basis=8, time_range=(0.1, 4.0)):
        super().__init__()
        if (isinstance(num_basis, bool) or not isinstance(num_basis, int) or num_basis < 1
                or len(time_range) != 2 or not all(math.isfinite(v) for v in time_range)
                or not 0 < time_range[0] <= time_range[1]):
            raise ValueError("Require integer num_basis>=1 and finite 0<time_min<=time_max")
        self.num_basis = num_basis
        self.register_buffer("times", torch.linspace(*time_range, num_basis))

    def forward(self, x):
        if not x.is_floating_point():
            raise ValueError("KAN bases require real floating-point features")
        return torch.exp(-F.softplus(x).unsqueeze(-1) * self.times)


class ZTransformBasis(nn.Module):
    """B_k(x)=(1+softplus(x))**(-k), k=0..num_basis-1.

    z(x)>1 avoids singularities for ordinary real features. k=0 is constant.
    The outer KAN coefficients learn a finite Z-kernel expansion.
    """

    def __init__(self, num_basis=8):
        super().__init__()
        if isinstance(num_basis, bool) or not isinstance(num_basis, int) or num_basis < 1:
            raise ValueError("num_basis must be a positive integer")
        self.num_basis = num_basis
        self.register_buffer("indices", torch.arange(num_basis).float())

    def forward(self, x):
        if not x.is_floating_point():
            raise ValueError("KAN bases require real floating-point features")
        # log1p provides a stable real representation of inverse powers.
        return torch.exp(-torch.log1p(F.softplus(x)).unsqueeze(-1) * self.indices)


def register_kan_bases():
    """Explicitly register 'laplace' and 'z_transform' with installed kangnn.

    Repeated registration of these same classes is safe. A name collision with
    another user's factory fails before modifying either registry entry.
    kangnn is optional and is imported only when this adapter is called.
    """
    from kangnn.basis import BASES, register_basis
    entries = {"laplace": LaplaceBasis, "z_transform": ZTransformBasis}
    for name, factory in entries.items():
        if name in BASES and BASES[name] is not factory:
            raise ValueError(f"KAN basis name {name!r} is already in use")
    for name, factory in entries.items():
        if name not in BASES:
            register_basis(name, factory)
