"""Univariate basis modules. A basis maps [..., in] to [..., in, num_basis]."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BSplineBasis(nn.Module):
    """Open uniform, clamped B-splines using Cox-de Boor recursion.

    Inputs outside the configured interval are clamped to its boundary.
    num_basis must exceed degree; degree=3 means cubic splines.
    """

    def __init__(self, num_basis=8, degree=3, domain=(-1.0, 1.0), **kwargs):
        super().__init__()
        if degree < 0 or num_basis <= degree or domain[0] >= domain[1]:
            raise ValueError("Require num_basis > degree >= 0 and increasing domain")
        self.num_basis, self.degree = num_basis, degree
        low, high = domain
        interior = torch.linspace(low, high, num_basis - degree + 1)[1:-1]
        self.register_buffer("knots", torch.cat((torch.full((degree + 1,), float(low)),
                                                interior,
                                                torch.full((degree + 1,), float(high)))))

    def forward(self, x):
        t = self.knots
        z = x.clamp(t[0], t[-1]).unsqueeze(-1)
        b = ((z >= t[:-1]) & (z < t[1:])).to(x.dtype)
        for p in range(1, self.degree + 1):
            n = len(t) - p - 1
            dl, dr = t[p:p+n] - t[:n], t[p+1:p+1+n] - t[1:1+n]
            left = (z - t[:n]) / torch.where(dl > 0, dl, torch.ones_like(dl))
            right = (t[p+1:p+1+n] - z) / torch.where(dr > 0, dr, torch.ones_like(dr))
            b = left * b[..., :n] * (dl > 0) + right * b[..., 1:n+1] * (dr > 0)
        endpoint = torch.zeros_like(b)
        endpoint[..., -1] = 1
        return torch.where(z == t[-1], endpoint, b)


class FourierBasis(nn.Module):
    """Constant followed by paired sine/cosine harmonics; period defaults to 2*pi."""

    def __init__(self, num_basis=9, period=2 * math.pi, **kwargs):
        super().__init__()
        if num_basis < 3 or num_basis % 2 != 1 or period <= 0:
            raise ValueError("Fourier requires odd num_basis >= 3 and positive period")
        self.num_basis, self.period = num_basis, period
        self.register_buffer("frequencies", torch.arange(1, (num_basis + 1) // 2).float())

    def forward(self, x):
        phase = x.unsqueeze(-1) * self.frequencies * (2 * math.pi / self.period)
        return torch.cat((torch.ones_like(x.unsqueeze(-1)), phase.sin(), phase.cos()), -1)


class ReLUBasis(nn.Module):
    """Learnable localized ReLU ramps: relu(x-a)-relu(x-a-width)."""

    def __init__(self, num_basis=8, domain=(-1.0, 1.0), **kwargs):
        super().__init__()
        if num_basis < 1 or domain[0] >= domain[1]:
            raise ValueError("Require num_basis >= 1 and increasing domain")
        self.num_basis = num_basis
        self.centers = nn.Parameter(torch.linspace(*domain, num_basis))
        self.raw_width = nn.Parameter(torch.zeros(num_basis))

    def forward(self, x):
        z = x.unsqueeze(-1) - self.centers
        return F.relu(z) - F.relu(z - F.softplus(self.raw_width))


class RBFBasis(nn.Module):
    """Gaussian kernels with independently configurable centers and widths.

    width denotes standard deviation in exp(-0.5*((x-center)/width)**2).
    Defaults retain the original learned, per-basis widths and initialization.
    """

    def __init__(self, num_basis=8, domain=(-1.0, 1.0), *, width=None,
                 trainable_centers=True, trainable_widths=True, shared_width=False):
        super().__init__()
        if (isinstance(num_basis, bool) or not isinstance(num_basis, int) or num_basis < 1
                or len(domain) != 2 or not all(math.isfinite(v) for v in domain)
                or domain[0] >= domain[1]):
            raise ValueError("Require integer num_basis >= 1 and finite increasing domain")
        if width is not None and (not math.isfinite(width) or width <= 1e-6):
            raise ValueError("width must be finite and greater than 1e-6")
        self.num_basis = num_basis
        self.centers = nn.Parameter(torch.linspace(*domain, num_basis),
                                    requires_grad=trainable_centers)
        raw = 0.0
        if width is not None:
            target = width - 1e-6
            raw = target + math.log(-math.expm1(-target))
        self.raw_width = nn.Parameter(torch.full((1 if shared_width else num_basis,), raw),
                                      requires_grad=trainable_widths)

    @property
    def widths(self):
        """Positive effective standard deviations (one entry for shared width)."""
        return F.softplus(self.raw_width) + 1e-6

    def forward(self, x):
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.widths).square())


BASES = {"bspline": BSplineBasis, "fourier": FourierBasis,
         "relu": ReLUBasis, "rbf": RBFBasis}


def register_basis(name, factory):
    """Register trusted Python factory(**kwargs) -> nn.Module with num_basis."""
    if not name or name in BASES or not callable(factory):
        raise ValueError("Basis name must be new, nonempty, and factory callable")
    BASES[name] = factory


def make_basis(name, **kwargs):
    if name not in BASES:
        raise ValueError(f"Unknown basis {name!r}; choose from {sorted(BASES)}")
    basis = BASES[name](**kwargs)
    if not isinstance(basis, nn.Module) or not isinstance(basis.num_basis, int):
        raise TypeError("Basis must be nn.Module exposing integer num_basis")
    return basis
