import math

import pytest
import torch
from torch import nn

from kangnn import KANLinear, TransformConfig, make_transform, register_basis
from kangnn.basis import BASES, make_basis


@pytest.mark.parametrize("name", ["bspline", "fourier", "relu", "rbf"])
def test_basis_values_gradients_and_state(name):
    basis = make_basis(name, num_basis=9).double()
    x = torch.tensor([[-0.73, 0.13], [0.47, 0.81]], dtype=torch.double, requires_grad=True)
    values = basis(x)
    assert values.shape == (2, 2, 9)
    assert torch.isfinite(values).all()
    assert torch.autograd.gradcheck(basis, (x,))
    clone = make_basis(name, num_basis=9).double()
    clone.load_state_dict(basis.state_dict())
    torch.testing.assert_close(clone(x), values)


def test_bspline_partition_support_and_endpoints():
    b = make_basis("bspline", num_basis=8, degree=3)
    x = torch.linspace(-1, 1, 101).unsqueeze(-1)
    values = b(x)
    torch.testing.assert_close(values.sum(-1), torch.ones_like(x))
    assert (values >= 0).all()
    assert values[0, 0, 0] == 1 and values[-1, 0, -1] == 1
    torch.testing.assert_close(b(torch.tensor([[-5.], [5.]])), values[[0, -1]])
    assert (values[50] > 0).sum() <= 4


def test_fourier_periodicity():
    b = make_basis("fourier", num_basis=9)
    x = torch.tensor([[-0.3, 0.7]])
    torch.testing.assert_close(b(x), b(x + 2 * math.pi), atol=1e-6, rtol=1e-5)


def test_rbf_formula_and_learnable_parameters():
    b = make_basis("rbf", num_basis=3)
    assert b(torch.tensor([[0.]]))[0, 0, 1] == 1
    b(torch.tensor([[0.2]])).sum().backward()
    assert b.centers.grad.abs().sum() > 0
    assert b.raw_width.grad.abs().sum() > 0


@pytest.mark.parametrize("basis", ["bspline", "fourier", "relu", "rbf"])
@pytest.mark.parametrize("residual", [True, False])
def test_kan_edge_coefficients_and_backward(basis, residual):
    cfg = TransformConfig(kind="kan", basis=basis, basis_kwargs={"num_basis": 9},
                          base_residual=residual, bias=False)
    layer = KANLinear(2, 3, cfg)
    x = torch.tensor([[0.2, 0.4]], requires_grad=True)
    y = layer(x)
    expected = (layer.basis(x).unsqueeze(1) * layer.coefficients.unsqueeze(0)).sum((-1, -2))
    if residual:
        expected = expected + layer.base(torch.nn.functional.silu(x))
    torch.testing.assert_close(y, expected)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.coefficients.grad.abs().sum() > 0


def test_custom_basis_and_contract():
    class Square(nn.Module):
        num_basis = 1

        def forward(self, x):
            return x.square().unsqueeze(-1)

    register_basis("test_square", lambda **kwargs: Square())
    layer = KANLinear(2, 1, TransformConfig(kind="kan", basis="test_square",
                                          base_residual=False, bias=False))
    with torch.no_grad():
        layer.coefficients.fill_(1)
    torch.testing.assert_close(layer(torch.tensor([[2., 3.]])), torch.tensor([[13.]]))
    with pytest.raises(ValueError):
        register_basis("test_square", Square)
    BASES.pop("test_square")


@pytest.mark.parametrize("name,kwargs", [("unknown", {}), ("bspline", {"degree": 8}),
                                         ("fourier", {"num_basis": 8}),
                                         ("relu", {"num_basis": 0}),
                                         ("rbf", {"domain": (1, -1)})])
def test_invalid_basis(name, kwargs):
    with pytest.raises(ValueError):
        make_basis(name, **kwargs)


def test_transforms():
    for kind in ("mlp", "kan"):
        t = make_transform(2, 3, TransformConfig(kind=kind, hidden=(4, 5)))
        assert t(torch.randn(6, 2)).shape == (6, 3)
    with pytest.raises(ValueError):
        make_transform(2, 3, TransformConfig(kind="identity"))
    with pytest.raises(ValueError):
        TransformConfig(kind="bad")
