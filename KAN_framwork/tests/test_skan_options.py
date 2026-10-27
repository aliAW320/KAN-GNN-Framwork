"""Independent equation checks for optional SKAN-inspired settings."""
import json
import math

import pytest
import torch
from torch import nn

from kangnn import (GraphKAN, KANLinear, LayerConfig, ModelConfig, TransformConfig,
                    load_model, save_model, skan_transform)
from kangnn.basis import RBFBasis
from kangnn.data import toy_graph


@pytest.mark.parametrize("centers", [False, True])
@pytest.mark.parametrize("widths", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_width_initialization_freezing_and_gradients(centers, widths, shared):
    basis = RBFBasis(4, width=0.3, trainable_centers=centers,
                     trainable_widths=widths, shared_width=shared).double()
    assert basis.centers.requires_grad == centers
    assert basis.raw_width.requires_grad == widths
    torch.testing.assert_close(basis.widths, torch.full_like(basis.widths, 0.3))
    assert basis.raw_width.shape == ((1,) if shared else (4,))
    x = torch.tensor([[0.13, 0.27]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(basis, (x,))
    basis(x).sum().backward()
    assert (basis.centers.grad is not None) == centers
    assert (basis.raw_width.grad is not None) == widths
    with torch.no_grad():
        basis.raw_width.fill_(-1000)
    assert torch.isfinite(basis(x)).all() and (basis.widths > 0).all()


def test_normalized_basis_and_raw_base_match_manual_equation():
    config = skan_transform(num_basis=4)
    layer = KANLinear(3, 2, config).double()
    x = torch.tensor([[0.2, -0.4, 0.8], [1.2, 0.1, -0.6]], dtype=torch.double,
                     requires_grad=True)
    normalized = nn.functional.layer_norm(x, (3,), layer.basis_norm.weight,
                                         layer.basis_norm.bias)
    spacing = 4 / 3
    kernels = torch.exp(-((normalized.unsqueeze(-1) - layer.basis.centers) / spacing).square())
    expected = nn.functional.linear(kernels.flatten(-2), layer.coefficients.flatten(1))
    expected = expected + layer.base(nn.functional.silu(x)) + layer.bias
    torch.testing.assert_close(layer(x), expected)
    assert torch.autograd.gradcheck(layer, (x,))
    layer(x).sum().backward()
    assert layer.basis_norm.weight.grad is not None


def test_normalization_does_not_change_base_input():
    layer = KANLinear(3, 2, skan_transform())
    with torch.no_grad():
        layer.coefficients.zero_()
    x = torch.tensor([[1., 3., 2.]])
    torch.testing.assert_close(layer(x), layer.base(nn.functional.silu(x)) + layer.bias)
    plain = KANLinear(3, 2)
    assert isinstance(plain.basis_norm, nn.Identity)
    assert not any(k.startswith("basis_norm.") for k in plain.state_dict())


def test_initialization_controls_and_invalid_options():
    torch.manual_seed(4)
    small = KANLinear(64, 64, TransformConfig(kind="kan", coefficient_init_std=0.02))
    assert math.isclose(small.coefficients.std().item(), 0.02, rel_tol=0.03)
    for kwargs in ({"basis_normalization": "batchnorm"}, {"coefficient_init": "bad"},
                   {"coefficient_init_std": 0}, {"coefficient_init_std": float("nan")}):
        with pytest.raises(ValueError):
            TransformConfig(**kwargs)
    for width in (0., -1., 1e-8, float("inf")):
        with pytest.raises(ValueError):
            RBFBasis(width=width)
    for kwargs in ({"num_basis": 1}, {"domain": (2., 1.)}, {"domain": (0., 1e-8)}):
        with pytest.raises(ValueError):
            skan_transform(**kwargs)


@pytest.mark.parametrize("operator", ["gcn", "sage", "gat", "gatv2"])
def test_skan_configs_checkpoint_and_backward(operator, tmp_path):
    cfg = skan_transform(num_basis=4, trainable_centers=False)
    config = ModelConfig(2, 4, 2, preprocessing=cfg, decision=cfg,
                         layers=[LayerConfig(operator=operator, update=cfg)])
    model = GraphKAN(ModelConfig(**json.loads(json.dumps(config.to_dict()))))
    data = toy_graph()
    nn.functional.cross_entropy(model(data), data.y).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path / "skan.pt"
    model.eval()
    save_model(model, path)
    restored = load_model(path)
    restored.eval()
    torch.testing.assert_close(restored(data), model(data), atol=0, rtol=0)
    assert not restored.preprocessing[0].basis.centers.requires_grad


def test_old_config_and_state_compatibility():
    # A pre-extension config/state has neither normalization fields nor norm weights.
    old_config = {"kind": "kan", "basis": "rbf", "basis_kwargs": {}, "hidden": (),
                  "bias": True, "base_residual": True}
    layer = KANLinear(2, 3, TransformConfig(**old_config))
    expected_keys = {"coefficients", "basis.centers", "basis.raw_width", "base.weight", "bias"}
    assert set(layer.state_dict()) == expected_keys
    restored = KANLinear(2, 3, TransformConfig(**old_config))
    restored.load_state_dict(layer.state_dict(), strict=True)
    x = torch.tensor([[0.1, 0.2]])
    torch.testing.assert_close(restored(x), layer(x), atol=0, rtol=0)


def test_skan_training_is_reproducible():
    from kangnn import TrainingConfig, fit, seed_everything

    def run():
        seed_everything(7)
        model = GraphKAN(ModelConfig(
            2, 8, 2, layers=[LayerConfig(message=TransformConfig(kind="identity"),
                                         update=skan_transform())],
            decision=skan_transform(num_basis=4),
        ))
        history = fit(model, toy_graph(), config=TrainingConfig(epochs=20))
        return history, model.state_dict()

    first, a = run()
    second, b = run()
    assert first == second and first[-1] < first[0]
    assert all(torch.equal(a[key], b[key]) for key in a)
