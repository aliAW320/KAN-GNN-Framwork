import pytest
import torch

from kan_transforms import LaplaceBasis, ZTransformBasis, register_kan_bases


@pytest.mark.parametrize("cls", [LaplaceBasis, ZTransformBasis])
def test_shape_stability_gradients_and_state(cls):
    basis = cls(num_basis=5).double()
    x = torch.tensor([[-1000., -0.2, 0.3, 1000.]], dtype=torch.double, requires_grad=True)
    values = basis(x)
    assert values.shape == (1, 4, 5)
    assert torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    assert torch.autograd.gradcheck(basis, (torch.tensor([[0.3]], dtype=torch.double,
                                                       requires_grad=True),))
    other = cls(num_basis=5).double()
    other.load_state_dict(basis.state_dict())
    torch.testing.assert_close(other(x), values, atol=0, rtol=0)
    with pytest.raises(ValueError):
        basis(torch.ones(2, dtype=torch.complex64))


def test_exact_kernel_formulas():
    x = torch.tensor([[0.2, -0.4]])
    laplace = LaplaceBasis(3)
    positive = torch.nn.functional.softplus(x).unsqueeze(-1)
    torch.testing.assert_close(laplace(x), torch.exp(-positive * laplace.times))
    z = ZTransformBasis(3)
    torch.testing.assert_close(z(x), (1 + positive).pow(-z.indices))


@pytest.mark.parametrize("cls", [LaplaceBasis, ZTransformBasis])
@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_invalid_counts(cls, count):
    with pytest.raises(ValueError):
        cls(num_basis=count)


@pytest.mark.parametrize("interval", [(0., 1.), (2., 1.), (float('nan'), 2.)])
def test_invalid_time_nodes(interval):
    with pytest.raises(ValueError):
        LaplaceBasis(time_range=interval)


@pytest.mark.parametrize("name", ["laplace", "z_transform"])
@pytest.mark.parametrize("operator", ["gcn", "sage", "gat", "gatv2"])
def test_kan_integration_and_checkpoint(name, operator, tmp_path):
    pytest.importorskip("kangnn")
    from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig, TransformConfig,
                        fit, load_model, save_model, seed_everything)
    from kangnn.data import toy_graph
    register_kan_bases()
    register_kan_bases()  # adapter is idempotent
    cfg = TransformConfig(kind="kan", basis=name, basis_kwargs={"num_basis": 4})
    def run():
        seed_everything(9)
        model = GraphKAN(ModelConfig(2, 4, 2, preprocessing=cfg, readout=cfg, decision=cfg,
                                    layers=[LayerConfig(operator=operator, message=cfg,
                                                        update=cfg, attention=cfg)]))
        history = fit(model, toy_graph(), config=TrainingConfig(epochs=3))
        return model, history
    model, first = run()
    _, second = run()
    assert first == second
    assert torch.isfinite(torch.tensor(first)).all()
    path = tmp_path / f"{name}-{operator}.pt"
    save_model(model, path)
    restored = load_model(path)
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(toy_graph()), restored(toy_graph()), atol=0, rtol=0)


def test_registration_collision_is_atomic(monkeypatch):
    pytest.importorskip("kangnn")
    from kangnn.basis import BASES
    monkeypatch.delitem(BASES, "laplace", raising=False)
    monkeypatch.setitem(BASES, "z_transform", object())
    with pytest.raises(ValueError):
        register_kan_bases()
    assert "laplace" not in BASES


@pytest.mark.parametrize("cls", [LaplaceBasis, ZTransformBasis])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_basis(cls):
    basis = cls().double()
    gpu_basis = cls().double().cuda()
    gpu_basis.load_state_dict(basis.state_dict())
    x = torch.tensor([[-0.3, 0.4]], dtype=torch.double)
    gpu = x.cuda().requires_grad_()
    result = gpu_basis(gpu)
    torch.testing.assert_close(result.cpu(), basis(x))
    result.sum().backward()
    assert torch.isfinite(gpu.grad).all()
