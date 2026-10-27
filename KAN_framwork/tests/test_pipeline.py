import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from kangnn import (FeatureStandardizer, GraphKAN, LayerConfig, ModelConfig, TrainingConfig,
                    TransformConfig, evaluate, fit, load_model, make_loss, node_split,
                    register_loss, save_model, seed_everything)
from kangnn.data import toy_graph
from kangnn.training import LOSSES, coefficient_penalty


@pytest.mark.parametrize("operator", ["gcn", "sage", "gat", "gatv2"])
@pytest.mark.parametrize("basis", [None, "bspline", "fourier", "relu", "rbf"])
@pytest.mark.parametrize("task", ["node", "graph", "link"])
def test_complete_pipeline(operator, basis, task):
    t = TransformConfig(kind="mlp" if basis is None else "kan", basis=basis or "rbf",
                        basis_kwargs={"num_basis": 9})
    config = ModelConfig(2, 4, 2, preprocessing=t, readout=t, decision=t, task=task,
                         layers=[LayerConfig(operator=operator, message=t, update=t)])
    model = GraphKAN(config)
    data = toy_graph()
    if task == "graph":
        data = Batch.from_data_list([data, data.clone()])
        data.y = torch.tensor([0, 1])
    elif task == "link":
        data.edge_label_index = torch.tensor([[0, 0, 3], [1, 4, 5]])
        data.y = torch.tensor([1, 0, 1])
    output = model(data)
    assert output.shape == (6 if task == "node" else 2 if task == "graph" else 3, 2)
    history = fit(model, data, config=TrainingConfig(epochs=2, kan_l1=0.001))
    assert all(torch.isfinite(torch.tensor(history)))
    assert set(evaluate(model, data)) == {"accuracy", "loss"}
    rebuilt = GraphKAN(ModelConfig(**json.loads(json.dumps(config.to_dict()))))
    rebuilt.load_state_dict(model.state_dict())
    model.eval()
    rebuilt.eval()
    torch.testing.assert_close(model(data), rebuilt(data))


@pytest.mark.parametrize("pooling", ["mean", "sum", "max"])
def test_batch_is_independent_and_permutation_invariant(pooling):
    data = toy_graph()
    model = GraphKAN(ModelConfig(2, 4, 2, task="graph", pooling=pooling))
    model.eval()
    first = model(data)
    second = data.clone()
    second.x = second.x * 2
    batch = Batch.from_data_list([data, second])
    torch.testing.assert_close(model(batch), torch.cat([first, model(second)]))
    p = torch.tensor([3, 1, 5, 2, 4, 0])
    permuted = Data(x=data.x[p], edge_index=torch.argsort(p)[data.edge_index])
    torch.testing.assert_close(model(permuted), first)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "nll", "bce_logits", "mse", "l1", "smooth_l1"])
def test_losses_train_and_backpropagate(loss_name):
    data = toy_graph()
    multiclass = loss_name in {"cross_entropy", "nll"}
    model = GraphKAN(ModelConfig(2, 4, 2 if multiclass else 1))
    if not multiclass:
        data.y = data.y.float().unsqueeze(-1)
    values = fit(model, data, loss_name, TrainingConfig(epochs=3))
    assert len(values) == 3 and all(torch.isfinite(torch.tensor(values)))


def test_custom_loss_and_callable():
    class Custom(nn.Module):
        def forward(self, pred, target):
            return nn.functional.cross_entropy(pred, target, label_smoothing=0.1)
    register_loss("test_custom", Custom)
    for loss in ("test_custom", Custom(), lambda p, y: nn.functional.cross_entropy(p, y)):
        model = GraphKAN(ModelConfig(2, 4, 2))
        assert len(fit(model, toy_graph(), loss, TrainingConfig(epochs=1))) == 1
    with pytest.raises(ValueError):
        register_loss("test_custom", Custom)
    LOSSES.pop("test_custom")
    with pytest.raises(ValueError):
        make_loss("bad")


def test_identical_runs_learn_and_masks_exclude_heldout_labels():
    def run():
        seed_everything(19)
        t = TransformConfig(kind="kan", basis="rbf")
        model = GraphKAN(ModelConfig(2, 8, 2, preprocessing=t, decision=t,
                                    layers=[LayerConfig(message=t)]))
        history = fit(model, toy_graph(), config=TrainingConfig(epochs=40))
        return model, history
    first, h1 = run()
    second, h2 = run()
    assert h1 == h2 and h1[-1] < h1[0] * 0.2
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    data = toy_graph()
    mask = torch.tensor([True, True, False, True, True, False])
    seed_everything(10)
    first = GraphKAN(ModelConfig(2, 4, 2))
    seed_everything(10)
    second = GraphKAN(ModelConfig(2, 4, 2))
    changed = data.clone()
    changed.y[~mask] = 100  # Invalid labels must never reach the objective.
    assert fit(first, data, mask=mask, config=TrainingConfig(epochs=2)) == fit(
        second, changed, mask=mask, config=TrainingConfig(epochs=2))


def test_standardizer_splits_and_serialization(tmp_path):
    train, validation, test = node_split(20, seed=5)
    assert torch.all(train.int() + validation.int() + test.int() == 1)
    assert torch.equal(train, node_split(20, seed=5)[0])
    x = torch.arange(40).float().view(20, 2)
    scaler = FeatureStandardizer(2).fit(x, train)
    transformed = scaler(x)
    torch.testing.assert_close(transformed[train].mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    changed = x.clone()
    changed[~train] += 10000
    other = FeatureStandardizer(2).fit(changed, train)
    torch.testing.assert_close(scaler.mean, other.mean)
    state = scaler.state_dict()
    loaded = FeatureStandardizer(2)
    loaded.load_state_dict(state)
    torch.testing.assert_close(loaded(x), transformed)
    for basis in ("bspline", "fourier", "relu", "rbf"):
        t = TransformConfig(kind="kan", basis=basis, basis_kwargs={"num_basis": 9})
        model = GraphKAN(ModelConfig(2, 4, 2, preprocessing=t))
        model.eval()
        path = tmp_path / f"{basis}.pt"
        save_model(model, path)
        restored = load_model(path)
        restored.eval()
        torch.testing.assert_close(model(toy_graph()), restored(toy_graph()), atol=0, rtol=0)


def test_validation_and_eval_restores_mode():
    model, data = GraphKAN(ModelConfig(2, 4, 2)), toy_graph()
    assert model.training
    evaluate(model, data)
    assert model.training
    assert coefficient_penalty(model) == 0
    with pytest.raises(ValueError):
        fit(model, data, mask=torch.zeros(6, dtype=torch.bool))
    with pytest.raises(ValueError):
        fit(model, data, lambda p, y: p, TrainingConfig(epochs=1))
    with pytest.raises(ValueError):
        fit(model, data, lambda p, y: p.sum() * float("nan"), TrainingConfig(epochs=1))
    with pytest.raises(ValueError):
        ModelConfig(0, 4, 2)
    with pytest.raises(ValueError):
        ModelConfig(2, 4, 2, task="bad")
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(RuntimeError):
        FeatureStandardizer(2)(data.x)
    with pytest.raises(ValueError):
        node_split(3)
    link = GraphKAN(ModelConfig(2, 4, 1, task="link"))
    with pytest.raises(ValueError):
        link(data)


def test_cli(tmp_path):
    result = subprocess.run([sys.executable, "-m", "kangnn.cli", "--epochs", "2",
                             "--operator", "gatv2", "--save", str(tmp_path / "demo.pt")],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["seed"] == 7
    assert (tmp_path / "demo.pt").exists()


@pytest.mark.parametrize("basis", ["bspline", "fourier", "relu", "rbf"])
def test_independent_stage_choices_and_gradients(basis):
    kan = TransformConfig(kind="kan", basis=basis, basis_kwargs={"num_basis": 9})
    # One KAN stage at a time, with all others MLP, to verify independence.
    for stage in ("preprocessing", "message", "update", "attention", "readout", "decision", "edge"):
        cfg = ModelConfig(2, 4, 2, readout=TransformConfig(),
                          layers=[LayerConfig(operator="gatv2", update=TransformConfig())])
        data = toy_graph()
        if stage in {"message", "update", "attention", "edge"}:
            setattr(cfg.layers[0], stage, kan)
            if stage == "edge":
                cfg.layers[0].edge_dim = 2
                data.edge_attr = torch.ones(data.edge_index.size(1), 2) * 0.2
        else:
            setattr(cfg, stage, kan)
        model = GraphKAN(cfg)
        nn.functional.cross_entropy(model(data), data.y).backward()
        parameters = [m.coefficients for m in model.modules() if hasattr(m, "coefficients")]
        assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all()
                                  for p in parameters)


def test_cli_in_process(monkeypatch, capsys, tmp_path):
    from kangnn.cli import main
    from pathlib import Path
    config = Path(__file__).parents[1] / "examples" / "mixed.json"
    for options in (["--kind", "mlp", "--loss", "nll"],
                    ["--basis", "fourier"], ["--config", str(config)]):
        monkeypatch.setattr(sys, "argv", ["kangnn", "--epochs", "2", *options,
                                          "--save", str(tmp_path / "model.pt")])
        main()
        assert "evaluation" in json.loads(capsys.readouterr().out)
    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps(ModelConfig(2, 4, 1, task="link").to_dict()))
    monkeypatch.setattr(sys, "argv", ["kangnn", "--config", str(invalid)])
    with pytest.raises(SystemExit):
        main()
